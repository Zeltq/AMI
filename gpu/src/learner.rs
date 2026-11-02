use anyhow::{Result,ensure};
use burn::{module::Module,optim::{AdamConfig,ModuleOptimizer,GradientsParams},
    tensor::{Tensor,Distribution}};
use serde::{Serialize,Deserialize};
use std::{path::Path,time::Instant};
use crate::{experience::{Experience,HORIZON},observe::{Observed,WIDTH},model::Policy,
    sampling::Sampler,world::{World,META},runtime::{device,tensor}};

#[derive(Clone,Serialize,Deserialize,PartialEq,Debug)]
#[serde(default,deny_unknown_fields)]
pub struct Config {
    pub engine:String,pub matches:usize,pub seed:u64,pub ppo_epochs:usize,pub batch_sequences:usize,
    pub learning_rate:f64,pub clip:f32,pub value_coefficient:f32,pub entropy_coefficient:f32,
    pub gradient_norm:f32,pub stop_kl:f32,
    pub initial_checkpoint:Option<String>,pub opponents:Vec<String>,
}
impl Default for Config {
    fn default()->Self {Self {engine:"runs/gpu/dota-calibration-003/engine.json".into(),matches:100,seed:20261030,
        ppo_epochs:4,batch_sequences:16,learning_rate:1e-4,clip:0.2,value_coefficient:0.5,entropy_coefficient:0.005,
        gradient_norm:0.5,stop_kl:0.015,initial_checkpoint:None,opponents:vec![]}}
}

pub struct Learner {pub policy:Policy,pub optimizer:ModuleOptimizer,pub config:Config,pub updates:u64}
#[derive(Clone,Copy,Debug)]
pub(crate) enum UpdateTerms {All,PolicyOnly,ValueOnly,ValueHeadOnly,AllDetachedValue}
impl Learner {
    pub fn new(config:Config)->Self {
        let d=device().autodiff();d.seed(config.seed);
        Self {policy:Policy::new(&d),optimizer:AdamConfig::new().with_epsilon(1e-8).init(),config,updates:0}
    }
    pub fn optimize(&mut self,experience:&Experience,bootstrap:Tensor<1>,sampler:&Sampler)->Result<serde_json::Value>{
        self.optimize_terms(experience,bootstrap,sampler,UpdateTerms::All)
    }
    pub(crate) fn optimize_terms(&mut self,experience:&Experience,bootstrap:Tensor<1>,sampler:&Sampler,terms:UpdateTerms)->Result<serde_json::Value>{
        let cfg=&self.config;let n=experience.actors;let d=device();
        let initial_actor=self.policy.action_head.weight.val().without_autodiff();
        let initial_value=self.policy.value_second.weight.val().without_autodiff();
        let (pc,pt,pp)=experience.buckets();
        let (adv,returns)=experience.gae(bootstrap);
        let signals=experience.signals();
        let valid=signals.clone().slice([0..n,0..HORIZON,11..12]).reshape([n,HORIZON]);
        let count=valid.clone().sum().clamp_min(1.0);
        let mean=(adv.clone()*valid.clone()).sum()/count.clone();
        let variance=((adv.clone()-mean.clone().reshape([1,1])).square()*valid.clone()).sum()/count;
        let adv=(adv-mean.reshape([1,1]))/variance.sqrt().clamp_min(1e-8).reshape([1,1]);
        let mut steps=0;let mut losses=vec![];let mut last_kl=0.;let mut stop=false;
        'epochs:for epoch in 0..cfg.ppo_epochs {
          // Rotate shard order by epoch; shuffle complete actor sequences on
          // device inside each shard. Memory is replayed from its true reset.
          for index in 0..experience.observations.len() {
            let shard_index=(index+epoch)%experience.observations.len();let shard=&experience.observations[shard_index];
            let shard_n=shard.meta.shape()[0];let actor_offset=shard_index*128;
            d.seed(cfg.seed.wrapping_add(self.updates).wrapping_mul(0x9e3779b97f4a7c15).wrapping_add((epoch*experience.observations.len()+shard_index) as u64));
            let order=Tensor::<1>::random([shard_n],Distribution::Uniform(0.,1.),&d).argsort(0);
            for start in (0..shard_n).step_by(cfg.batch_sequences){
                let size=cfg.batch_sequences.min(shard_n-start);let local_ids=order.clone().slice([start..start+size]);let ids=local_ids.clone()+actor_offset as i32;
                let input=tensor::<3>(shard.clone()).select(0,local_ids).reshape([size*HORIZON,WIDTH]);
                let observation=Observed {packed:input,refs:shard.clone(),reset:Tensor::zeros([size*HORIZON],&d)};
                let b=observation.decode_bucket(pc,pt,pp);let encoded=self.policy.encode(&b);
                let projected=self.policy.memory.input.forward(encoded.fused).reshape([size,HORIZON,1024]);
                let signal=signals.clone().select(0,ids.clone());
                let reset=signal.clone().slice([0..size,0..HORIZON,10..11]);
                let mut h=Tensor::<2>::zeros([size,256],&d);let mut c=h.clone();let mut sequence=vec![];
                for t in 0..HORIZON {
                    let keep:Tensor<2>=1.0f32-reset.clone().slice([0..size,t..t+1,0..1]).reshape([size,1]);
                    h=h*keep.clone();c=c*keep;
                    let x=projected.clone().slice([0..size,t..t+1,0..1024]).reshape([size,1024]);
                    (h,c)=self.policy.memory.projected(x,h,c);sequence.push(h.clone());
                }
                let h=Tensor::stack::<3>(sequence,1).reshape([size*HORIZON,256]);
                let actions=signal.clone().slice([0..size,0..HORIZON,0..6]).reshape([size*HORIZON,6]).int();
                let target=actions.clone().slice([0..size*HORIZON,3..4]);
                let shifted=target.clone()-target.clone().greater_equal_elem((2+crate::observe::CREEPS) as i32).int()*(crate::observe::CREEPS-pc) as i32
                    -target.equal_elem((crate::observe::TARGETS-1) as i32).int()*(crate::observe::TREES-pt) as i32;
                let actions=actions.slice_assign([0..size*HORIZON,3..4],shifted);
                let detached_value=if matches!(terms,UpdateTerms::ValueHeadOnly|UpdateTerms::AllDetachedValue){Some(self.policy.value(h.clone().without_autodiff()))}else{None};
                let heads=self.policy.heads(&b,h,encoded.keys,Some(actions),sampler);
                let old=signal.clone().slice([0..size,0..HORIZON,6..7]).reshape([size*HORIZON]);
                let mask=signal.slice([0..size,0..HORIZON,11..12]).reshape([size*HORIZON]);let count=mask.clone().sum().clamp_min(1.);
                let a=adv.clone().select(0,ids.clone()).reshape([size*HORIZON]);
                let target=returns.clone().select(0,ids).reshape([size*HORIZON]);
                let delta=heads.logp-old;let ratio=delta.clone().exp();
                // This KL describes the current policy before the proposed
                // update. Do not apply another gradient once it is too large.
                let kl=((delta.clone().exp()-1.-delta.clone())*mask.clone()).sum()/count.clone();
                last_kl=kl.into_scalar::<f32>();
                ensure!(last_kl.is_finite(),"nonfinite PPO importance ratio");
                if last_kl>cfg.stop_kl {stop=true;break 'epochs;}
                let actor=-(ratio.clone()*a.clone()).min_pair(ratio.clamp(1.-cfg.clip,1.+cfg.clip)*a);
                let critic=(detached_value.unwrap_or(heads.value)-target).square()*cfg.value_coefficient;
                let objective=match terms {
                    UpdateTerms::All|UpdateTerms::AllDetachedValue=>actor+critic-heads.entropy*cfg.entropy_coefficient,
                    UpdateTerms::PolicyOnly=>actor-heads.entropy*cfg.entropy_coefficient,
                    UpdateTerms::ValueOnly|UpdateTerms::ValueHeadOnly=>critic,
                };
                let loss=(objective*mask.clone()).sum()/count.clone();
                let value=loss.clone().without_autodiff().into_scalar::<f32>();
                ensure!(value.is_finite(),"nonfinite PPO loss");
                let mut grads=GradientsParams::from_grads(loss.backward(),&self.policy);
                let _norm=crate::gradients::clip(&self.policy,&mut grads,cfg.gradient_norm);
                self.policy=self.optimizer.step(cfg.learning_rate,self.policy.clone(),grads);self.updates+=1;steps+=1;
                // Only aggregate scalars cross the boundary per minibatch.
                losses.push(value);
            }
          }
        }
        let actor_delta=(self.policy.action_head.weight.val().without_autodiff()-initial_actor).square().sum().sqrt().into_scalar::<f32>();
        let value_delta=(self.policy.value_second.weight.val().without_autodiff()-initial_value).square().sum().sqrt().into_scalar::<f32>();
        Ok(serde_json::json!({"optimizer_steps":steps,"total_optimizer_steps":self.updates,"last_kl":last_kl,"action_head_delta":actor_delta,"value_head_delta":value_delta,"update_terms":format!("{terms:?}"),
            "mean_loss":losses.iter().sum::<f32>()/losses.len().max(1) as f32,"early_stop":stop,"buckets":[pc,pt,pp]}))
    }
}

pub fn train(config:Config,cycles:usize)->Result<serde_json::Value>{run(config,cycles,None,false,true)}

pub fn run(config:Config,cycles:usize,output:Option<&Path>,resume:bool,allow_draft:bool)->Result<serde_json::Value>{
    run_terms(config,cycles,output,resume,allow_draft,UpdateTerms::All,false)
}
pub fn run_audited(config:Config,cycles:usize,output:&Path,resume:bool,allow_draft:bool)->Result<serde_json::Value>{
    ensure!(allow_draft&&cycles>0&&cycles<=2,"rollout audit requires draft and1..2 bounded cycles");
    run_terms(config,cycles,Some(output),resume,allow_draft,UpdateTerms::All,true)
}
pub fn run_loss_diagnostic(config:Config,cycles:usize,output:&Path,resume:bool,allow_draft:bool,label:&str)->Result<serde_json::Value>{
    ensure!(allow_draft,"loss-term diagnostic requires --draft");
    let terms=match label {"policy-only"=>UpdateTerms::PolicyOnly,"value-only"=>UpdateTerms::ValueOnly,"value-head-only"=>UpdateTerms::ValueHeadOnly,"all-detached-value"=>UpdateTerms::AllDetachedValue,_=>anyhow::bail!("unknown diagnostic loss terms")};
    run_terms(config,cycles,Some(output),resume,allow_draft,terms,false)
}
fn run_terms(config:Config,cycles:usize,output:Option<&Path>,resume:bool,allow_draft:bool,terms:UpdateTerms,audit_rollouts:bool)->Result<serde_json::Value>{
    use crate::artifacts::{Writer,Checkpoint,atomic_json};
    use std::sync::{Arc,atomic::{AtomicBool,Ordering}};
    ensure!(config.batch_sequences>0 && config.ppo_epochs>0,"empty training configuration");
    let _lock=output.map(|root|->Result<std::fs::File>{
        std::fs::create_dir_all(root)?;let mut file=std::fs::OpenOptions::new().create(true).truncate(false).read(true).write(true).open(root.join("run.lock"))?;
        file.try_lock().map_err(|e|anyhow::anyhow!("run already locked: {e}"))?;
        use std::io::Write;file.set_len(0)?;writeln!(file,"{}",std::process::id())?;Ok(file)
    }).transpose()?;
    let engine=if let Some(root)=output {crate::artifacts::freeze_environment(Path::new(&config.engine),root,resume)?}else{Path::new(&config.engine).to_path_buf()};
    let d=device();let mut world=World::load(&engine,config.matches,&d)?;
    ensure!(world.engine.behavior_validated || allow_draft,"Dota scenario validation has not passed; use --draft only for development measurements");
    let n=config.matches*2;let mut sample=Sampler::new(n,config.seed as u32,&d);let mut rng=Sampler::new(config.matches,config.seed as u32+1,&d);
    let mut learner=Learner::new(config.clone());let experience=Experience::new(n,&d);
    if !resume {if let Some(path)=config.initial_checkpoint.as_ref(){learner.policy=crate::selfplay::load(Path::new(path),&d)?.train();}}
    let opponents=crate::selfplay::Pool::new(&config.opponents,config.matches,&d)?;
    let mut previous=crate::observe::initial_previous(n,&d);let mut history=std::collections::VecDeque::new();
    let mut start_cycle=0;
    if let Some(root)=output {
        if resume {
            if root.join("stop.json").exists(){std::fs::rename(root.join("stop.json"),root.join("last-stop.json"))?;}
            let mut p=crate::runtime::primitive(previous);
            let (policy,optimizer,state)=crate::artifacts::restore(root,learner.policy,learner.optimizer,&mut world,&mut p,&mut sample,&mut rng)?;
            let saved:Config=serde_json::from_value(state["config"].clone())?;
            ensure!(saved==config,"resume configuration differs: saved {saved:?}, requested {config:?}");
            ensure!(state.get("update_terms").and_then(|v|v.as_str()).unwrap_or("All")==format!("{terms:?}"),"resume loss terms differ");
            ensure!(state["opponent_sha256"]==serde_json::to_value(&opponents.identities)?,"saved opponent weights changed since checkpoint");
            learner.policy=policy.train();learner.optimizer=optimizer;learner.updates=state["optimizer_updates"].as_u64().unwrap();
            start_cycle=state["cycle"].as_u64().unwrap() as usize;previous=tensor(p);
        } else {ensure!(!root.join("latest.json").exists(),"existing run requires --resume");}
        atomic_json(&root.join("config.json"),&config)?;
        let executable=std::env::current_exe()?;
        let manifest=serde_json::json!({"burn":"0.22.1","cubecl":"0.11.1","cargo_lock_sha256":crate::runtime::bytes_sha256(include_bytes!("../Cargo.lock")),
            "rust_toolchain":include_str!("../rust-toolchain.toml"),"device":format!("{d:?}"),"device_identity":format!("{:?}",d.identity()),
            "os":std::env::consts::OS,"architecture":std::env::consts::ARCH,"executable_sha256":crate::runtime::file_sha256(&executable)?,
            "environment_sha256":world.fingerprint,"seed":config.seed,"opponent_sha256":opponents.identities,"config":config});
        atomic_json(&root.join(if resume{"resume-manifest.json"}else{"manifest.json"}),&manifest)?;
        if !resume {
            let frozen=root.join(if cfg!(windows){"sf-duel-gpu.exe"}else{"sf-duel-gpu"});
            if executable.canonicalize()?!=frozen.canonicalize().unwrap_or_else(|_|frozen.clone()){std::fs::copy(&executable,frozen)?;}
        }
    }
    ensure!(!audit_rollouts||start_cycle+cycles<=2,"audited counterexample stops after two total cycles");
    let stopped=Arc::new(AtomicBool::new(false));
    if output.is_some(){let flag=stopped.clone();ctrlc::set_handler(move||flag.store(true,Ordering::Release))?;}
    let writer=output.map(Writer::new).transpose()?;let trace=writer.as_ref().map(|_|crate::trace::Trace::new(&world));
    let started=Instant::now();let mut total_collection=0.;let mut total_training=0.;let mut completed_cycles=0;
    let mut last_snapshot=Instant::now();let mut last_snapshot_cycle=usize::MAX;
    let initial_ticks=tensor::<1>(world.meta.clone()).reshape([config.matches,META]).slice([0..config.matches,3..4]).sum().into_scalar::<f32>();
    let initial_completed=tensor::<1>(world.meta.clone()).reshape([config.matches,META]).slice([0..config.matches,4..5]).sum().into_scalar::<f32>();
    let initial_updates=learner.updates;
    let initial_outcomes=tensor::<1>(world.meta.clone()).reshape([config.matches,META]).slice([0..config.matches,18..32]).sum_dim(0)+0.0f32;
    let mut previous_ticks=tensor::<1>(world.meta.clone()).reshape([config.matches,META]).slice([0..config.matches,3..4]).reshape([config.matches])+0.0f32;
    let status=|phase:&str,cycle:usize,writer:Option<&Writer>|->Result<()> {
        if let Some(root)=output {atomic_json(&root.join("status.json"),&serde_json::json!({"phase":phase,"cycle":cycle,
            "pid":std::process::id(),"elapsed_seconds":started.elapsed().as_secs_f64(),"matches":config.matches,
            "artifacts":writer.map(Writer::status),"behavior_validated":world.engine.behavior_validated}))?;}Ok(())
    };
    for cycle in start_cycle.. {
        if (cycles>0 && completed_cycles>=cycles) || stopped.load(Ordering::Acquire) || output.is_some_and(|p|p.join("stop.json").exists()){break;}
        if let Some(writer)=writer.as_ref(){writer.check()?;}status("collecting",cycle,writer.as_ref())?;
        let t=Instant::now();let (bootstrap,collection_profile)=d.profile(||{
        let actor=learner.policy.clone().valid();let mut h=Tensor::<2>::zeros([n,256],&d);let mut c=h.clone();
        for step in 0..HORIZON {
            let mut observed=world.observe(previous.clone());observed.packed=observed.packed.cast(burn::tensor::DType::F16);
            let keep:Tensor<2>=1.0f32-observed.reset.clone().unsqueeze_dim::<2>(1);h=h*keep.clone();c=c*keep;
            let prior_h=h.clone();let prior_c=c.clone();
            let b=observed.decode();let encoded=actor.encode(&b);(h,c)=actor.memory.step(encoded.fused,h,c);
            let heads=actor.heads(&b,h.clone(),encoded.keys,None,&sample);
            let (next_h,next_c,heads,learn,versions)=opponents.decide(&world,&b,prior_h,prior_c,h,c,heads,&sample,cycle+1);h=next_h;c=next_c;
            let actions=heads.actions.clone().float();
            let next_previous=observed.previous(heads.actions.clone(),&world.bases,&world.state);
            let (reward,done)=world.advance(heads.actions,&observed.refs,&rng);
            previous=next_previous.mask_where(done.clone().greater_elem(0.5).unsqueeze_dim::<2>(1),crate::observe::initial_previous(n,&d));
            let signals=Tensor::cat(vec![actions,heads.logp.unsqueeze_dim::<2>(1),heads.value.unsqueeze_dim::<2>(1),
                reward.unsqueeze_dim::<2>(1),done.unsqueeze_dim::<2>(1),observed.reset.clone().unsqueeze_dim::<2>(1),learn,versions],1);
            if !config.opponents.is_empty(){crate::selfplay::record_outcomes(&world,signals.clone());}
            experience.store(step,&observed,signals.clone());
            if let Some(trace)=trace.as_ref(){trace.record(&world,step,signals);}
        }
        let mut next=world.observe(previous.clone());next.packed=next.packed.cast(burn::tensor::DType::F16);let b=next.decode();let encoded=actor.encode(&b);
        // The following action is at the next periodic memory reset.
        let (next_h,_)=actor.memory.step(encoded.fused,Tensor::zeros([n,256],&d),Tensor::zeros([n,256],&d));
        actor.value(next_h)
        })?;d.sync()?;let collection=t.elapsed().as_secs_f64();
        if audit_rollouts{crate::startup_audit::save_rollout(&experience,bootstrap.clone(),&output.unwrap().join(format!("rollout-{:02}",cycle+1)))?;}
        status("updating_policy",cycle,writer.as_ref())?;
        let t=Instant::now();let (learning,training_profile)=d.profile(||learner.optimize_terms(&experience,bootstrap,&sample,terms))?;let learning=learning?;d.sync()?;let training=t.elapsed().as_secs_f64();
        let meta=tensor::<1>(world.meta.clone()).reshape([config.matches,META]);
        ensure!(meta.clone().slice([0..config.matches,7..8]).max().into_scalar::<f32>()==0.,"environment capacity/error flag");
        total_collection+=collection;total_training+=training;completed_cycles+=1;
        let ticks=meta.clone().slice([0..config.matches,3..4]).reshape([config.matches])+0.0f32;
        let speeds=(ticks.clone()-previous_ticks)/30.0f32/collection as f32;previous_ticks=ticks;
        let mean_speed=speeds.clone().mean();let speed_std=(speeds.clone()-mean_speed.clone()).square().mean().sqrt().into_scalar::<f32>();
        let competition=crate::selfplay::outcome_report((meta.clone().slice([0..config.matches,18..32]).sum_dim(0)-initial_outcomes.clone()).into_data().try_to_vec::<f32>()?);
        let completed=meta.slice([0..config.matches,4..5]).sum().into_scalar::<f32>()-initial_completed;
        // These are counts, not model/state floats. Fail before another update
        // if a future layout error aliases them with fractional simulation data.
        for key in ["games","wins","losses","draws"]{
            let value=competition[key].as_f64().unwrap_or(f64::NAN);
            ensure!(value.is_finite()&&value>=0.&&value.fract()==0.&&value<=completed as f64,"invalid selfplay count {key}={value}, completed={completed}");
        }
        let row=serde_json::json!({"cycle":cycle+1,"collection_seconds":collection,"training_seconds":training,"learning":learning,
            "gpu_profile":{"collection":crate::runtime::profile_value(collection_profile),"ppo":crate::runtime::profile_value(training_profile)},
            "collection_per_match_speed":mean_speed.into_scalar::<f32>(),"collection_match_speed_std":speed_std,
            "collection_match_speed_min":speeds.clone().min().into_scalar::<f32>(),"collection_match_speed_max":speeds.max().into_scalar::<f32>(),
            "completed":completed,"training_vs_frozen":competition,"memory_pool":format!("{:?}",d.memory_pool_usage()),
            "artifacts":writer.as_ref().map(Writer::status)});
        if let Some(writer)=writer.as_ref(){
            writer.metrics(&row)?;
            if let Some(trace)=trace.as_ref(){writer.trace(&trace.buffer,trace.header(&world,cycle as u64+1),cycle as u64+1)?;}
            if last_snapshot.elapsed().as_secs()>=300 && writer.reserve_checkpoint(false)? {
                d.memory_cleanup();
                let metadata=serde_json::json!({"cycle":cycle+1,"optimizer_updates":learner.updates,"matches":config.matches,
                    "config":config,"game_build":world.engine.build,"recurrent_reset_interval":128,"observation_storage":"f16",
                    "state_schema":3,"environment_sha256":world.fingerprint,"opponent_sha256":opponents.identities,"update_terms":format!("{terms:?}")});
                writer.checkpoint(Checkpoint::capture(&learner.policy,&learner.optimizer,&world,&crate::runtime::primitive(previous.clone()),&sample,&rng,metadata),false)?;
                last_snapshot=Instant::now();last_snapshot_cycle=cycle+1;
            }
        }
        eprintln!("{}",row);history.push_back(row);if history.len()>64{history.pop_front();}
    }
    // No rollout remains trainable at this boundary. Release it before the
    // final snapshot so export has VRAM headroom even at the largest batch.
    drop(experience);d.memory_cleanup();
    if let Some(writer)=writer.as_ref(){
        if last_snapshot_cycle!=start_cycle+completed_cycles && writer.reserve_checkpoint(true)? {
            let metadata=serde_json::json!({"cycle":start_cycle+completed_cycles,"optimizer_updates":learner.updates,"matches":config.matches,
                "config":config,"game_build":world.engine.build,"recurrent_reset_interval":128,"observation_storage":"f16",
                "state_schema":3,"environment_sha256":world.fingerprint,"opponent_sha256":opponents.identities,"update_terms":format!("{terms:?}")});
            writer.checkpoint(Checkpoint::capture(&learner.policy,&learner.optimizer,&world,&crate::runtime::primitive(previous.clone()),&sample,&rng,metadata),true)?;
        }
    }
    status("draining_artifacts",start_cycle+completed_cycles,writer.as_ref())?;
    if let Some(writer)=writer{writer.finish()?;}
    let wall=started.elapsed().as_secs_f64();let meta=tensor::<1>(world.meta.clone()).reshape([config.matches,META]);
    let advanced=(meta.clone().slice([0..config.matches,3..4]).sum().into_scalar::<f32>()-initial_ticks) as f64/30.;
    let competition=crate::selfplay::outcome_report((meta.clone().slice([0..config.matches,18..32]).sum_dim(0)-initial_outcomes).into_data().try_to_vec::<f32>()?);
    let completed=meta.slice([0..config.matches,4..5]).sum().into_scalar::<f32>()-initial_completed;
    let report=serde_json::json!({"matches":config.matches,"cycles":completed_cycles,"optimizer_steps":learner.updates-initial_updates,
        "wall_seconds":wall,"per_match_speed":advanced/config.matches as f64/wall,
        "collection_per_match_speed":advanced/config.matches as f64/total_collection.max(1e-9),
        "collection_seconds":total_collection,"training_seconds":total_training,
        "observations_per_second":(completed_cycles*HORIZON*n) as f64/wall,"completed_matches_per_hour":completed as f64*3600./wall,
        "updates_per_second":(learner.updates-initial_updates) as f64/wall,"history":history,"completed":completed,"training_vs_frozen":competition,
        "behavior_validated":world.engine.behavior_validated,"decision_hz":7.5,"simulation_hz":30});
    if let Some(root)=output{atomic_json(&root.join("result.json"),&report)?;atomic_json(&root.join("status.json"),&serde_json::json!({"phase":"stopped","cycle":start_cycle+completed_cycles,"report":"result.json"}))?;}
    Ok(report)
}


