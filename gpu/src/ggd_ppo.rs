//! Recurrent GGD PPO against a configured fixed teacher. Physics and observation
//! construction are shared with evaluate-ggd; P08 and potential rewards coexist.
use anyhow::{ensure, Result};
use burn::{module::Module, optim::{AdamConfig, GradientsParams, ModuleOptimizer}, tensor::{Tensor, TensorData, Int, DType}};
use cubecl::prelude::*;
use serde::{Serialize, Deserialize};
use std::{path::Path, time::Instant, io::Write};
use crate::{experience::{Experience,HORIZON}, model::Policy, observe::Observed,
    runtime::{device,tensor,primitive,argument}, sampling::Sampler, world::*, contract::*};

pub const COMPONENTS:usize=9;
pub const NAMES:[&str;COMPONENTS]=["lh","deny","opponent_lh","opponent_deny","death","opponent_death","tower_loss","tower_kill","timeout_result"];
#[derive(Clone,Serialize,Deserialize,Debug)]
#[serde(default,deny_unknown_fields)]
pub struct Config {
    pub engine:String,pub initial_weights:String,pub matches:usize,pub seed:u32,
    pub minutes:f64,pub max_blocks:usize,pub batch_sequences:usize,pub epochs:usize,
    pub learning_rate:f64,pub critic_learning_rate:f64,pub warmup_blocks:usize,
    pub entropy:f32,pub kl_start:f32,pub kl_end:f32,pub stop_kl:f32,
    pub checkpoint_minutes:f64,pub evaluate:bool,pub evaluation_minutes:f64,
    pub initialize_weights:Option<String>,pub skip_warmup:bool,pub critic_only:bool,pub desynchronize:bool,pub verify_phase_histogram:bool,
    pub teacher_version:crate::teacher_v10::Version,pub reward_mode:String,pub mana_weight:f32,
    pub warmup_min_blocks:usize,pub separate_critic_lr:bool,pub resume_checkpoint:Option<String>,
    pub schedule_minutes:f64,pub stop_at_minutes:f64,
}
impl Default for Config {fn default()->Self{Self{
    engine:"C:/AMI/runs/gpu/ggd0-il-20261102-202637/dataset/environment/engine.json".into(),
    initial_weights:"C:/AMI/runs/p02/experiment/train-round-2-attempt-1/export/weights".into(),
    matches:16,seed:20261103,minutes:20.,max_blocks:0,batch_sequences:4,epochs:2,
    learning_rate:1e-5,critic_learning_rate:1e-4,warmup_blocks:32,entropy:0.005,
    kl_start:0.2,kl_end:0.05,stop_kl:0.02,checkpoint_minutes:10.,evaluate:false,evaluation_minutes:60.,
    initialize_weights:None,skip_warmup:false,critic_only:false,desynchronize:false,verify_phase_histogram:false,
    teacher_version:crate::teacher_v10::Version::V9,reward_mode:"p08".into(),mana_weight:0.5,warmup_min_blocks:30,separate_critic_lr:false,resume_checkpoint:None,schedule_minutes:480.,stop_at_minutes:0.,
}}}

// Cumulative components: differencing these after the same four physics ticks
// captures simultaneous events, pregame deaths and exactly one timeout result.
#[cube(launch)]
fn counters(state:&[f32],meta:&[f32],audit:&[f32],out:&mut [f32],actors:usize){
    let a=ABSOLUTE_POS;if a<actors {
        let m=a/2;let s=a%2;let base=m*UNITS*STRIDE;let p=base+s*STRIDE;let q=base+(1-s)*STRIDE;let r=a*(COMPONENTS+1);
        out[r]=state[p+UNIT_LAST_HITS];out[r+1]=state[p+UNIT_DENIES]*0.5f32;
        out[r+2]=-state[q+UNIT_LAST_HITS]*0.5f32;out[r+3]=-state[q+UNIT_DENIES]*0.25f32;
        out[r+4]=-state[p+DEATHS]*10.0f32;out[r+5]=state[q+DEATHS]*10.0f32;
        out[r+6]=if state[base+(2+s*2)*STRIDE+UNIT_HEALTH]<=0.0f32{-10.0f32}else{0.0f32};
        out[r+7]=if state[base+(2+(1-s)*2)*STRIDE+UNIT_HEALTH]<=0.0f32{10.0f32}else{0.0f32};
        out[r+8]=if meta[m*META]/30.0f32>=600.0f32 {meta[m*META+17]*if s==0{5.0f32}else{-5.0f32}}else{0.0f32};
        out[r+9]=audit[m*crate::scripted_teacher::AUDIT_FIELDS];
    }
}
fn counts(w:&World,t:&crate::scripted_teacher::Teacher)->Tensor<2>{
    let out=primitive(Tensor::<2>::zeros([w.matches*2,COMPONENTS+1],&w.device));
    counters::launch(&w.state.client,CubeCount::Static((w.matches*2).div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&w.state),argument(&w.meta),argument(&t.audit),argument(&out),w.matches*2);
    tensor(out)
}
fn observation(w:&World,teacher:&crate::scripted_teacher::Teacher,observer:&crate::combat_observe::CombatObserver,enemy:&crate::enemy_observe::Observer,controller:&crate::teacher_v10::Controller,previous:Tensor<2>,manifest:&serde_json::Value)->(Observed,Tensor<2,Int>){
    let obs=w.observe(previous);let mut extended=observer.enrich(w,&obs);
    let scripted=controller.actions(teacher,w,&obs);
    if manifest["observation_profile"]=="ggd-v3"||manifest["observation_profile"]=="ggd-v4" {
        extended=crate::timing_observe::extend(w,extended,teacher,manifest["timing_honest"].as_bool().unwrap_or(true),manifest["timing_oracle"].as_bool().unwrap_or(false));
    }
    if manifest["observation_profile"]=="ggd-v4" {extended=enemy.extend(w,extended);}
    extended.packed=extended.packed.cast(DType::F16);(extended,scripted)
}
/// Stagger initial activation on recurrent boundaries, then independently reset
/// terminal arenas at boundaries. Only native simulation steps create the states.
fn disperse(w:&World,t:&crate::scripted_teacher::Teacher,o:&crate::combat_observe::CombatObserver,enemy:&crate::enemy_observe::Observer,controller:&crate::teacher_v10::Controller,metrics:&crate::ggd_metrics::Metrics,rng:&Sampler,sampler:&Sampler,actor:&Policy,manifest:&serde_json::Value,root:&Path,seed:u32,sides:&mut [usize])->Result<(Tensor<2>,serde_json::Value)>{
    let d=&w.device;let n=w.matches*2;let blocks=((w.engine.pre_game_seconds+600.)*7.5/HORIZON as f32).floor()as usize;
    let delays=crate::ggd_arenas::delays(w.matches,blocks,seed);let mut active=vec![false;w.matches];let started=Instant::now();
    let mut previous=crate::observe::initial_previous(n,d);let mut decisions=Tensor::<1>::zeros([1],d);let mut completed=0;
    crate::ggd_arenas::hold_all(t,w.matches);
    for block in 0..=blocks {
        let activate:Vec<_>=delays.iter().enumerate().map(|(m,b)|!active[m]&&*b==block).collect();
        if activate.iter().any(|x|*x){let mask=crate::ggd_arenas::reset(w,t,metrics,&activate);previous=previous.mask_where(mask.unsqueeze_dim::<2>(1),crate::observe::initial_previous(n,d));for m in 0..w.matches{active[m]|=activate[m];}}
        if block==blocks{break;}
        let rows=crate::ggd_arenas::learner_rows(sides,d);let mut h=Tensor::<2>::zeros([n,256],d);let mut c=h.clone();
        for _ in 0..HORIZON {
            let snapshot=(manifest["observation_profile"]!="ggd-v2").then(||t.command_snapshot());
            let (obs,scripted)=observation(w,t,o,enemy,controller,previous.clone(),manifest);let keep:Tensor<2>=1.0f32-obs.reset.clone().unsqueeze_dim::<2>(1);h=h*keep.clone();c=c*keep;
            let b=obs.decode();let encoded=actor.encode(&b);(h,c)=actor.memory.step(encoded.fused,h,c);let predicted=actor.heads(&b,h.clone(),encoded.keys,None,sampler).actions;
            let actions=scripted.clone().mask_where(rows.clone().unsqueeze_dim::<2>(1),predicted);
            if let Some(snapshot)=snapshot{t.reconcile_commands(w,&obs,snapshot,scripted,actions.clone());}
            decisions=decisions+(1.0f32-tensor::<1>(t.audit.clone()).reshape([w.matches,crate::scripted_teacher::AUDIT_FIELDS]).slice([0..w.matches,0..1])).sum();
            previous=obs.previous(actions.clone(),&w.bases,&w.state);metrics.commands(w,t,actions.clone(),&obs.refs);
            let _=w.advance_teacher(actions,&obs.refs,rng,t,true);t.record(w,600.);metrics.sample(w,t);
        }
        let audit=t.values();ensure!(audit.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).all(|x|x[39]==0.),"initialization simulation error");
        let done:Vec<_>=audit.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).enumerate().map(|(m,x)|active[m]&&x[0]>0.).collect();
        if done.iter().any(|x|*x){completed+=done.iter().filter(|x|**x).count();let mask=crate::ggd_arenas::reset(w,t,metrics,&done);previous=previous.mask_where(mask.unsqueeze_dim::<2>(1),crate::observe::initial_previous(n,d));for m in 0..w.matches{if done[m]{sides[m]=1-sides[m];}}}
        crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"stage":"desynchronizing","pid":std::process::id(),"blocks":block+1,"total_blocks":blocks,"wall_seconds":started.elapsed().as_secs_f64(),"decisions":scalar(decisions.clone()),"completed_initialization_episodes":completed}))?;
        ensure!(!root.join("stop.json").exists(),"stopped during phase initialization");
    }
    d.sync()?;let summary=serde_json::json!({"seconds":started.elapsed().as_secs_f64(),"learner_decisions":scalar(decisions),"completed_episodes":completed,"activation_delay_blocks":delays,"recurrent_block_decisions":HORIZON,"recorded_in_rollout":false,"optimizer_updates":0});
    crate::artifacts::atomic_json(&root.join("desynchronization.json"),&summary)?;Ok((previous,summary))
}
fn recurrent(policy:&Policy,b:&crate::model::Observation,resets:Tensor<3>,size:usize)->(Tensor<2>,Tensor<3>){
    let encoded=policy.encode(b);let projected=policy.memory.input.forward(encoded.fused).reshape([size,HORIZON,1024]);
    let mut h=Tensor::<2>::zeros([size,256],&projected.device());let mut c=h.clone();let mut sequence=vec![];
    for t in 0..HORIZON {
        let keep:Tensor<2>=1.0f32-resets.clone().slice([0..size,t..t+1,0..1]).reshape([size,1]);h=h*keep.clone();c=c*keep;
        (h,c)=policy.memory.projected(projected.clone().slice([0..size,t..t+1,0..1024]).reshape([size,1024]),h,c);sequence.push(h.clone());
    }
    (Tensor::stack::<3>(sequence,1).reshape([size*HORIZON,256]),encoded.keys)
}
#[derive(Default)]
struct CriticWarmup {minimum:usize,post_blocks:usize,recent:std::collections::VecDeque<Option<f64>>}
impl CriticWarmup {
    fn observe(&mut self,block:usize,limit:usize,post_count:f64,ev:Option<f64>,reward_activity:f32)->bool {
        if post_count>0. {self.post_blocks+=1;self.recent.push_back(ev);if self.recent.len()>5{self.recent.pop_front();}}
        block>=limit || (reward_activity>0. && self.post_blocks>=if self.minimum==0{30}else{self.minimum} && self.mean().is_some_and(|x|x>0.3))
    }
    fn mean(&self)->Option<f64>{
        if self.recent.len()!=5 || self.recent.iter().any(Option::is_none){return None;}
        Some(self.recent.iter().flatten().sum::<f64>()/5.)
    }
}
fn explained_variance(target:Tensor<1>,prediction:Tensor<1>,mask:Tensor<1>)->Option<f64>{
    if scalar(mask.clone().sum())<2. {return None;}
    let var=scalar(variance(target.clone(),mask.clone()));
    if var<=1e-8{return None;}
    let ev=1.-scalar(variance(target-prediction,mask))/var;
    ev.is_finite().then_some(ev as f64)
}
fn pregame_death_events(components:Tensor<2>,respawn_at:Tensor<1>,valid:Tensor<1>,pregame:f32)->Tensor<1>{
    let n=components.dims()[0];
    // Physics schedules every hero respawn at death_elapsed + 12 seconds.
    // Classify the event timestamp, including a four-tick step across 0:00.
    ((-components.slice([0..n,4..5]).reshape([n])/10.).clamp_min(0.)
        *(respawn_at-12.-pregame).lower_elem(0.).float()*valid).sum()
}
fn weighted(x:Tensor<1>,mask:Tensor<1>)->Tensor<1>{(x*mask.clone()).sum()/mask.sum().clamp_min(1.)}
fn scalar(x:Tensor<1>)->f32{x.without_autodiff().into_scalar::<f32>()}
fn variance(x:Tensor<1>,mask:Tensor<1>)->Tensor<1>{let mean=weighted(x.clone(),mask.clone());weighted((x-mean).square(),mask)}

fn take_value_gradients(policy:&Policy,grads:&mut GradientsParams)->GradientsParams {
    use burn::module::{ModuleVisitor,Param};
    struct Take<'a>{source:&'a mut GradientsParams,target:GradientsParams}
    impl ModuleVisitor for Take<'_>{fn visit_float<const D:usize>(&mut self,p:&Param<Tensor<D>>){if let Some(g)=self.source.remove::<D>(p.id){self.target.register(p.id,g);}}}
    let mut take=Take{source:grads,target:GradientsParams::new()};policy.value_first.visit(&mut take);policy.value_second.visit(&mut take);take.target
}

fn sequence_order(shard_n:usize,offset:usize,active:Option<&[f32]>,rotation:usize)->Vec<i32>{
    let mut ids:Vec<i32>=(0..shard_n).filter(|i|active.is_none_or(|a|a[offset+*i]>0.)).map(|i|i as i32).collect();
    if !ids.is_empty(){let count=ids.len();ids.rotate_left(rotation%count);}ids
}

fn optimize(policy:&mut Policy,optimizer:&mut ModuleOptimizer,cfg:&Config,e:&Experience,bootstrap:Tensor<1>,warmup:bool,coefficient:f32,block:usize,postgame:Tensor<2>)->Result<serde_json::Value>{
    let d=device();let n=e.actors;let signals=e.signals();let (advantages,returns)=e.gae_discounted(bootstrap,0.99,0.95);
    let mask=signals.clone().slice([0..n,0..HORIZON,11..12]).reshape([n*HORIZON]);
    let raw=advantages.reshape([n*HORIZON]);let mean=weighted(raw.clone(),mask.clone());let sd=variance(raw.clone(),mask.clone()).sqrt().clamp_min(1e-8);
    let adv=((raw-mean)/sd).reshape([n,HORIZON]);
    let old_value=signals.clone().slice([0..n,0..HORIZON,7..8]).reshape([n*HORIZON]);let ret=returns.clone().reshape([n*HORIZON]);
    let ev=explained_variance(ret.clone(),old_value.clone(),mask.clone());
    let phase=postgame.reshape([n*HORIZON]);let post_mask=mask.clone()*phase.clone();let pre_mask=mask.clone()*(1.0f32-phase);
    let ev_post=explained_variance(ret.clone(),old_value.clone(),post_mask.clone());
    let ev_pre=explained_variance(ret,old_value,pre_mask.clone());
    let post_count=scalar(post_mask.sum());let pre_count=scalar(pre_mask.sum());
    let active=if cfg.reward_mode!="p08"{Some(signals.clone().slice([0..n,0..HORIZON,11..12]).sum_dim(1).reshape([n]).into_data().try_to_vec::<f32>()?)}else{None};
    let (pc,pt,pp)=e.buckets();let mut values=Vec::new();let mut stopped=false;let mut offset;
    for epoch in 0..cfg.epochs {
        offset=0;
        for shard in &e.observations {
            let shard_n=shard.meta.shape()[0];let width=shard.meta.shape()[2];
            // Deterministic cyclic sequence permutation, with full recurrent resets.
            let ids=sequence_order(shard_n,offset,active.as_deref(),epoch+block);let sequences=ids.len();
            if sequences==0{offset+=shard_n;continue;}
            let order=Tensor::<1,Int>::from_data(TensorData::new(ids,[sequences]),&d);
            for start in (0..sequences).step_by(cfg.batch_sequences){
                let size=cfg.batch_sequences.min(sequences-start);let local=order.clone().slice([start..start+size]);let ids=local.clone()+offset as i32;
                let signal=signals.clone().select(0,ids.clone());let mask=signal.clone().slice([0..size,0..HORIZON,11..12]).reshape([size*HORIZON]);
                if scalar(mask.clone().sum())==0. {continue;}
                let input=tensor::<3>(shard.clone()).select(0,local).reshape([size*HORIZON,width]);
                let obs=Observed{packed:input,refs:shard.clone(),reset:Tensor::zeros([size*HORIZON],&d)};let b=obs.decode_bucket(pc,pt,pp);
                // Value-only warmup must not build an unused encoder/LSTM tape.
                // Merely detaching its output retains large autodiff intermediates.
                let frozen=warmup.then(||policy.clone().valid());
                let (h,keys)=recurrent(frozen.as_ref().unwrap_or(policy),&b,signal.clone().slice([0..size,0..HORIZON,10..11]),size);
                let target=returns.clone().select(0,ids.clone()).reshape([size*HORIZON]);
                if warmup {
                    let prediction=policy.value(h.without_autodiff());let loss=weighted((prediction-target).square(),mask);
                    let vl=scalar(loss.clone());ensure!(vl.is_finite(),"nonfinite warmup loss");
                    let mut grads=GradientsParams::from_grads(loss.backward(),policy);crate::gradients::clip(policy,&mut grads,0.5);
                    *policy=optimizer.step(if cfg.learning_rate==0.{0.}else{cfg.critic_learning_rate},policy.clone(),grads);
                    values.push([vl,0.,0.,0.,0.]);continue;
                }
                let actions=signal.clone().slice([0..size,0..HORIZON,0..6]).reshape([size*HORIZON,6]).int();
                let target_index=actions.clone().slice([0..size*HORIZON,3..4]);
                let shifted=target_index.clone()-target_index.clone().greater_equal_elem((2+crate::observe::CREEPS)as i32).int()*(crate::observe::CREEPS-pc)as i32-target_index.equal_elem((crate::observe::TARGETS-1)as i32).int()*(crate::observe::TREES-pt)as i32;
                let actions=actions.slice_assign([0..size*HORIZON,3..4],shifted);
                let sampler=Sampler::new(size*HORIZON,7,&d);let heads=policy.heads(&b,h,keys,Some(actions),&sampler);
                let old=signal.clone().slice([0..size,0..HORIZON,6..7]).reshape([size*HORIZON]);
                let reference=signal.clone().slice([0..size,0..HORIZON,12..13]).reshape([size*HORIZON]);
                let delta=heads.logp.clone()-old;let ratio=delta.clone().exp();
                let old_kl=scalar(weighted(ratio.clone()-1.-delta.clone(),mask.clone()));
                ensure!(old_kl.is_finite(),"nonfinite PPO KL");if old_kl>cfg.stop_kl{stopped=true;break;}
                let a=adv.clone().select(0,ids).reshape([size*HORIZON]);
                let actor=-(ratio.clone()*a.clone()).min_pair(ratio.clone().clamp(0.8,1.2)*a);
                // Importance-weighted nonnegative estimator of KL(current || M2).
                // E_old[rho*(exp(-d)-1+d)] = KL(current || M2), d=log(current/M2).
                // Keep rho in the gradient: detaching it would change the objective.
                let diff=heads.logp-reference;let anchor=ratio.clone()*((-diff.clone()).exp()-1.+diff);
                let vl=(heads.value-target).square();let entropy=heads.entropy;
                let clipped=ratio.clone().lower_elem(0.8).bool_or(ratio.greater_elem(1.2));
                let clip=Tensor::<1>::zeros([size*HORIZON],&d).mask_fill(clipped,1.);
                let row=[scalar(weighted(vl.clone(),mask.clone())),scalar(weighted(anchor.clone(),mask.clone())),scalar(weighted(entropy.clone(),mask.clone())),scalar(weighted(clip,mask.clone())),old_kl];
                ensure!((0.0..=1.0).contains(&row[3]),"clip fraction outside [0,1]: {}",row[3]);
                let loss=weighted(actor+vl*0.5-entropy*cfg.entropy+anchor*coefficient,mask);
                ensure!(scalar(loss.clone()).is_finite(),"nonfinite GGD PPO objective");
                let mut grads=GradientsParams::from_grads(loss.backward(),policy);crate::gradients::clip(policy,&mut grads,0.5);
                if cfg.separate_critic_lr {
                    let value_grads=take_value_gradients(policy,&mut grads);
                    *policy=optimizer.step(cfg.learning_rate,policy.clone(),grads);
                    *policy=optimizer.step(if cfg.learning_rate==0.{0.}else{cfg.critic_learning_rate},policy.clone(),value_grads);
                }else{*policy=optimizer.step(cfg.learning_rate,policy.clone(),grads);}values.push(row);
            }
            offset+=shard_n;if stopped{break;}
        }
        if stopped{break;}
    }
    let mean=|i:usize|values.iter().map(|v|v[i]as f64).sum::<f64>()/values.len().max(1)as f64;
    Ok(serde_json::json!({"optimizer_steps":values.len(),"warmup":warmup,"explained_variance":ev,"ev_post0":ev_post,"ev_pregame":ev_pre,"postgame_decisions":post_count,"pregame_decisions":pre_count,"value_loss":mean(0),"kl_m2":mean(1),"entropy":mean(2),"clip_fraction":mean(3),"old_policy_kl":mean(4),"kl_coefficient":coefficient,"early_stop":stopped,"buckets":[pc,pt,pp]}))
}
fn checkpoint(policy:&Policy,optimizer:&ModuleOptimizer,manifest:&serde_json::Value,root:&Path,block:usize)->Result<std::path::PathBuf>{
    let path=root.join(format!("checkpoint-{block:06}"));std::fs::create_dir_all(&path)?;
    crate::bridge::export(&policy.clone().valid(),&path.join("weights"))?;
    let file=path.join("weights/manifest.json");let mut output:serde_json::Value=serde_json::from_slice(&std::fs::read(&file)?)?;
    for key in ["observation_profile","timing_honest","timing_oracle","zero_previous"] {if let Some(v)=manifest.get(key){output[key]=v.clone();}}
    for entry in output["tensors"].as_object_mut().unwrap().values_mut(){entry["sha256"]=serde_json::json!(crate::runtime::file_sha256(&path.join("weights").join(entry["file"].as_str().unwrap()))?);}
    output["source_weights"]=manifest.clone();crate::artifacts::atomic_json(&file,&output)?;
    policy.clone().into_record().save(path.join("model.bpk"))?;optimizer.save(path.join("optimizer.bpk"))?;
    crate::artifacts::atomic_json(&root.join("latest.json"),&serde_json::json!({"block":block,"path":path}))?;Ok(path)
}

pub fn run(cfg:Config,root:&Path)->Result<serde_json::Value>{
    ensure!(cfg.matches>0&&cfg.matches%2==0&&cfg.batch_sequences>0&&cfg.epochs>0&&cfg.minutes>0.&&cfg.minutes<=840.,"invalid PPO limits");
    ensure!(["p08","potential-v1","event-v2"].contains(&cfg.reward_mode.as_str()),"unknown reward mode");
    ensure!(cfg.schedule_minutes>0.&&cfg.mana_weight==0.5,"invalid schedule or unvalidated mana weight");
    let state_reward=cfg.reward_mode!="p08";let event_reward=cfg.reward_mode=="event-v2";
    if cfg.teacher_version==crate::teacher_v10::Version::V10 {crate::teacher_v10::check_pin()?;}
    std::fs::create_dir_all(root)?;let lock=std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(root.join("run.lock"))?;lock.try_lock()?;
    ensure!(!root.join("config.json").exists(),"fresh run directory required");
    let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(Path::new(&cfg.initial_weights).join("manifest.json"))?)?;
    let profile=manifest["observation_profile"].as_str().ok_or_else(||anyhow::anyhow!("checkpoint must declare observation_profile"))?;
    ensure!(["ggd-v2","ggd-v3","ggd-v4"].contains(&profile),"unsupported profile {profile}");
    ensure!(profile!="ggd-v4"||(manifest["timing_honest"].as_bool().unwrap_or(true)&&!manifest["timing_oracle"].as_bool().unwrap_or(false)),"ggd-v4 requires honest timing without oracle");
    ensure!(!manifest["zero_previous"].as_bool().unwrap_or(false),"zero_previous ablation not supported by PPO");
    crate::artifacts::atomic_json(&root.join("config.json"),&cfg)?;
    let engine=crate::artifacts::freeze_environment(Path::new(&cfg.engine),root,false)?;
    let d=device();d.seed(cfg.seed as u64);let reference=crate::bridge::import(Path::new(&cfg.initial_weights),&d)?.valid();ensure!(reference.is_ggd() && reference.is_ggd_v3()==(profile=="ggd-v3") && reference.is_ggd_v4()==(profile=="ggd-v4"),"checkpoint profile/shape mismatch");let mut policy=if let Some(path)=cfg.initialize_weights.as_ref(){crate::bridge::import(Path::new(path),&d)?.train()}else{reference.clone().train()};
    crate::artifacts::atomic_json(&root.join("provenance.json"),&serde_json::json!({"initial_weights":cfg.initial_weights,"initial_manifest_sha256":crate::runtime::file_sha256(&Path::new(&cfg.initial_weights).join("manifest.json"))?,"executable_sha256":crate::runtime::file_sha256(&std::env::current_exe()?)?,"teacher_sha256":cfg.teacher_version.sha256(),"reward_mode":cfg.reward_mode,"profile":profile,"ppo_source_sha256":crate::runtime::bytes_sha256(include_bytes!("ggd_ppo.rs")),"arena_source_sha256":crate::runtime::bytes_sha256(include_bytes!("ggd_arenas.rs")),"initialize_weights":cfg.initialize_weights,"gamma":0.99,"gae_lambda":0.95,"clip":0.2,"reset_decisions":128,"seed":cfg.seed}))?;
    let mut optimizer:ModuleOptimizer=AdamConfig::new().with_epsilon(1e-8).init();
    let mut resume=serde_json::Value::Null;
    if let Some(path)=&cfg.resume_checkpoint {
        let path=Path::new(path);resume=serde_json::from_slice(&std::fs::read(path.join("progress.json"))?)?;
        ensure!(resume["anchor_sha256"]==crate::runtime::file_sha256(&Path::new(&cfg.initial_weights).join("manifest.json"))?,"resume anchor changed");
        ensure!(resume["reward_mode"]==cfg.reward_mode&&resume["schedule_minutes"]==cfg.schedule_minutes,"resume reward or schedule changed");
        policy=policy.try_load_record(burn::store::ModuleRecord::load(path.join("model.bpk"))?)?;
        optimizer=optimizer.load(path.join("optimizer.bpk"))?;
    }
    let n=cfg.matches*2;let mut sides:Vec<usize>=(0..cfg.matches).map(|m|m%2).collect();let mut rows=crate::ggd_arenas::learner_rows(&sides,&d);
    let started=Instant::now();let mut training_seconds=0.;let mut evaluation_seconds=0.;let mut block=0;let mut batch=0;let mut episodes=0;
    if !resume.is_null(){training_seconds=resume["training_seconds"].as_f64().unwrap();block=resume["block"].as_u64().unwrap()as usize;}
    let resumed_seconds=training_seconds;
    let mut warmup=!cfg.skip_warmup;let mut warmup_state=CriticWarmup{minimum:cfg.warmup_min_blocks,..Default::default()};let mut total_decisions=0u64;let mut pregame_deaths=0u64;let mut prefix_pregame_deaths=0u64;let mut pregame_death_matches=0u64;let mut last_checkpoint=0.;let mut last_evaluation=0.;let mut last_path=None;let mut initialization=serde_json::Value::Null;let mut episodes_started=0usize;
    let mut previous_evaluation_weights=cfg.initial_weights.clone();
    if !resume.is_null(){last_checkpoint=training_seconds;last_evaluation=resume["last_evaluation"].as_f64().unwrap_or(0.);previous_evaluation_weights=resume["previous_evaluation_weights"].as_str().unwrap_or(&cfg.initial_weights).to_string();warmup=warmup&&!resume["warmup_complete"].as_bool().unwrap_or(false);batch=block as u32;}
    let mut history=std::fs::OpenOptions::new().create_new(true).write(true).open(root.join("history.jsonl"))?;
    let mut episode_file=std::fs::OpenOptions::new().create_new(true).write(true).open(root.join("episodes.jsonl"))?;
    'training:loop {
        let w=World::load(&engine,cfg.matches,&d)?;let teacher=crate::scripted_teacher::Teacher::new(&w);let observer=crate::combat_observe::CombatObserver::new(&w);let enemy=crate::enemy_observe::Observer::new(&w);let controller=crate::teacher_v10::Controller::new(&w,cfg.teacher_version,cfg.teacher_version,false);
        let rng=Sampler::new(cfg.matches,cfg.seed.wrapping_add(batch*1009),&d);let sampler=Sampler::new(n,cfg.seed.wrapping_add(batch*1009+1),&d);
        let metrics=crate::ggd_metrics::Metrics::new(&w);metrics.sample(&w,&teacher);
        let batch_offset=episodes;let mut recorded=vec![false;cfg.matches];episodes_started+=cfg.matches;
        let mut previous=crate::observe::initial_previous(n,&d);
        if cfg.desynchronize {(previous,initialization)=disperse(&w,&teacher,&observer,&enemy,&controller,&metrics,&rng,&sampler,&policy.clone().valid(),&manifest,root,cfg.seed,&mut sides)?;rows=crate::ggd_arenas::learner_rows(&sides,&d);}
        let mut prefix_decisions=tensor::<1>(w.meta.clone()).reshape([cfg.matches,META]).slice([0..cfg.matches,6..7]).reshape([cfg.matches]).into_data().try_to_vec::<f32>()?;
        let mut potential=crate::potential_reward::Tracker::new(&w,&teacher,cfg.mana_weight);
        let mut events=event_reward.then(||crate::event_reward::Tracker::new(&w,&teacher));
        let mut prior=counts(&w,&teacher);let mut totals=prior.clone().slice([0..n,0..COMPONENTS])*rows.clone().float().unsqueeze_dim::<2>(1);
        if cfg.desynchronize {let initial_deaths=scalar(-totals.clone().slice([0..n,4..5]).reshape([n]).sum()/10.)as u64;pregame_deaths+=initial_deaths;prefix_pregame_deaths+=initial_deaths;}
        let limit=((w.engine.pre_game_seconds+600.)*7.5).ceil()as usize+2;
        for _chunk in 0..if cfg.desynchronize{usize::MAX}else{limit.div_ceil(HORIZON)} {
            let clock=Instant::now();
            let start_ticks=if cfg.verify_phase_histogram{Some(tensor::<1>(w.meta.clone()).reshape([cfg.matches,META]).slice([0..cfg.matches,0..1]).reshape([cfg.matches]).into_data().try_to_vec::<f32>()?)}else{None};
            let actor=policy.clone().valid();let mut h=Tensor::<2>::zeros([n,256],&d);let mut c=h.clone();let mut rh=h.clone();let mut rc=c.clone();
            let mut experience=None;let mut postgame=vec![];let mut rollout_entropy=Tensor::<1>::zeros([1],&d);let mut rollout_kl=rollout_entropy.clone();let mut valid_count=rollout_entropy.clone();let mut reward_activity=rollout_entropy.clone();let mut pregame_events=rollout_entropy.clone();let mut phase_bins=Tensor::<1>::zeros([4],&d);
            for step in 0..HORIZON {
                let snapshot=if profile!="ggd-v2"{Some(teacher.command_snapshot())}else{None};
                let (obs,scripted)=observation(&w,&teacher,&observer,&enemy,&controller,previous.clone(),&manifest);
                if experience.is_none(){experience=Some(Experience::with_width(n,obs.packed.dims()[1],cfg.batch_sequences.max(32)*if state_reward{2}else{1},&d));}
                let keep:Tensor<2>=1.0f32-obs.reset.clone().unsqueeze_dim::<2>(1);h=h*keep.clone();c=c*keep.clone();rh=rh*keep.clone();rc=rc*keep;
                let b=obs.decode();let encoded=actor.encode(&b);(h,c)=actor.memory.step(encoded.fused,h,c);let heads=actor.heads(&b,h.clone(),encoded.keys,None,&sampler);
                let enc=reference.encode(&b);(rh,rc)=reference.memory.step(enc.fused,rh,rc);let anchor=reference.heads(&b,rh.clone(),enc.keys,Some(heads.actions.clone()),&sampler);
                let actions=scripted.clone().mask_where(rows.clone().unsqueeze_dim::<2>(1),heads.actions.clone());
                if let Some(s)=snapshot{teacher.reconcile_commands(&w,&obs,s,scripted,actions.clone());}
                previous=obs.previous(actions.clone(),&w.bases,&w.state);metrics.commands(&w,&teacher,actions.clone(),&obs.refs);
                let game_seconds=tensor::<1>(w.meta.clone()).reshape([cfg.matches,META]).slice([0..cfg.matches,0..1]).expand([cfg.matches,2]).reshape([n])/30.;postgame.push(game_seconds.clone().greater_equal_elem(0.).float());
                let valid=(1.0f32-prior.clone().slice([0..n,9..10]).reshape([n]))*rows.clone().float();
                phase_bins=phase_bins+crate::ggd_arenas::phase_counts(game_seconds,valid.clone());
                rollout_entropy=rollout_entropy+(heads.entropy.clone()*valid.clone()).sum();
                let diff=heads.logp.clone()-anchor.logp.clone();rollout_kl=rollout_kl+(((-diff.clone()).exp()-1.+diff)*valid.clone()).sum();valid_count=valid_count+valid.clone().sum();
                let _=w.advance_teacher(actions,&obs.refs,&rng,&teacher,true);teacher.record(&w,600.);metrics.sample(&w,&teacher);
                let current=counts(&w,&teacher);let components=current.clone().slice([0..n,0..COMPONENTS])-prior.slice([0..n,0..COMPONENTS]);
                let learning_components=if let Some(e)=events.as_mut(){e.step(&w,&teacher)}else if state_reward{potential.step(&w,&teacher)}else{components.clone()};
                let respawn_at=tensor::<1>(w.state.clone()).reshape([cfg.matches,UNITS,STRIDE]).slice([0..cfg.matches,0..2,RESPAWN_AT..RESPAWN_AT+1]).reshape([n]);
                pregame_events=pregame_events+pregame_death_events(components.clone(),respawn_at,valid.clone(),w.engine.pre_game_seconds);
                reward_activity=reward_activity+(learning_components.clone().abs()*valid.clone().unsqueeze_dim::<2>(1)).sum();
                totals=totals+components.clone()*valid.clone().unsqueeze_dim::<2>(1);
                let reward=learning_components.sum_dim(1);let done=current.clone().slice([0..n,9..10]);prior=current;
                let signals=Tensor::cat(vec![heads.actions.float(),heads.logp.unsqueeze_dim::<2>(1),heads.value.unsqueeze_dim::<2>(1),reward,done,obs.reset.clone().unsqueeze_dim::<2>(1),valid.unsqueeze_dim::<2>(1),anchor.logp.unsqueeze_dim::<2>(1)],1);
                experience.as_ref().unwrap().store(step,&obs,signals);
            }
            // Bootstrap at periodic memory reset, without advancing teacher state.
            let next=w.observe(previous.clone());let mut next=observer.enrich(&w,&next);
            if profile!="ggd-v2"{next=crate::timing_observe::extend(&w,next,&teacher,manifest["timing_honest"].as_bool().unwrap_or(true),manifest["timing_oracle"].as_bool().unwrap_or(false));}if profile=="ggd-v4"{next=enemy.extend(&w,next);}next.packed=next.packed.cast(DType::F16);
            let encoded=actor.encode(&next.decode());let (bh,_)=actor.memory.step(encoded.fused,Tensor::zeros([n,256],&d),Tensor::zeros([n,256],&d));
            let bootstrap=actor.value(bh)*(1.0f32-prior.clone().slice([0..n,9..10]).reshape([n]));
            d.sync()?;let collection=clock.elapsed().as_secs_f64();let update=Instant::now();
            let coefficient=cfg.kl_start+(cfg.kl_end-cfg.kl_start)*(training_seconds/(if state_reward{cfg.schedule_minutes}else{cfg.minutes}*60.)).min(1.)as f32;
            let mut learning=optimize(&mut policy,&mut optimizer,&cfg,experience.as_ref().unwrap(),bootstrap,warmup,coefficient,block,Tensor::stack(postgame,1))?;
            learning["rollout_entropy"]=serde_json::json!(scalar(rollout_entropy/valid_count.clone().clamp_min(1.)));
            learning["rollout_kl_m2"]=serde_json::json!(scalar(rollout_kl/valid_count.clone().clamp_min(1.)));
            let decisions=scalar(valid_count)as u64;total_decisions+=decisions;
            let phase_values=phase_bins.into_data().try_to_vec::<f32>()?;ensure!(phase_values.iter().sum::<f32>()==decisions as f32,"phase histogram count mismatch");
            if let Some(ticks)=start_ticks {
                let valid=experience.as_ref().unwrap().signals().slice([0..n,0..HORIZON,11..12]).sum_dim(1).reshape([n]).into_data().try_to_vec::<f32>()?;
                let mut expected=[0f32;4];for m in 0..cfg.matches{for step in 0..valid[m*2+sides[m]]as usize{let seconds=(ticks[m]+step as f32*4.)/30.;let bin=if seconds<0.{0}else if seconds<180.{1}else if seconds<360.{2}else{3};expected[bin]+=1.;}}
                ensure!(phase_values==expected,"phase histogram differs from native clock and valid decision counts: {phase_values:?} vs {expected:?}");
            }learning["valid_decisions"]=serde_json::json!(decisions);
            d.sync()?;let optimization=update.elapsed().as_secs_f64();training_seconds+=collection+optimization;block+=1;
            pregame_deaths+=scalar(pregame_events)as u64;let activity=scalar(reward_activity);learning["reward_activity"]=serde_json::json!(activity);
            if warmup&&warmup_state.observe(block,cfg.warmup_blocks,learning["postgame_decisions"].as_f64().unwrap_or(0.),learning["ev_post0"].as_f64(),activity){warmup=false;}
            learning["warmup_post_blocks"]=serde_json::json!(warmup_state.post_blocks);
            learning["warmup_ev_last5"]=serde_json::json!(warmup_state.mean());
            let audit=teacher.values();let finished=audit.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).filter(|x|x[0]>0.).count();
            ensure!(audit.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).all(|x|x[39]==0.),"simulation error");
            if finished>recorded.iter().filter(|x|**x).count(){
                let components=totals.clone().into_data().try_to_vec::<f32>()?;let mut games=metrics.games(&audit,w.engine.pre_game_seconds,batch_offset,Some("teacher"));
                crate::demo_telemetry::enrich(&w,&mut games)?;if state_reward{potential.enrich(&mut games)?;}if let Some(e)=&events{e.enrich(&mut games)?;}
                for m in 0..cfg.matches {if !recorded[m]&&audit[m*crate::scripted_teacher::AUDIT_FIELDS]>0. {
                    for side in 0..2 {games[m]["sides"][side]["controller"]=serde_json::json!(if side==sides[m]{"GGD"}else{"teacher"});}
                    if cfg.desynchronize {games[m]["game"]=serde_json::json!(episodes);}
                    let learner=&games[m]["sides"][sides[m]];if learner["first_death_game_seconds"].as_f64().is_some_and(|x|x<0.){pregame_death_matches+=1;}
                    let a=m*2+sides[m];let legacy=&components[a*COMPONENTS..(a+1)*COMPONENTS];let reward_values:Vec<f32>=if state_reward{serde_json::from_value(games[m]["sides"][sides[m]][if event_reward{"event_v2_components"}else{"reward_components"}].clone())?}else{legacy.to_vec()};let values=&reward_values;writeln!(episode_file,"{}",serde_json::json!({"block":block,"training_seconds":training_seconds,"reward":values.iter().sum::<f32>(),"components":values,"arena":m,"initialization_prefix_decisions":prefix_decisions[m],"game":games[m]}))?;recorded[m]=true;episodes+=1;
                }}episode_file.flush()?;
            }
            let row=serde_json::json!({"block":block,"batch":batch,"training_seconds":training_seconds,"collection_seconds":collection,"optimization_seconds":optimization,"episodes":episodes,"episodes_started":episodes_started,"total_decisions":total_decisions,"decisions_per_second":total_decisions as f64/training_seconds,"pregame_deaths":pregame_deaths,"pregame_deaths_initial_prefix":prefix_pregame_deaths,"pregame_death_matches":pregame_death_matches,"memory_pool":format!("{:?}",d.memory_pool_usage()),"finished_in_batch":finished,"phase_decisions":phase_values,"phase_histogram_verified":cfg.verify_phase_histogram,"phase_names":["pregame","0-3min","3-6min","6-10min"],"components_sum":if let Some(e)=&events{(e.totals.clone()*rows.clone().float().unsqueeze_dim::<2>(1)).sum_dim(0).reshape([6]).into_data().try_to_vec::<f32>()?}else if state_reward{(potential.totals.clone()*rows.clone().float().unsqueeze_dim::<2>(1)).sum_dim(0).reshape([5]).into_data().try_to_vec::<f32>()?}else{totals.clone().sum_dim(0).reshape([COMPONENTS]).into_data().try_to_vec::<f32>()?},"reward_component_names":if event_reward{crate::event_reward::NAMES.to_vec()}else if state_reward{crate::potential_reward::NAMES.to_vec()}else{NAMES.to_vec()},"learning":learning});
            writeln!(history,"{}",row)?;history.flush()?;eprintln!("{}",row);
            crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"stage":"training","pid":std::process::id(),"latest":row,"elapsed_seconds":started.elapsed().as_secs_f64()}))?;
            drop(experience);d.memory_cleanup();
            let user_stop=root.join("stop.json").exists();
            let stop=(cfg.critic_only&&!warmup)||training_seconds>=cfg.minutes*60.||(cfg.stop_at_minutes>0.&&training_seconds>=cfg.stop_at_minutes*60.)||(cfg.max_blocks>0&&block>=cfg.max_blocks)||user_stop;
            let eval=cfg.evaluate&&!user_stop&&(training_seconds-last_evaluation>=cfg.evaluation_minutes*60.||stop);
            if training_seconds-last_checkpoint>=cfg.checkpoint_minutes*60.||stop||eval{
                let path=checkpoint(&policy,&optimizer,&manifest,root,block)?;last_checkpoint=training_seconds;last_path=Some(path.clone());
                let progress=||serde_json::json!({"training_seconds":training_seconds,"block":block,"anchor_sha256":crate::runtime::file_sha256(&Path::new(&cfg.initial_weights).join("manifest.json")).unwrap(),"reward_mode":cfg.reward_mode,"schedule_minutes":cfg.schedule_minutes,"warmup_complete":!warmup});
                let mut saved=progress();saved["last_evaluation"]=last_evaluation.into();saved["previous_evaluation_weights"]=previous_evaluation_weights.clone().into();crate::artifacts::atomic_json(&path.join("progress.json"),&saved)?;
                if eval&&state_reward {
                    for opponent in [false,true] {for seed in [20261103u32,20261104] {
                        let name=if opponent{"previous"}else{"v10"};let target=root.join(format!("eval-{block:06}-{name}-{seed}"));
                        crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"stage":"evaluating","pid":std::process::id(),"block":block,"training_seconds":training_seconds,"opponent":name,"seed":seed,"report":target.join("report.json")}))?;
                        let report=crate::imitation::evaluate_against(&engine,&path.join("weights"),&target,64,600.,"teacher",seed,false,false,false,false,cfg.teacher_version,if opponent{Some(Path::new(&previous_evaluation_weights))}else{None})?;
                        ensure!(report["complete"]==true,"incomplete scheduled evaluation");evaluation_seconds+=report["wall_seconds"].as_f64().unwrap_or(0.);
                    }}
                    previous_evaluation_weights=path.join("weights").to_string_lossy().to_string();last_evaluation=training_seconds;
                    let mut saved=progress();saved["last_evaluation"]=last_evaluation.into();saved["previous_evaluation_weights"]=previous_evaluation_weights.clone().into();crate::artifacts::atomic_json(&path.join("progress.json"),&saved)?;
                    crate::artifacts::atomic_json(&root.join("evaluation-complete.json"),&serde_json::json!({"block":block,"training_seconds":training_seconds,"checkpoint":path,"previous_evaluation_weights":previous_evaluation_weights}))?;
                }
                if eval&&!state_reward {for greedy in [false,true]{let name=if greedy{"argmax"}else{"sampling"};
                    crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"stage":"evaluating","pid":std::process::id(),"block":block,"training_seconds":training_seconds,"mode":name}))?;
                    let report=crate::imitation::evaluate(&engine,&path.join("weights"),&root.join(format!("eval-{block:06}-{name}")),64,600.,"teacher",20261103,true,false,greedy,false,cfg.teacher_version)?;
                    evaluation_seconds+=report["wall_seconds"].as_f64().unwrap_or(0.);
                }last_evaluation=training_seconds;}
            }
            if stop{break 'training;}
            if cfg.desynchronize {
                let done:Vec<_>=audit.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).map(|x|x[0]>0.).collect();
                if done.iter().any(|x|*x){
                    let mask=crate::ggd_arenas::reset(&w,&teacher,&metrics,&done);potential.reset(&w,&teacher,mask.clone());if let Some(e)=events.as_mut(){e.reset(&w,&teacher,mask.clone());}previous=previous.mask_where(mask.clone().unsqueeze_dim::<2>(1),crate::observe::initial_previous(n,&d));totals=totals.mask_fill(mask.unsqueeze_dim::<2>(1),0.);
                    for m in 0..cfg.matches{if done[m]{sides[m]=1-sides[m];recorded[m]=false;prefix_decisions[m]=0.;episodes_started+=1;}}
                    rows=crate::ggd_arenas::learner_rows(&sides,&d);prior=counts(&w,&teacher);
                }
            }else if finished==cfg.matches{break;}
        }
        batch+=1;
    }
    let result=serde_json::json!({"blocks":block,"episodes":episodes,"episodes_started":episodes_started,"total_decisions":total_decisions,"pregame_deaths":pregame_deaths,"pregame_deaths_initial_prefix":prefix_pregame_deaths,"pregame_death_matches":pregame_death_matches,"warmup_complete":!warmup,"warmup_post_blocks":warmup_state.post_blocks,"warmup_ev_last5":warmup_state.mean(),"training_seconds":training_seconds,"evaluation_seconds":evaluation_seconds,"latest":last_path,"desynchronization":initialization,"profile":profile,"reward_components":if event_reward{crate::event_reward::NAMES.to_vec()}else if state_reward{crate::potential_reward::NAMES.to_vec()}else{NAMES.to_vec()},"resumed_training_seconds":resumed_seconds});
    crate::artifacts::atomic_json(&root.join("result.json"),&result)?;crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"stage":"complete","result":result}))?;Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn p09c_clip_fraction_counts_binary_events(){
        let d=device();let r=Tensor::<1>::from_data(TensorData::new(vec![0.1f32,0.8,1.,1.2,2.,4.],[6]),&d);
        let condition=r.clone().lower_elem(0.8).bool_or(r.greater_elem(1.2));
        let legacy=condition.clone().float().into_data().try_to_vec::<f32>().unwrap();
        let clip=Tensor::<1>::zeros([6],&d).mask_fill(condition,1.);
        assert_eq!(clip.clone().into_data().try_to_vec::<f32>().unwrap(),vec![1.,0.,0.,0.,1.,1.]);
        let mask=Tensor::<1>::from_data(TensorData::new(vec![1f32,1.,1.,1.,1.,0.],[6]),&d);
        assert!((scalar(weighted(clip,mask))-0.4).abs()<1e-6);
        eprintln!("P09c legacy bool-to-float diagnostic: {legacy:?}");
    }
    #[test]
    fn p09b_minibatch_uses_learner_sequences_and_preserves_legacy_order(){
        let active:Vec<f32>=(0..384).map(|i|if i%2==0{128.}else{0.}).collect();
        for offset in (0..384).step_by(96){
            let order=sequence_order(96,offset,Some(&active),17);
            assert_eq!(order.len(),48);assert!(order.iter().all(|i|*i%2==0));
            let mut sorted=order;sorted.sort();assert_eq!(sorted,(0..96).step_by(2).collect::<Vec<i32>>());
        }
        assert_eq!(sequence_order(4,0,None,1),vec![1,2,3,0]);
        assert!(sequence_order(4,0,Some(&[0.;4]),1).is_empty());
    }
    #[test]
    fn p08_critic_warmup_requires_30_postgame_blocks_and_five_ev_values(){
        let mut gate=CriticWarmup::default();
        for block in 1..=10 {assert!(!gate.observe(block,150,0.,Some(0.99),0.));}
        for block in 11..40 {assert!(!gate.observe(block,150,100.,Some(0.5),1.));}
        assert!(!gate.observe(40,150,100.,Some(0.5),0.),"the threshold also requires observed reward");assert_eq!(gate.post_blocks,30);
        assert!(gate.observe(41,150,100.,Some(0.5),1.));
        let mut gate=CriticWarmup::default();for block in 1..149{assert!(!gate.observe(block,150,100.,Some(0.1),1.));}
        assert!(gate.observe(150,150,0.,None,0.));
        let mut gate=CriticWarmup::default();for block in 1..=30{assert!(!gate.observe(block,150,100.,if block==30{Some(1.)}else{Some(0.)},1.));}
        assert!(!gate.observe(31,150,100.,None,0.));assert_eq!(gate.mean(),None);
    }
    #[test]
    fn p08_pregame_death_counter_uses_event_time_and_counts_events(){
        let d=device();let mut c=vec![0f32;4*COMPONENTS];c[4]=-10.;c[COMPONENTS+4]=-10.;c[2*COMPONENTS+4]=10.;c[3*COMPONENTS+4]=-10.;
        let components=Tensor::from_data(TensorData::new(c,[4,COMPONENTS]),&d);
        let events=pregame_death_events(components,Tensor::from_data([94.99f32,95.,90.,90.],&d),Tensor::from_data([1f32,1.,1.,0.],&d),83.);
        assert_eq!(scalar(events),1.);
    }
    #[test]
    fn p08_phase_ev_masks_only_diagnostics(){
        let d=device();let target=Tensor::from_data([0f32,1.,2.,3.],&d);let pred=Tensor::from_data([10f32,10.,2.,3.],&d);
        assert_eq!(explained_variance(target.clone(),pred.clone(),Tensor::from_data([0f32,0.,1.,1.],&d)),Some(1.));
        assert_eq!(explained_variance(target,pred,Tensor::from_data([1f32,1.,0.,0.],&d)),Some(0.));
    }
    #[test]
    fn p08_discounted_gae_respects_terminal_and_bootstrap(){
        let d=device();let mut e=Experience::with_width(2,crate::observe::GGD_WIDTH,2,&d);
        let mut signals=vec![0f32;2*HORIZON*crate::experience::SIGNALS];
        for a in 0..2 {for t in 0..HORIZON {let r=(a*HORIZON+t)*13;signals[r+7]=0.2;signals[r+8]=if t==30{1.}else{0.};signals[r+9]=if a==0&&t==31{1.}else{0.};}}
        e.signals=primitive(Tensor::<3>::from_data(TensorData::new(signals.clone(),[2,HORIZON,13]),&d));
        let (adv,ret)=e.gae_discounted(Tensor::from_data([0.7f32,0.9],&d),0.99,0.95);
        let av=adv.into_data().try_to_vec::<f32>().unwrap();let rv=ret.into_data().try_to_vec::<f32>().unwrap();
        for a in 0..2 {let mut next=if a==0{0.7}else{0.9};let mut carry=0.;for t in (0..HORIZON).rev(){let r=(a*HORIZON+t)*13;let live=1.-signals[r+9];let delta=signals[r+8]+0.99*next*live-signals[r+7];carry=delta+0.99*0.95*live*carry;assert!((av[a*HORIZON+t]-carry).abs()<2e-6);assert!((rv[a*HORIZON+t]-carry-0.2).abs()<2e-6);next=signals[r+7];}}
    }
    #[test]
    fn p08_reward_components_and_terminal_are_symmetric(){
        let d=device();let mut state=vec![0f32;UNITS*STRIDE];state[UNIT_LAST_HITS]=3.;state[UNIT_DENIES]=2.;state[STRIDE+UNIT_LAST_HITS]=4.;state[STRIDE+UNIT_DENIES]=6.;state[DEATHS]=1.;state[STRIDE+DEATHS]=2.;
        // Both towers destroyed: both signed terms must be retained independently.
        let mut meta=vec![0f32;META];meta[0]=18000.;meta[17]=-1.;let mut audit=vec![0f32;crate::scripted_teacher::AUDIT_FIELDS];audit[0]=1.;
        let state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));let meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));let audit=primitive(Tensor::<1>::from_data(TensorData::new(audit,[crate::scripted_teacher::AUDIT_FIELDS]),&d));let out=primitive(Tensor::<2>::zeros([2,10],&d));
        counters::launch(&out.client,CubeCount::Static(1,1,1),CubeDim::new_1d(64),argument(&state),argument(&meta),argument(&audit),argument(&out),2);
        assert_eq!(tensor::<2>(out).into_data().try_to_vec::<f32>().unwrap(),vec![3.,1.,-2.,-1.5,-10.,20.,-10.,10.,-5.,1.,4.,3.,-1.5,-0.5,-20.,10.,-10.,10.,5.,1.]);
    }
    #[test]
    fn p08_kl_importance_estimator_matches_exact_distribution(){
        let p=[0.15f64,0.55,0.3];let q=[0.3f64,0.3,0.4];let old=[0.2f64,0.7,0.1];
        let exact:f64=(0..3).map(|i|p[i]*(p[i]/q[i]).ln()).sum();
        let estimate:f64=(0..3).map(|i|{let delta=(p[i]/q[i]).ln();old[i]*(p[i]/old[i])*((-delta).exp()-1.+delta)}).sum();
        assert!((exact-estimate).abs()<1e-14);
    }
}
