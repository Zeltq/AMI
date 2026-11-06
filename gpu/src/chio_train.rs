//! CHiO PPO and snapshot self-play; all controllers use the shared runtime.
use anyhow::{Result,ensure};
use burn::{module::Module,optim::{AdamConfig,GradientsParams,ModuleOptimizer},tensor::{Tensor,TensorData,Int}};
use serde::{Serialize,Deserialize};
use std::{path::Path,time::Instant};
use crate::{chio_model::Policy,chio_observe as co,chio_runtime::Runtime,world::World,observe::Observed,
    experience::{Experience,HORIZON},runtime::{device,tensor},sampling::Sampler,teacher_v10::Version};

#[derive(Clone,Serialize,Deserialize)]
#[serde(default,deny_unknown_fields)]
pub struct Config {
    pub observation_profile:String,pub engine:String,pub initial_weights:String,pub reward_mode:String,
    pub decision_ticks:usize,pub offset_ticks:usize,pub gamma:f32,pub gae_lambda:f32,pub zero_sum:bool,
    pub matches:usize,pub seed:u32,pub minutes:f64,pub max_blocks:usize,pub batch_sequences:usize,pub epochs:usize,
    pub learning_rate:f64,pub entropy:f32,pub clip:f32,pub gradient_norm:f32,pub stop_kl:f32,
    pub desynchronize:bool,pub self_play:bool,pub opponent_pool:Vec<String>,pub snapshot_blocks:usize,pub checkpoint_blocks:usize,
    pub resume_checkpoint:Option<String>,pub teacher_version:Version,
    pub critic_learning_rate:f64,pub separate_critic_lr:bool,pub critic_only:bool,pub warmup_min_blocks:usize,pub warmup_blocks:usize,
    pub kl_start:f32,pub kl_end:f32,pub schedule_minutes:f64,pub checkpoint_minutes:f64,pub snapshot_minutes:f64,
    pub mixed_pool:bool,pub bootstrap_checkpoint:Option<String>,pub evaluation_minutes:f64,
}
impl Default for Config {fn default()->Self{Self{observation_profile:"chio-v1".into(),engine:String::new(),initial_weights:String::new(),reward_mode:"lastorder-fb-v1".into(),
    decision_ticks:6,offset_ticks:6,gamma:0.997,gae_lambda:0.95,zero_sum:true,matches:16,seed:20261103,minutes:480.,max_blocks:0,batch_sequences:4,epochs:2,
    learning_rate:1e-5,entropy:0.005,clip:0.2,gradient_norm:0.5,stop_kl:0.02,desynchronize:true,self_play:false,opponent_pool:vec![],snapshot_blocks:20,checkpoint_blocks:10,
    resume_checkpoint:None,teacher_version:Version::V10,critic_learning_rate:3e-4,separate_critic_lr:false,critic_only:false,warmup_min_blocks:20,warmup_blocks:60,
    kl_start:0.,kl_end:0.,schedule_minutes:240.,checkpoint_minutes:0.,snapshot_minutes:0.,mixed_pool:false,bootstrap_checkpoint:None,evaluation_minutes:120.}}}
fn scalar(x:Tensor<1>)->f32{x.without_autodiff().into_scalar::<f32>()}
fn weighted(x:Tensor<1>,mask:Tensor<1>)->Tensor<1>{(x*mask.clone()).sum()/mask.sum().clamp_min(1.)}
fn coefficient(cfg:&Config,seconds:f64)->f32 {cfg.kl_start+(cfg.kl_end-cfg.kl_start)*(seconds/(cfg.schedule_minutes*60.)).clamp(0.,1.)as f32}
fn evaluation_due(minutes:f64,seconds:f64,last:f64)->bool {minutes>0.&&(seconds/(minutes*60.)).floor()>(last/(minutes*60.)).floor()}
fn value_gradients(policy:&Policy,source:&mut GradientsParams)->GradientsParams {
    use burn::module::{ModuleVisitor,Param};
    struct Take<'a>{source:&'a mut GradientsParams,target:GradientsParams}
    impl ModuleVisitor for Take<'_>{fn visit_float<const D:usize>(&mut self,p:&Param<Tensor<D>>){if let Some(g)=self.source.remove::<D>(p.id){self.target.register(p.id,g);}}}
    let mut take=Take{source,target:GradientsParams::new()};policy.value_first.visit(&mut take);policy.value_second.visit(&mut take);take.target
}
fn explained_variance(target:Tensor<1>,prediction:Tensor<1>,mask:Tensor<1>)->Option<f32>{
    if scalar(mask.clone().sum())==0.{return None;}let error=target.clone()-prediction;
    let tm=weighted(target.clone(),mask.clone());let em=weighted(error.clone(),mask.clone());
    let variance=scalar(weighted((target-tm).square(),mask.clone()));
    if variance<1e-8 {None}else{Some(1.-scalar(weighted((error-em).square(),mask))/variance)}
}
fn recurrent(policy:&Policy,b:&crate::model::Observation,resets:Tensor<3>,n:usize)->(Tensor<2>,Tensor<3>){
    let encoded=policy.encode(b);let projected=policy.memory.input.forward(encoded.fused).reshape([n,HORIZON,1024]);
    let mut h=Tensor::<2>::zeros([n,256],&projected.device());let mut c=h.clone();let mut sequence=vec![];
    for t in 0..HORIZON {let keep:Tensor<2>=1.0f32-resets.clone().slice([0..n,t..t+1,0..1]).reshape([n,1]);h=h*keep.clone();c=c*keep;
        (h,c)=policy.memory.projected(projected.clone().slice([0..n,t..t+1,0..1024]).reshape([n,1024]),h,c);sequence.push(h.clone());}
    (Tensor::stack::<3>(sequence,1).reshape([n*HORIZON,256]),encoded.keys)
}
fn optimize(policy:&mut Policy,optimizer:&mut ModuleOptimizer,e:&Experience,bootstrap:Tensor<1>,cfg:&Config,block:usize,anchor_weight:f32,postgame:Tensor<2>)->Result<serde_json::Value>{
    let d=device();let n=e.actors;let signals=e.signals();let (raw,returns)=e.gae_discounted(bootstrap,cfg.gamma,cfg.gae_lambda);
    let mask=signals.clone().slice([0..n,0..128,11..12]).reshape([n*128]);let raw=raw.reshape([n*128]);
    let mean=weighted(raw.clone(),mask.clone());let sd=weighted((raw.clone()-mean.clone()).square(),mask.clone()).sqrt().clamp_min(1e-8);
    let old_value=signals.clone().slice([0..n,0..128,7..8]).reshape([n*128]);
    let ev_post=explained_variance(returns.clone().reshape([n*128]),old_value,mask.clone()*postgame.reshape([n*128]));
    let active=signals.clone().slice([0..n,0..128,11..12]).sum_dim(1).reshape([n]).into_data().try_to_vec::<f32>()?;
    let adv=((raw-mean)/sd).reshape([n,128]);let (pc,pt,pp)=e.buckets();let mut rows=vec![];let mut stopped=false;
    for epoch in 0..cfg.epochs {let mut offset=0;
        for shard in &e.observations {let count=shard.meta.shape()[0];let width=shard.meta.shape()[2];
            let mut order:Vec<_>=(0..count).filter(|i|active[offset+*i]>0.).map(|i|i as i32).collect();let len=order.len();if len>0{order.rotate_left((epoch+block)%len);}
            for chunk in order.chunks(cfg.batch_sequences){let size=chunk.len();let local=Tensor::<1,Int>::from_data(TensorData::new(chunk.to_vec(),[size]),&d);let ids=local.clone()+offset as i32;
                let signal=signals.clone().select(0,ids.clone());let valid=signal.clone().slice([0..size,0..128,11..12]).reshape([size*128]);
                if scalar(valid.clone().sum())==0.{continue;}
                let input=tensor::<3>(shard.clone()).select(0,local).reshape([size*128,width]);let obs=Observed{packed:input,refs:shard.clone(),reset:Tensor::zeros([size*128],&d)};
                let (b,masks)=co::decode(&obs,pc,pt,pp);let frozen=cfg.critic_only.then(||policy.clone().valid());let (h,keys)=recurrent(frozen.as_ref().unwrap_or(policy),&b,signal.clone().slice([0..size,0..128,10..11]),size);
                let target=returns.clone().select(0,ids.clone()).reshape([size*128]);
                if cfg.critic_only {
                    let loss=weighted((policy.value(h.without_autodiff())-target).square(),valid);let vl=scalar(loss.clone());ensure!(vl.is_finite(),"nonfinite CHiO critic loss");
                    let mut grads=GradientsParams::from_grads(loss.backward(),policy);let norm=scalar(crate::gradients::clip(policy,&mut grads,cfg.gradient_norm));
                    *policy=optimizer.step(cfg.critic_learning_rate,policy.clone(),grads);rows.push(serde_json::json!({"value_loss":vl,"gradient_norm":norm}));continue;
                }
                let labels=signal.clone().slice([0..size,0..128,0..4]).reshape([size*128,4]).int();let sampler=Sampler::new(size*128,7,&d);
                let heads=policy.heads(&masks,h,keys,Some(labels),&sampler);let old=signal.clone().slice([0..size,0..128,6..7]).reshape([size*128]);
                let delta=heads.logp.clone()-old;let ratio=delta.clone().exp();let kl=scalar(weighted(ratio.clone()-1.-delta,valid.clone()));
                ensure!(kl.is_finite(),"nonfinite CHiO PPO KL");if kl>cfg.stop_kl{stopped=true;break;}
                let advantage=adv.clone().select(0,ids.clone()).reshape([size*128]);let _=ids;
                let actor=-(ratio.clone()*advantage.clone()).min_pair(ratio.clone().clamp(1.-cfg.clip,1.+cfg.clip)*advantage);
                let value=(heads.value-target).square();let entropy=heads.entropy;
                let vl=scalar(weighted(value.clone(),valid.clone()));let en=scalar(weighted(entropy.clone(),valid.clone()));
                let reference=signal.clone().slice([0..size,0..128,12..13]).reshape([size*128]);
                let diff=heads.logp-reference;let anchor=ratio*((-diff.clone()).exp()-1.+diff);
                let anchor_kl=scalar(weighted(anchor.clone(),valid.clone()));
                let loss=weighted(actor+value*0.5-entropy*cfg.entropy+anchor*anchor_weight,valid);let loss_value=scalar(loss.clone());ensure!(loss_value.is_finite(),"nonfinite CHiO PPO loss");
                let mut grads=GradientsParams::from_grads(loss.backward(),policy);let norm=scalar(crate::gradients::clip(policy,&mut grads,cfg.gradient_norm));ensure!(norm.is_finite(),"nonfinite CHiO gradient");
                if cfg.separate_critic_lr {let value=value_gradients(policy,&mut grads);*policy=optimizer.step(cfg.learning_rate,policy.clone(),grads);*policy=optimizer.step(cfg.critic_learning_rate,policy.clone(),value);}
                else{*policy=optimizer.step(cfg.learning_rate,policy.clone(),grads);}rows.push(serde_json::json!({"loss":loss_value,"value_loss":vl,"entropy":en,"old_policy_kl":kl,"anchor_kl":anchor_kl,"gradient_norm":norm}));
            }offset+=count;if stopped{break;}
        }if stopped{break;}
    }
    Ok(serde_json::json!({"optimizer_steps":rows.len(),"critic_only":cfg.critic_only,"ev_post0":ev_post,"anchor_coefficient":anchor_weight,"minibatches":rows,"early_stop_kl":stopped,"buckets":[pc,pt,pp]}))
}
fn checkpoint(policy:&Policy,optimizer:&ModuleOptimizer,root:&Path,block:usize,cfg:&Config,state:&serde_json::Value)->Result<std::path::PathBuf>{
    let path=root.join(format!("checkpoint-{block:06}"));std::fs::create_dir_all(&path)?;crate::chio_bridge::export(&policy.clone().valid(),&path.join("weights"))?;
    policy.clone().into_record().save(path.join("model.bpk"))?;optimizer.save(path.join("optimizer.bpk"))?;crate::artifacts::atomic_json(&path.join("config.json"),cfg)?;
    crate::artifacts::atomic_json(&path.join("state.json"),state)?;
    crate::artifacts::atomic_json(&root.join("latest.json"),&serde_json::json!({"block":block,"path":path}))?;Ok(path)
}
fn reset_finished(runtime:&mut Runtime,sides:&mut [usize],pool:&mut Option<crate::chio_pool::Pool>)->Result<usize>{
    let audit=runtime.teacher.values();ensure!(audit.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).all(|v|v[39]==0.),"CHiO simulation error");
    let finished:Vec<_>=audit.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).map(|v|v[0]>0.).collect();let count=finished.iter().filter(|x|**x).count();
    if count>0{runtime.reset(&finished);if let Some(p)=pool{p.reset(&finished);}for (m,done) in finished.iter().enumerate(){if *done{sides[m]=1-sides[m];}}}Ok(count)
}

pub fn run(cfg:Config,root:&Path)->Result<serde_json::Value>{
    ensure!(cfg.observation_profile=="chio-v1" && cfg.reward_mode=="lastorder-fb-v1" && cfg.decision_ticks==6 && cfg.offset_ticks==6 && cfg.gamma==0.997 && cfg.gae_lambda==0.95,"CHiO PPO contract mismatch");
    ensure!(cfg.matches>0 && cfg.matches<=1024 && cfg.batch_sequences>0 && cfg.epochs>0 && cfg.minutes>0. && cfg.learning_rate>0. && cfg.gradient_norm>0. && cfg.clip>0. && cfg.clip<1. && cfg.checkpoint_blocks>0 && cfg.snapshot_blocks>0,"invalid CHiO PPO dimensions");
    ensure!(cfg.self_play||cfg.opponent_pool.is_empty(),"opponent_pool requires self_play");ensure!(cfg.teacher_version==Version::V10,"CHiO PPO teacher must be v10");crate::teacher_v10::check_pin()?;
    ensure!(!cfg.mixed_pool||cfg.self_play,"mixed_pool requires self_play");ensure!(cfg.schedule_minutes>0.&&cfg.critic_learning_rate>0.&&cfg.warmup_min_blocks<=cfg.warmup_blocks,"invalid CHiO P16 schedule");
    std::fs::create_dir_all(root)?;let lock=std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(root.join("run.lock"))?;lock.try_lock()?;ensure!(!root.join("config.json").exists(),"fresh training directory required");crate::artifacts::atomic_json(&root.join("config.json"),&cfg)?;
    let frozen=crate::artifacts::freeze_environment(Path::new(&cfg.engine),root,false)?;let d=device();let world=World::load(&frozen,cfg.matches,&d)?;let n=cfg.matches*2;
    let mut runtime=Runtime::new(world,Version::V10,Version::V10,false,false,cfg.zero_sum)?;let mut policy=crate::chio_bridge::import(Path::new(&cfg.initial_weights),&d)?.train();
    let mut optimizer=AdamConfig::new().with_epsilon(1e-8).init();
    let anchor=policy.clone().valid();let anchor_sha=crate::runtime::file_sha256(&Path::new(&cfg.initial_weights).join("manifest.json"))?;
    let mut saved=serde_json::json!({});
    if let Some(path)=&cfg.bootstrap_checkpoint {let path=Path::new(path);policy=policy.try_load_record(burn::store::ModuleRecord::load(path.join("model.bpk"))?)?;optimizer=optimizer.load(path.join("optimizer.bpk"))?;}
    if let Some(path)=&cfg.resume_checkpoint {
        let path=Path::new(path);let previous:Config=serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        ensure!(previous.zero_sum==cfg.zero_sum && previous.reward_mode==cfg.reward_mode && previous.gamma==cfg.gamma,"resume reward mismatch");
        policy=policy.try_load_record(burn::store::ModuleRecord::load(path.join("model.bpk"))?)?;optimizer=optimizer.load(path.join("optimizer.bpk"))?;
        saved=serde_json::from_slice(&std::fs::read(path.join("state.json"))?)?;
        ensure!(saved["anchor_sha256"].as_str()==Some(anchor_sha.as_str())&&previous.initial_weights==cfg.initial_weights&&previous.schedule_minutes==cfg.schedule_minutes&&previous.kl_start==cfg.kl_start&&previous.kl_end==cfg.kl_end,"resume must preserve stage anchor and schedule");
    }
    let rng=Sampler::new(cfg.matches,cfg.seed,&d);let sampler=Sampler::new(n,cfg.seed.wrapping_add(1),&d);let enemy_sampler=Sampler::new(n,cfg.seed.wrapping_add(2),&d);
    let mut pool:Vec<String>=saved.get("pool").and_then(|x|x.get("paths")).map(|x|serde_json::from_value(x.clone())).transpose()?.unwrap_or_else(||cfg.opponent_pool.clone());if cfg.self_play && pool.is_empty(){pool.push(cfg.initial_weights.clone());}
    let mut mixed=if cfg.mixed_pool{Some(crate::chio_pool::Pool::new(&runtime.world,pool.clone(),cfg.seed.wrapping_add(saved["block"].as_u64().unwrap_or(0)as u32))?)}else{None};
    let mut opponent=if cfg.self_play&&!cfg.mixed_pool{Some(crate::chio_bridge::import(Path::new(&pool[0]),&d)?.valid())}else{None};
    let mut sides:Vec<_>=(0..cfg.matches).map(|m|m%2).collect();let experience=Experience::with_width(n,co::WIDTH,32,&d);
    let max_delay=((runtime.world.engine.pre_game_seconds+600.)*5./128.).floor()as usize;
    let delays=if cfg.desynchronize{crate::ggd_arenas::delays(cfg.matches,max_delay,cfg.seed)}else{vec![0;cfg.matches]};
    let warm_blocks=*delays.iter().max().unwrap();let mut activated=vec![false;cfg.matches];if warm_blocks>0{crate::ggd_arenas::hold_all(&runtime.teacher,cfg.matches);}
    let started=Instant::now();let mut block=saved["block"].as_u64().unwrap_or(0)as usize;let mut completed=saved["completed_episodes"].as_u64().unwrap_or(0)as usize;let mut optimizer_steps=saved["optimizer_steps"].as_u64().unwrap_or(0);
    let mut training_seconds=saved["training_seconds"].as_f64().unwrap_or(0.);let mut last_checkpoint_seconds=training_seconds;let mut last_snapshot_seconds=saved["last_snapshot_seconds"].as_f64().unwrap_or(0.);
    let mut ev_history:Vec<f64>=saved.get("ev_history").map(|x|serde_json::from_value(x.clone())).transpose()?.unwrap_or_default();let mut history=vec![];let mut phase_total=vec![0f32;4];let mut component_total=vec![0f32;28];let mut last_checkpoint=None;
    for iteration in 0usize.. {
        let block_started=Instant::now();let warming=iteration<warm_blocks;let activate:Vec<_>=(0..cfg.matches).map(|m|!activated[m]&&delays[m]<=iteration).collect();
        if activate.iter().any(|x|*x){runtime.reset(&activate);for m in 0..cfg.matches{activated[m]|=activate[m];}}
        let rows=crate::ggd_arenas::learner_rows(&sides,&d);let actor=policy.clone().valid();
        if cfg.self_play && !cfg.mixed_pool && !warming {let index=(block+cfg.seed as usize)%pool.len();opponent=Some(crate::chio_bridge::import(Path::new(&pool[index]),&d)?.valid());}
        if let Some(p)=&mut mixed {p.begin_block(&runtime.world,&sides);runtime.route_v9(&p.v9_rows(&sides));}
        let mut ah=Tensor::<2>::zeros([n,256],&d);let mut ac=ah.clone();let mut postgame=Vec::new();
        let mut h=Tensor::<2>::zeros([n,256],&d);let mut c=h.clone();let mut oh=h.clone();let mut oc=h.clone();let mut phase=Tensor::<1>::zeros([4],&d);let mut parts=Tensor::<1>::zeros([28],&d);
        for step in 0..128 {
            let frame=runtime.observe();let keep:Tensor<2>=1.0f32-frame.observation.reset.clone().unsqueeze_dim::<2>(1);h=h*keep.clone();c=c*keep.clone();oh=oh*keep.clone();oc=oc*keep.clone();ah=ah*keep.clone();ac=ac*keep;
            let (b,masks)=co::decode(&frame.observation,crate::observe::CREEPS,crate::observe::TREES,crate::observe::PROJECTILES);let encoded=actor.encode(&b);(h,c)=actor.memory.step(encoded.fused,h,c);
            let heads=actor.heads(&masks,h.clone(),encoded.keys,None,&sampler);let mut chosen=frame.labels.clone();
            if let Some(enemy)=&opponent {let encoded=enemy.encode(&b);(oh,oc)=enemy.memory.step(encoded.fused,oh,oc);chosen=enemy.heads(&masks,oh.clone(),encoded.keys,None,&enemy_sampler).actions;}
            if let Some(p)=&mut mixed{chosen=p.actions(&frame.observation,chosen);}
            let reference=if !warming&&!cfg.critic_only&&(cfg.kl_start>0.||cfg.kl_end>0.) {let encoded=anchor.encode(&b);(ah,ac)=anchor.memory.step(encoded.fused,ah,ac);anchor.heads(&masks,ah.clone(),encoded.keys,Some(heads.actions.clone()),&sampler).logp}else{Tensor::<1>::zeros([n],&d)};
            chosen=chosen.mask_where(rows.clone().unsqueeze_dim::<2>(1),heads.actions.clone());let valid=frame.valid.clone()*rows.clone().float();
            let transition=runtime.advance(&frame,chosen,&rng,600.);let seconds=tensor::<2>(transition.before.clone()).slice([0..n,0..1]).reshape([n]);postgame.push(seconds.clone().greater_equal_elem(0.).float());phase=phase+crate::ggd_arenas::phase_counts(seconds,valid.clone());
            parts=parts+(transition.reward.components*valid.clone().unsqueeze_dim::<2>(1)).sum_dim(0).reshape([28]);
            if !warming {let done=transition.done.max_pair(1.0f32-frame.valid);let signals=Tensor::cat(vec![heads.actions.float(),Tensor::zeros([n,2],&d),heads.logp.unsqueeze_dim::<2>(1),heads.value.unsqueeze_dim::<2>(1),
                (transition.reward.total*valid.clone()).unsqueeze_dim::<2>(1),done.unsqueeze_dim::<2>(1),frame.observation.reset.clone().unsqueeze_dim::<2>(1),valid.unsqueeze_dim::<2>(1),reference.unsqueeze_dim::<2>(1)],1);experience.store(step,&frame.observation,signals);}
        }
        if warming {let audit=runtime.teacher.values();let finished:Vec<_>=audit.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).enumerate().map(|(m,v)|activated[m]&&v[0]>0.).collect();if finished.iter().any(|x|*x){runtime.reset(&finished);if let Some(p)=&mut mixed{p.reset(&finished);}}}
        else {
            let frame=runtime.observe();let (b,_)=co::decode(&frame.observation,crate::observe::CREEPS,crate::observe::TREES,crate::observe::PROJECTILES);
            // At the 128-decision boundary the next policy state starts at zero.
            let encoded=actor.encode(&b);let (next,_)=actor.memory.step(encoded.fused,Tensor::zeros([n,256],&d),Tensor::zeros([n,256],&d));
            let bootstrap=actor.value(next)*frame.valid*rows.float();block+=1;let update=optimize(&mut policy,&mut optimizer,&experience,bootstrap,&cfg,block,coefficient(&cfg,training_seconds),Tensor::stack(postgame,1))?;
            if let Some(ev)=update["ev_post0"].as_f64(){ev_history.push(ev);}training_seconds+=block_started.elapsed().as_secs_f64();
            optimizer_steps+=update["optimizer_steps"].as_u64().unwrap();let pv=phase.into_data().try_to_vec::<f32>()?;let cv=parts.into_data().try_to_vec::<f32>()?;
            for i in 0..4{phase_total[i]+=pv[i];}for i in 0..28{component_total[i]+=cv[i];}
            let mut games=runtime.games(0)?;if let Some(p)=&mixed {for (m,g) in games.iter_mut().enumerate(){g["opponent_assignment"]=serde_json::to_value(&p.assignments[m])?;g["learner_side"]=sides[m].into();}}let audit=runtime.teacher.values();let terminal:Vec<_>=games.into_iter().enumerate().filter_map(|(m,g)|(audit[m*crate::scripted_teacher::AUDIT_FIELDS]>0.).then_some(g)).collect();
            if !terminal.is_empty(){crate::artifacts::atomic_json(&root.join(format!("games-{block:06}.json")),&terminal)?;}
            completed+=reset_finished(&mut runtime,&mut sides,&mut mixed)?;let record=serde_json::json!({"block":block,"update":update,"phase_counts":pv,"reward_components":cv,"completed_episodes":completed,"training_seconds":training_seconds,"anchor_sha256":anchor_sha,"wall_seconds":started.elapsed().as_secs_f64()});history.push(record.clone());
            crate::artifacts::atomic_json(&root.join(format!("block-{block:06}.json")),&record)?;
            let snapshot=cfg.self_play&&if cfg.snapshot_minutes>0.{training_seconds-last_snapshot_seconds>=cfg.snapshot_minutes*60.}else{block%cfg.snapshot_blocks==0};
            let evaluation_due=evaluation_due(cfg.evaluation_minutes,training_seconds,last_checkpoint_seconds);
            let save=evaluation_due||if cfg.checkpoint_minutes>0.{training_seconds-last_checkpoint_seconds>=cfg.checkpoint_minutes*60.}else{block%cfg.checkpoint_blocks==0};
            if save||snapshot {let state=serde_json::json!({"block":block,"training_seconds":training_seconds,"anchor_sha256":anchor_sha,"optimizer_steps":optimizer_steps,"completed_episodes":completed,"pool":mixed.as_ref().map(|p|p.evidence()),"ev_history":ev_history,"last_snapshot_seconds":last_snapshot_seconds});
                let path=checkpoint(&policy,&optimizer,root,block,&cfg,&state)?;
                if snapshot{let weights=path.join("weights").to_string_lossy().into_owned();pool.push(weights.clone());if let Some(p)=&mut mixed{p.add(&weights,&d)?;}last_snapshot_seconds=training_seconds;
                    let mut updated=state;updated["pool"]=mixed.as_ref().map(|p|p.evidence()).unwrap_or(serde_json::Value::Null);updated["last_snapshot_seconds"]=last_snapshot_seconds.into();crate::artifacts::atomic_json(&path.join("state.json"),&updated)?;}
                last_checkpoint_seconds=training_seconds;last_checkpoint=Some((block,path));}
        }
        let status=serde_json::json!({"stage":if warming{"desynchronizing"}else if cfg.critic_only{"critic-chio"}else{"training-chio"},"pid":std::process::id(),"block":block,"warmup_block":iteration.min(warm_blocks),"warmup_blocks":warm_blocks,"optimizer_steps":optimizer_steps,"completed_episodes":completed,"training_seconds":training_seconds,"anchor_sha256":anchor_sha,"wall_seconds":started.elapsed().as_secs_f64()});
        crate::artifacts::atomic_json(&root.join("status.json"),&status)?;eprintln!("{status}");
        let critic_done=cfg.critic_only&&(block>=cfg.warmup_blocks||(block>=cfg.warmup_min_blocks&&ev_history.len()>=5&&ev_history.iter().rev().take(5).sum::<f64>()/5.>0.3));
        if !warming && (critic_done||(cfg.max_blocks>0&&block>=cfg.max_blocks)||training_seconds>=cfg.minutes*60.||root.join("stop.json").exists()){break;}
    }
    let path=if let Some((at,path))=last_checkpoint.filter(|(at,_)|*at==block){let _=at;path}else{checkpoint(&policy,&optimizer,root,block,&cfg,&serde_json::json!({"block":block,"training_seconds":training_seconds,"anchor_sha256":anchor_sha,"optimizer_steps":optimizer_steps,"completed_episodes":completed,"pool":mixed.as_ref().map(|p|p.evidence()),"ev_history":ev_history,"last_snapshot_seconds":last_snapshot_seconds}))?};
    let result=serde_json::json!({"family":"CHiO","training_seconds":training_seconds,"anchor_sha256":anchor_sha,"mixed_pool":mixed.as_ref().map(|p|p.evidence()),"critic_only":cfg.critic_only,"ev_history":ev_history,"blocks":block,"optimizer_steps":optimizer_steps,"completed_episodes":completed,"phase_counts":phase_total,"reward_components":component_total,
        "component_names":crate::chio_reward::NAMES,"enemy_components_are_subtracted":true,"zero_sum":cfg.zero_sum,"gamma":cfg.gamma,"gae_lambda":cfg.gae_lambda,"activation_delay_blocks":delays,"opponent_pool":pool,"checkpoint":path,"wall_seconds":started.elapsed().as_secs_f64()});
    crate::chio_bridge::export(&policy.clone().valid(),&root.join("final-weights"))?;
    crate::artifacts::atomic_json(&root.join("result.json"),&result)?;Ok(result)
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn p16_anchor_schedule_uses_elapsed_training_time(){let c=Config{kl_start:0.03,kl_end:0.01,schedule_minutes:120.,..Config::default()};assert!((coefficient(&c,0.)-0.03).abs()<1e-7);assert!((coefficient(&c,3600.)-0.02).abs()<1e-7);assert!((coefficient(&c,9000.)-0.01).abs()<1e-7);assert!(!evaluation_due(120.,7199.,6600.));assert!(evaluation_due(120.,7201.,6600.));assert!(!evaluation_due(120.,7202.,7201.));assert!(evaluation_due(120.,14401.,13800.));}
    #[test] fn p16_critic_freeze_anchor_resume_and_mixed_episode_assignment(){
        let base=std::path::PathBuf::from(format!("C:/AMI/runs/p16/rust-training-check-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let source="C:/AMI/runs/p15/imitation-aim-smoke/weights";
        let c=Config{engine:"C:/AMI/runs/p16/environment/engine.json".into(),initial_weights:source.into(),matches:2,max_blocks:1,batch_sequences:1,epochs:1,desynchronize:false,critic_only:true,warmup_min_blocks:1,warmup_blocks:1,separate_critic_lr:true,learning_rate:3e-5,critic_learning_rate:3e-4,kl_start:0.03,kl_end:0.01,..Config::default()};
        let r=run(c.clone(),&base.join("critic")).unwrap();assert!(r["optimizer_steps"].as_u64().unwrap()>0);
        let old:serde_json::Value=serde_json::from_slice(&std::fs::read(Path::new(source).join("manifest.json")).unwrap()).unwrap();let new:serde_json::Value=serde_json::from_slice(&std::fs::read(base.join("critic/final-weights/manifest.json")).unwrap()).unwrap();
        let mut frozen=0;for (name,value) in old["tensors"].as_object().unwrap(){if !name.starts_with("value_"){assert_eq!(value["sha256"],new["tensors"][name]["sha256"],"{name}");frozen+=1;}}assert_eq!(frozen,45);
        let cfg=Config{critic_only:false,warmup_min_blocks:20,warmup_blocks:60,..c.clone()};let first=run(cfg.clone(),&base.join("ppo")).unwrap();
        let resumed=Config{resume_checkpoint:Some(first["checkpoint"].as_str().unwrap().into()),max_blocks:2,..cfg};let second=run(resumed,&base.join("resume")).unwrap();
        assert_eq!(second["blocks"],2);assert_eq!(first["anchor_sha256"],second["anchor_sha256"]);assert!(second["training_seconds"].as_f64().unwrap()>first["training_seconds"].as_f64().unwrap());
        let w=World::load(Path::new(&c.engine),2,&device()).unwrap();let mut pool=crate::chio_pool::Pool::new(&w,vec![source.into(),base.join("ppo/final-weights").to_string_lossy().into()],19).unwrap();
        let prior=pool.assignments.clone();pool.add(base.join("resume/final-weights").to_str().unwrap(),&device()).unwrap();assert_eq!(pool.assignments,prior);pool.reset(&[true,false]);assert_eq!(pool.assignments[1],prior[1]);
        crate::artifacts::atomic_json(&base.join("result.json"),&serde_json::json!({"passed":true,"frozen_policy_tensors":frozen,"total_policy_tensors":45,"resume_blocks":2,"anchor_sha256":first["anchor_sha256"]})).unwrap();
    }
}
