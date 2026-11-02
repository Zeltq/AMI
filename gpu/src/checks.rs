use anyhow::{Result,ensure};
use burn::{module::Module,tensor::{Tensor,Int}};
use std::{path::Path,time::Instant};
use crate::{world::{World,META},observe::{WIDTH,TARGETS},model::Policy,runtime::{device,tensor},sampling::Sampler};

/// Explicit offline diagnostic; never called by collection or PPO.
pub fn dump_initial_observation(engine:&Path,output:&Path,ticks:usize)->Result<serde_json::Value>{
    let d=device();let world=World::load(engine,1,&d)?;let rng=Sampler::new(1,20261031,&d);
    if ticks>0 {let _=world.step_ticks(&rng,ticks);}
    let packed=world.observe(crate::observe::initial_previous(2,&d)).packed;
    let values=packed.into_data().try_to_vec::<f32>()?;
    let bytes:Vec<u8>=values.iter().flat_map(|v|v.to_le_bytes()).collect();
    if let Some(parent)=output.parent(){std::fs::create_dir_all(parent)?;}
    std::fs::write(output,bytes)?;
    Ok(serde_json::json!({"output":output,"shape":[2,WIDTH],"dtype":"f32_le","ticks":ticks,"scope":"initial simulator observation; offline diagnostic only"}))
}

fn measured<T:Send+'static>(d:&burn::tensor::Device,enabled:bool,label:&'static str,profiles:&mut Vec<(&'static str,burn::tensor::ProfileDuration)>,f:impl FnOnce()->T+Send)->Result<T>{
    if enabled {let (out,time)=d.profile(f)?;profiles.push((label,time));Ok(out)}else{Ok(f())}
}

/// Production-style queued collection. Only rollout-sized aggregate counters
/// cross to the host; per-stage device windows sample one decision per block.
pub fn benchmark(config:&crate::learner::Config,steps:usize,warmup:usize,physics_only:bool,profile:bool,continuous_memory:bool,trace_output:Option<&Path>)->Result<serde_json::Value>{
    ensure!(steps>0,"benchmark needs at least one step");let d=device();d.seed(config.seed);
    let engine=if let Some(root)=trace_output {
        ensure!(!root.join("metrics.jsonl").exists(),"trace output already used");
        let executable=std::env::current_exe()?;let target=root.join(executable.file_name().unwrap());
        if executable.canonicalize()?!=target.canonicalize().unwrap_or_default(){std::fs::copy(executable,target)?;}
        crate::artifacts::freeze_environment(Path::new(&config.engine),root,false)?
    }else{Path::new(&config.engine).to_path_buf()};
    let mut world=World::load(&engine,config.matches,&d)?;let n=config.matches*2;
    let writer=trace_output.map(crate::artifacts::Writer::new).transpose()?;
    let trace=writer.as_ref().map(|_|crate::trace::Trace::new(&world));
    let policy=if let Some(path)=config.initial_checkpoint.as_ref(){crate::selfplay::load(Path::new(path),&d)?}else{Policy::new(&d)}.valid();
    let pool=crate::selfplay::Pool::new(&config.opponents,config.matches,&d)?;
    let sampler=Sampler::new(n,config.seed as u32,&d);let rng=Sampler::new(config.matches,config.seed as u32+1,&d);
    let experience=if physics_only {None}else{Some(crate::experience::Experience::new(n,&d))};
    let mut previous=crate::observe::initial_previous(n,&d);let mut h=Tensor::<2>::zeros([n,256],&d);let mut c=h.clone();
    let mut last_done=Tensor::<1>::zeros([n],&d);
    let refs=crate::runtime::primitive(Tensor::<2>::zeros([n,TARGETS],&d));
    let mut profiles=vec![];let mut blocks=vec![];let mut started=Instant::now();let mut block=Instant::now();let mut initial_ticks=0.;let mut initial_completed=0.;let mut block_ticks=0.;
    for step in 0..warmup+steps {
        if step==warmup {
            d.sync()?;world=World::load(&engine,config.matches,&d)?;
            previous=crate::observe::initial_previous(n,&d);h=Tensor::zeros([n,256],&d);c=h.clone();last_done=Tensor::zeros([n],&d);d.sync()?;
            let meta=tensor::<1>(world.meta.clone()).reshape([config.matches,META]);
            initial_ticks=meta.clone().slice([0..config.matches,3..4]).sum().into_scalar::<f32>();
            initial_completed=meta.slice([0..config.matches,4..5]).sum().into_scalar::<f32>();block_ticks=initial_ticks;
            started=Instant::now();block=Instant::now();
        }
        let sample=profile && step>=warmup && (step-warmup)%128==0;
        if physics_only {
            let _=measured(&d,sample,"simulation",&mut profiles,||world.advance(Tensor::<2,Int>::zeros([n,6],&d),&refs,&rng))?;
        }else{
            let observed=measured(&d,sample,"observations",&mut profiles,||{
                let mut observed=world.observe(previous.clone());observed.packed=observed.packed.cast(burn::tensor::DType::F16);
                if continuous_memory {observed.reset=last_done.clone();}observed
            })?;
            let (next_h,next_c,heads,learn,versions)=measured(&d,sample,"inference",&mut profiles,||{
                let keep:Tensor<2>=1.0f32-observed.reset.clone().unsqueeze_dim::<2>(1);let prior_h=h.clone()*keep.clone();let prior_c=c.clone()*keep;
                let b=observed.decode();let encoded=policy.encode(&b);let (next_h,next_c)=policy.memory.step(encoded.fused,prior_h.clone(),prior_c.clone());
                let heads=policy.heads(&b,next_h.clone(),encoded.keys,None,&sampler);
                pool.decide(&world,&b,prior_h,prior_c,next_h,next_c,heads,&sampler,1)
            })?;h=next_h;c=next_c;
            measured(&d,sample,"simulation_and_experience",&mut profiles,||{
                let actions=heads.actions.clone().float();let next_previous=observed.previous(heads.actions.clone(),&world.bases,&world.state);
                let (reward,done)=world.advance(heads.actions,&observed.refs,&rng);
                last_done=done.clone();
                previous=next_previous.mask_where(done.clone().greater_elem(0.5).unsqueeze_dim::<2>(1),crate::observe::initial_previous(n,&d));
                let signals=Tensor::cat(vec![actions,heads.logp.unsqueeze_dim::<2>(1),heads.value.unsqueeze_dim::<2>(1),reward.unsqueeze_dim::<2>(1),done.unsqueeze_dim::<2>(1),observed.reset.clone().unsqueeze_dim::<2>(1),learn,versions],1);
                if !config.opponents.is_empty(){crate::selfplay::record_outcomes(&world,signals.clone());}
                if step>=warmup {if let Some(trace)=trace.as_ref(){trace.record(&world,(step-warmup)%128,signals.clone());}}
                experience.as_ref().unwrap().store(step%128,&observed,signals);
            })?;
        }
        if step>=warmup && ((step-warmup+1)%128==0 || step+1==warmup+steps){
            d.sync()?;let elapsed=block.elapsed().as_secs_f64();
            let ticks=tensor::<1>(world.meta.clone()).reshape([config.matches,META]).slice([0..config.matches,3..4]).sum().into_scalar::<f32>();
            let speed=(ticks-block_ticks) as f64/30./config.matches as f64/elapsed;
            blocks.push(speed);block_ticks=ticks;block=Instant::now();
            if let (Some(writer),Some(trace))=(writer.as_ref(),trace.as_ref()) {
                let cycle=((step-warmup+1)/128) as u64;writer.trace(&trace.buffer,trace.header(&world,cycle),cycle)?;
            }
        }
    }
    d.sync()?;let artifact_status=writer.as_ref().map(crate::artifacts::Writer::status);
    if let Some(writer)=writer{writer.finish()?;}
    let wall=started.elapsed().as_secs_f64();
    let meta=tensor::<1>(world.meta.clone()).reshape([config.matches,META]);
    ensure!(meta.clone().slice([0..config.matches,7..8]).max().into_scalar::<f32>()==0.,"benchmark environment error/capacity flag");
    let ticks=meta.clone().slice([0..config.matches,3..4]).sum().into_scalar::<f32>()-initial_ticks;
    let mut frozen=crate::selfplay::outcome_report(meta.clone().slice([0..config.matches,18..32]).sum_dim(0).into_data().try_to_vec::<f32>()?);
    frozen["estimate"]="fixed-policy simulator diagnostic; not Dota transfer".into();
    let completed=meta.slice([0..config.matches,4..5]).sum().into_scalar::<f32>()-initial_completed;
    let mean=blocks.iter().sum::<f64>()/blocks.len() as f64;let std=(blocks.iter().map(|x|(x-mean).powi(2)).sum::<f64>()/blocks.len() as f64).sqrt();
    let stage_profiles:Vec<_>=profiles.into_iter().map(|(stage,time)|serde_json::json!({"stage":stage,"timing":crate::runtime::profile_value(time)})).collect();
    Ok(serde_json::json!({"mode":if physics_only{"simulation_only"}else{"collection_without_ppo"},"matches":config.matches,"steps":steps,"warmup_steps":warmup,"wall_seconds":wall,
        "per_match_speed":ticks as f64/30./config.matches as f64/wall,"block_speed_mean":mean,"block_speed_std":std,"block_speeds":blocks,
        "observations_per_second":if physics_only {0.}else{(n*steps) as f64/wall},"completed":completed,"completed_matches_per_hour":completed as f64*3600./wall,
        "updates_per_second":0,"training_updates":0,"fixed_policy_vs_frozen":frozen,"memory_pool":format!("{:?}",d.memory_pool_usage()),"stage_profiles":stage_profiles,"artifacts_enabled":trace_output.is_some(),"artifact_status_before_drain":artifact_status,
        "decision_hz":7.5,"simulation_hz":30,"recurrent_reset_interval":if continuous_memory{None}else{Some(128)},"behavior_validated":world.engine.behavior_validated,
        "environment_sha256":world.fingerprint,"device_identity":format!("{:?}",d.identity()),"seed":config.seed,
        "executable_sha256":crate::runtime::file_sha256(&std::env::current_exe()?)?,
        "cargo_lock_sha256":crate::runtime::bytes_sha256(include_bytes!("../Cargo.lock"))}))
}

pub fn simulator(path:&Path,matches:usize,steps:usize,inference:bool)->Result<serde_json::Value>{
    let d=device();d.seed(20261030);
    let world=World::load(path,matches,&d)?;
    let n=matches*2;let sampler=Sampler::new(n,20261030,&d);let rng=Sampler::new(matches,20261031,&d);
    let policy=Policy::new(&d).no_grad();let mut h=Tensor::<2>::zeros([n,256],&d);let mut c=h.clone();
    let mut previous=crate::observe::initial_previous(n,&d);
    let mut sim_seconds=0.;let mut obs_seconds=0.;let mut infer_seconds=0.;
    let started=Instant::now();
    for _ in 0..steps {
        let t=Instant::now();let observed=world.observe(previous.clone());d.sync()?;obs_seconds+=t.elapsed().as_secs_f64();
        let t=Instant::now();
        let actions=if inference {
            let reset:Tensor<2>=1.0f32-observed.reset.clone().unsqueeze_dim::<2>(1);h=h*reset.clone();c=c*reset;
            let b=observed.decode();let encoded=policy.encode(&b);(h,c)=policy.memory.step(encoded.fused,h,c);
            policy.heads(&b,h.clone(),encoded.keys,None,&sampler).actions
        } else {Tensor::<2,Int>::zeros([n,6],&d)};
        d.sync()?;infer_seconds+=t.elapsed().as_secs_f64();
        let t=Instant::now();let next_previous=observed.previous(actions.clone(),&world.bases,&world.state);let (_,done)=world.advance(actions,&observed.refs,&rng);
        previous=next_previous.mask_where(done.greater_elem(0.5).unsqueeze_dim::<2>(1),crate::observe::initial_previous(n,&d));d.sync()?;sim_seconds+=t.elapsed().as_secs_f64();
    }
    let wall=started.elapsed().as_secs_f64();
    let meta=tensor::<1>(world.meta.clone()).reshape([matches,META]);
    let error=meta.clone().slice([0..matches,7..8]).max().into_scalar::<f32>();ensure!(error==0.,"simulator overflow/error flag {error}");
    let advanced=meta.clone().slice([0..matches,3..4]).sum().into_scalar::<f32>() as f64/30.;
    let completed=meta.slice([0..matches,4..5]).sum().into_scalar::<f32>();
    Ok(serde_json::json!({"matches":matches,"steps":steps,"inference":inference,"wall_seconds":wall,
        "per_match_speed":advanced/matches as f64/wall,"observation_seconds":obs_seconds,"inference_seconds":infer_seconds,
        "simulation_seconds":sim_seconds,"completed":completed,"observations_per_second":(steps*n) as f64/wall,
        "memory_pool":format!("{:?}",d.memory_pool_usage()),"observation_width":WIDTH,"targets":TARGETS,
        "diagnostic_sequential_stage_sync":true,"behavior_validated":false,"training_updates":0}))
}

pub fn cell_gradient()->Result<serde_json::Value>{
    use burn::tensor::{Distribution,activation};
    let d=device().autodiff();d.seed(19);
    let gates=Tensor::<2>::random([7,1024],Distribution::Uniform(-2.,2.),&d).require_grad();
    let c=Tensor::<2>::random([7,256],Distribution::Uniform(-1.,1.),&d).require_grad();
    let reference_gates=gates.clone().detach().require_grad();let reference_c=c.clone().detach().require_grad();
    let (h,new_c)=crate::cell::cell(gates.clone(),c.clone());
    let i=activation::sigmoid(reference_gates.clone().slice([0..7,0..256]));
    let f=activation::sigmoid(reference_gates.clone().slice([0..7,256..512]));
    let g=reference_gates.clone().slice([0..7,512..768]).tanh();
    let o=activation::sigmoid(reference_gates.clone().slice([0..7,768..1024]));
    let ref_c=f*reference_c.clone()+i*g;let ref_h=o*ref_c.clone().tanh();
    let forward_error=(h.clone()-ref_h.clone()).abs().max().into_scalar::<f32>();
    let weight=Tensor::<2>::random([7,512],Distribution::Uniform(-1.,1.),&device());
    let actual=(Tensor::cat(vec![h,new_c],1)*weight.clone()).sum();
    let expected=(Tensor::cat(vec![ref_h,ref_c],1)*weight).sum();
    let gradients=actual.backward();let reference_gradients=expected.backward();
    let gate_error=(gates.grad(&gradients).unwrap()-reference_gates.grad(&reference_gradients).unwrap()).abs().max().into_scalar::<f32>();
    let cell_error=(c.grad(&gradients).unwrap()-reference_c.grad(&reference_gradients).unwrap()).abs().max().into_scalar::<f32>();
    ensure!(forward_error<1e-5 && gate_error<1e-5 && cell_error<1e-5,"fused LSTM gradient mismatch");
    Ok(serde_json::json!({"passed":true,"forward_error":forward_error,"gate_gradient_error":gate_error,"cell_gradient_error":cell_error}))
}

