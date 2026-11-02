//! Authorized one-rollout LR comparison. Outputs are diagnostic, never promoted.
use anyhow::{Result,ensure};
use burn::{module::Module,tensor::Tensor};
use std::{path::Path,time::Instant};
use crate::{learner::{Config,Learner},experience::{Experience,HORIZON},runtime::device,
    sampling::Sampler,world::World};

pub fn run(config:Config,root:&Path)->Result<serde_json::Value>{
    run_branches(config,root,false)
}
pub fn diagnose_updates(config:Config,root:&Path)->Result<serde_json::Value>{
    run_branches(config,root,true)
}
fn run_branches(config:Config,root:&Path,compare_terms:bool)->Result<serde_json::Value>{
    ensure!(config.matches==100&&config.learning_rate==if compare_terms{2e-5}else{1e-4},"use the pinned100-match diagnostic learning rate");
    ensure!(config.initial_checkpoint.is_some()&&config.opponents.len()==1,"original frozen MIA2 is required");
    std::fs::create_dir_all(root)?;
    let lock=std::fs::OpenOptions::new().create(true).truncate(false).read(true).write(true).open(root.join("run.lock"))?;
    lock.try_lock().map_err(|e|anyhow::anyhow!("LR diagnostic already active:{e}"))?;
    ensure!(!root.join("config.json").exists(),"use a fresh diagnostic directory");
    crate::artifacts::atomic_json(&root.join("config.json"),&config)?;
    let engine=crate::artifacts::freeze_environment(Path::new(&config.engine),root,false)?;
    let d=device();d.seed(config.seed);let world=World::load(&engine,config.matches,&d)?;
    let actor=crate::selfplay::load(Path::new(config.initial_checkpoint.as_ref().unwrap()),&d)?.valid();
    let n=config.matches*2;let experience=Experience::new(n,&d);
    let sample=Sampler::new(n,config.seed as u32,&d);let rng=Sampler::new(config.matches,config.seed as u32+1,&d);
    let opponents=crate::selfplay::Pool::new(&config.opponents,config.matches,&d)?;
    let mut previous=crate::observe::initial_previous(n,&d);
    let mut h=Tensor::<2>::zeros([n,256],&d);let mut c=h.clone();let started=Instant::now();
    crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"phase":"collecting_one_rollout","pid":std::process::id()}))?;
    for step in 0..HORIZON{
        let mut observed=world.observe(previous.clone());observed.packed=observed.packed.cast(burn::tensor::DType::F16);
        let keep:Tensor<2>=1.0f32-observed.reset.clone().unsqueeze_dim::<2>(1);h=h*keep.clone();c=c*keep;
        let ph=h.clone();let pc=c.clone();let b=observed.decode();let encoded=actor.encode(&b);
        (h,c)=actor.memory.step(encoded.fused,h,c);let heads=actor.heads(&b,h.clone(),encoded.keys,None,&sample);
        let(nh,nc,heads,learn,versions)=opponents.decide(&world,&b,ph,pc,h,c,heads,&sample,1);h=nh;c=nc;
        let actions=heads.actions.clone().float();let next=observed.previous(heads.actions.clone(),&world.bases,&world.state);
        let(reward,done)=world.advance(heads.actions,&observed.refs,&rng);
        previous=next.mask_where(done.clone().greater_elem(0.5).unsqueeze_dim::<2>(1),crate::observe::initial_previous(n,&d));
        let signals=Tensor::cat(vec![actions,heads.logp.unsqueeze_dim::<2>(1),heads.value.unsqueeze_dim::<2>(1),reward.unsqueeze_dim::<2>(1),done.unsqueeze_dim::<2>(1),observed.reset.clone().unsqueeze_dim::<2>(1),learn,versions],1);
        experience.store(step,&observed,signals);
    }
    let mut next=world.observe(previous);next.packed=next.packed.cast(burn::tensor::DType::F16);let b=next.decode();let encoded=actor.encode(&b);
    let(nh,_)=actor.memory.step(encoded.fused,Tensor::zeros([n,256],&d),Tensor::zeros([n,256],&d));let bootstrap=actor.value(nh);
    d.sync()?;let collection=started.elapsed().as_secs_f64();
    let signals=experience.signals().into_data().try_to_vec::<f32>()?;
    let bytes:Vec<u8>=signals.iter().flat_map(|x|x.to_le_bytes()).collect();
    let signal_hash=crate::runtime::bytes_sha256(&bytes);std::fs::write(root.join("shared-signals.f32"),bytes)?;
    let mut reports=vec![];
    use crate::learner::UpdateTerms;
    let branches=if compare_terms{vec![("all",2e-5,UpdateTerms::All),("policy-only",2e-5,UpdateTerms::PolicyOnly),("value-only",2e-5,UpdateTerms::ValueOnly),("value-head-only",2e-5,UpdateTerms::ValueHeadOnly)]}else{vec![("lr-1e-4",1e-4,UpdateTerms::All),("lr-2e-5",2e-5,UpdateTerms::All)]};
    for(label,rate,terms)in branches{
        let folder=root.join(label);std::fs::create_dir_all(&folder)?;
        let mut cfg=config.clone();cfg.learning_rate=rate;
        let mut learner=Learner::new(cfg.clone());
        learner.policy=crate::selfplay::load(Path::new(config.initial_checkpoint.as_ref().unwrap()),&d)?.train();
        crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"phase":"one_rollout_ppo","branch":label,"pid":std::process::id()}))?;
        let time=Instant::now();let report=learner.optimize_terms(&experience,bootstrap.clone(),&sample,terms)?;d.sync()?;
        crate::artifacts::atomic_json(&folder.join("config.json"),&cfg)?;
        let valid=learner.policy.valid();crate::bridge::export(&valid,&folder.join("tensors"))?;
        valid.into_record().save(&folder.join("model.bpk"))?;
        reports.push(serde_json::json!({"label":label,"learning_rate":rate,"wall_seconds":time.elapsed().as_secs_f64(),"learning":report}));
        let after=experience.signals().into_data().try_to_vec::<f32>()?;
        ensure!(signals.iter().zip(&after).all(|(a,b)|a.to_bits()==b.to_bits()),"shared rollout signals mutated");
        crate::artifacts::atomic_json(&root.join("branches.json"),&reports)?;
    }
    let report=serde_json::json!({"complete":true,"shared_rollouts":1,"decisions":HORIZON,"actors":n,"matches":config.matches,
        "shared_signal_sha256":signal_hash,"same_experience_buffers":true,"collection_seconds":collection,"branches":reports,
        "environment_sha256":world.fingerprint,"initial_checkpoint":config.initial_checkpoint,"opponent_sha256":opponents.identities,
        "training_eligible":false,"compare_loss_terms":compare_terms,"scope":if compare_terms{"Only enabled loss terms differ; LR2e-5, original weights, fresh Adam, same immutable GPU Experience/bootstrap/order/settings. ValueHeadOnly detaches recurrent features. Diagnostic only, no strength claim."}else{"Only LR differs. Each branch reloads original weights and a fresh Adam state; exact same immutable GPU Experience and bootstrap; KL stop and all other PPO settings retained. No match strength conclusion."}});
    crate::artifacts::atomic_json(&root.join("report.json"),&report)?;
    crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"phase":"complete","pid":std::process::id()}))?;Ok(report)
}
