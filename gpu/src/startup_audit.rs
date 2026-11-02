//! Bounded, fixed-data two-rollout PPO audit. Never initializes a training run.
use anyhow::{Result,ensure};
use burn::{module::Module,tensor::{Tensor,f16}};
use std::path::Path;
use crate::{learner::{Config,Learner,UpdateTerms},experience::{Experience,HORIZON},
    runtime::{device,tensor},sampling::Sampler,world::World,observe::{Observed,WIDTH}};

fn floats<const D:usize>(path:&Path,value:Tensor<D>)->Result<()> {
    let values=value.into_data().try_to_vec::<f32>()?;
    std::fs::write(path,values.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;Ok(())
}

pub fn run(config:Config,root:&Path)->Result<serde_json::Value>{
    ensure!(config.matches==100&&config.seed==2026110215&&config.learning_rate==2e-5,"requires the original minimal counterexample configuration");
    ensure!(config.initial_checkpoint.is_some()&&config.opponents.len()==1,"requires original initialization and one frozen original opponent");
    std::fs::create_dir_all(root)?;
    let lock=std::fs::OpenOptions::new().create(true).truncate(false).read(true).write(true).open(root.join("run.lock"))?;
    lock.try_lock().map_err(|e|anyhow::anyhow!("audit already active: {e}"))?;
    ensure!(!root.join("config.json").exists(),"use a fresh audit directory");
    crate::artifacts::atomic_json(&root.join("config.json"),&config)?;
    let engine=crate::artifacts::freeze_environment(Path::new(&config.engine),root,false)?;
    let d=device();let world=World::load(&engine,config.matches,&d)?;let n=config.matches*2;
    let mut learners=Vec::new();
    for(label,terms)in[("all",UpdateTerms::All),("policy-only",UpdateTerms::PolicyOnly),
        ("value-only",UpdateTerms::ValueOnly),("value-head-only",UpdateTerms::ValueHeadOnly),
        ("all-detached-value",UpdateTerms::AllDetachedValue)]{
        let mut learner=Learner::new(config.clone());
        learner.policy=crate::selfplay::load(Path::new(config.initial_checkpoint.as_ref().unwrap()),&d)?.train();
        learners.push((label,terms,learner));
    }
    let sample=Sampler::new(n,config.seed as u32,&d);let rng=Sampler::new(config.matches,config.seed as u32+1,&d);
    let opponents=crate::selfplay::Pool::new(&config.opponents,config.matches,&d)?;
    let experience=Experience::new(n,&d);let mut previous=crate::observe::initial_previous(n,&d);
    let mut report=serde_json::json!({"complete":false,"scope":"Two ordinary-All behavior rollouts. Every loss branch starts with identical original weights and fresh Adam and consumes exactly the same observations, actions, old_logp, old_value, reward, reset, learn mask and bootstrap. Branch-specific weights/Adam then evolve. Cycle2 is deliberately fixed-data, not an on-policy rollout of each branch. No training promotion.","cycles":[],"training_eligible":false});
    for cycle in 1..=2{
        let folder=root.join(format!("cycle-{cycle:02}"));std::fs::create_dir_all(&folder)?;
        crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"phase":"collecting","cycle":cycle,"pid":std::process::id()}))?;
        let actor=learners[0].2.policy.clone().valid();
        crate::bridge::export(&actor,&folder.join("behavior-tensors"))?;
        actor.clone().into_record().save(&folder.join("behavior.bpk"))?;
        let mut h=Tensor::<2>::zeros([n,256],&d);let mut c=h.clone();
        for step in 0..HORIZON{
            let mut observed=world.observe(previous.clone());observed.packed=observed.packed.cast(burn::tensor::DType::F16);
            let keep:Tensor<2>=1.0f32-observed.reset.clone().unsqueeze_dim::<2>(1);h=h*keep.clone();c=c*keep;
            let ph=h.clone();let pc=c.clone();let b=observed.decode();let encoded=actor.encode(&b);
            (h,c)=actor.memory.step(encoded.fused,h,c);let heads=actor.heads(&b,h.clone(),encoded.keys,None,&sample);
            let(nh,nc,heads,learn,versions)=opponents.decide(&world,&b,ph,pc,h,c,heads,&sample,cycle);h=nh;c=nc;
            let actions=heads.actions.clone().float();let next=observed.previous(heads.actions.clone(),&world.bases,&world.state);
            let(reward,done)=world.advance(heads.actions,&observed.refs,&rng);
            previous=next.mask_where(done.clone().greater_elem(0.5).unsqueeze_dim::<2>(1),crate::observe::initial_previous(n,&d));
            let signals=Tensor::cat(vec![actions,heads.logp.unsqueeze_dim::<2>(1),heads.value.unsqueeze_dim::<2>(1),reward.unsqueeze_dim::<2>(1),done.unsqueeze_dim::<2>(1),observed.reset.clone().unsqueeze_dim::<2>(1),learn,versions],1);
            experience.store(step,&observed,signals);
        }
        let mut next=world.observe(previous.clone());next.packed=next.packed.cast(burn::tensor::DType::F16);
        let b=next.decode();let encoded=actor.encode(&b);
        let(nh,nc)=actor.memory.step(encoded.fused.clone(),Tensor::zeros([n,256],&d),Tensor::zeros([n,256],&d));
        let bootstrap=actor.value(nh.clone());
        floats(&folder.join("signals.f32"),experience.signals())?;
        floats(&folder.join("bootstrap.f32"),bootstrap.clone())?;
        floats(&folder.join("bootstrap-h.f32"),nh.slice([0..4,0..256]))?;
        floats(&folder.join("bootstrap-c.f32"),nc.slice([0..4,0..256]))?;
        floats(&folder.join("bootstrap-fused.f32"),encoded.fused.slice([0..4,0..256]))?;
        let next_data=next.packed.slice([0..4,0..WIDTH]).into_data().try_to_vec::<f16>()?;
        std::fs::write(folder.join("bootstrap-inputs.f16"),next_data.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect::<Vec<_>>())?;
        let(adv,returns)=experience.gae(bootstrap.clone());
        floats(&folder.join("advantages.f32"),adv.clone())?;floats(&folder.join("returns.f32"),returns)?;
        let valid=experience.signals().slice([0..n,0..HORIZON,11..12]).reshape([n,HORIZON]);
        let count=valid.clone().sum().clamp_min(1.);let mean=(adv.clone()*valid.clone()).sum()/count.clone();
        let variance=((adv.clone()-mean.clone().reshape([1,1])).square()*valid).sum()/count;
        floats(&folder.join("normalized-advantages.f32"),(adv-mean.reshape([1,1]))/variance.sqrt().clamp_min(1e-8).reshape([1,1]))?;
        let raw=tensor::<3>(experience.observations[0].clone()).slice([0..4,0..HORIZON,0..WIDTH]);
        let obs=raw.clone().into_data().try_to_vec::<f16>()?;
        std::fs::write(folder.join("inputs.f16"),obs.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect::<Vec<_>>())?;
        let observed=Observed{packed:raw.reshape([4*HORIZON,WIDTH]),refs:experience.observations[0].clone(),reset:Tensor::zeros([4*HORIZON],&d)};
        let buckets=experience.buckets();let b=observed.decode_bucket(buckets.0,buckets.1,buckets.2);let encoded=actor.encode(&b);
        let fused=encoded.fused.clone().reshape([4,HORIZON,256]);
        floats(&folder.join("fused.f32"),fused.clone())?;
        let projected=actor.memory.input.forward(fused.reshape([4*HORIZON,256])).reshape([4,HORIZON,1024]);
        let resets=experience.signals().slice([0..4,0..HORIZON,10..11]);
        let mut h=Tensor::<2>::zeros([4,256],&d);let mut c=h.clone();let mut hs=vec![];let mut cs=vec![];
        for t in 0..HORIZON{
            let keep:Tensor<2>=1.0f32-resets.clone().slice([0..4,t..t+1,0..1]).reshape([4,1]);h=h*keep.clone();c=c*keep;
            (h,c)=actor.memory.projected(projected.clone().slice([0..4,t..t+1,0..1024]).reshape([4,1024]),h,c);hs.push(h.clone());cs.push(c.clone());
        }
        let hs=Tensor::stack::<3>(hs,1);let cs=Tensor::stack::<3>(cs,1);
        floats(&folder.join("h.f32"),hs.clone())?;floats(&folder.join("c.f32"),cs)?;
        let actions=experience.signals().slice([0..4,0..HORIZON,0..6]).reshape([4*HORIZON,6]).int();
        let heads=actor.heads(&b,hs.reshape([4*HORIZON,256]),encoded.keys,Some(actions),&Sampler::new(4*HORIZON,1,&d));
        floats(&folder.join("replay-logp.f32"),heads.logp.reshape([4,HORIZON]))?;
        floats(&folder.join("replay-value.f32"),heads.value.reshape([4,HORIZON]))?;
        floats(&folder.join("replay-entropy.f32"),heads.entropy.reshape([4,HORIZON]))?;
        let hash=crate::runtime::file_sha256(&folder.join("signals.f32"))?;let mut branches=vec![];
        for(label,terms,learner)in &mut learners{
            crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"phase":"fixed_data_update","cycle":cycle,"branch":label,"pid":std::process::id()}))?;
            let metrics=learner.optimize_terms(&experience,bootstrap.clone(),&sample,*terms)?;d.sync()?;
            let output=folder.join(*label);std::fs::create_dir_all(&output)?;
            let valid=learner.policy.clone().valid();crate::bridge::export(&valid,&output.join("tensors"))?;
            valid.into_record().save(&output.join("model.bpk"))?;
            branches.push(serde_json::json!({"label":label,"metrics":metrics}));
            let current=experience.signals().into_data().try_to_vec::<f32>()?;
            ensure!(crate::runtime::bytes_sha256(&current.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())==hash,"shared signals mutated");
        }
        let row=serde_json::json!({"cycle":cycle,"actors":n,"selected_actors":4,"horizon":HORIZON,"width":WIDTH,"buckets":[buckets.0,buckets.1,buckets.2],"signals_sha256":hash,"bootstrap_sha256":crate::runtime::file_sha256(&folder.join("bootstrap.f32"))?,"branches":branches});
        crate::artifacts::atomic_json(&folder.join("audit.json"),&row)?;report["cycles"].as_array_mut().unwrap().push(row);crate::artifacts::atomic_json(&root.join("result.json"),&report)?;
    }
    report["complete"]=true.into();crate::artifacts::atomic_json(&root.join("result.json"),&report)?;
    crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"phase":"complete","pid":std::process::id()}))?;Ok(report)
}
