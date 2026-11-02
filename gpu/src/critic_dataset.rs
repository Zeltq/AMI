//! Frozen, first-complete-episode critic features under the training runtime.
use anyhow::{Result,ensure};
use burn::{module::Module,tensor::Tensor};
use cubecl::prelude::*;
use std::path::Path;
use std::time::Instant;
use crate::{world::*,contract::*,learner::Config,runtime::{device,primitive,tensor,argument},sampling::Sampler};
pub const AUX:usize=19;
#[cube(launch)]
fn record_before(state:&mut [f32],meta:&[f32],hidden:&[f32],values:&[f32],logp:&[f32],actions:&[f32],reset:&[f32],learn:&[f32],finished:&[f32],features:&mut [f32],aux:&mut [f32],step:usize,actors:usize){
    let i=ABSOLUTE_POS;
    if i<actors*256{
        let a=i/256;let f=i%256;
        if finished[a]==0.0f32{features[(step*actors+a)*256+f]=hidden[i];}
    }
    if i<actors*AUX{
        let a=i/AUX;let f=i%AUX;let m=a/2;let side=a%2;let base=m*UNITS*STRIDE;let p=base+side*STRIDE;let at=(step*actors+a)*AUX+f;
        if finished[a]==0.0f32{
            let mut v=0.0f32.runtime();
            if f==0{v=meta[m*META]/30.0f32;}if f==1{v=state[p+UNIT_X];}if f==2{v=state[p+UNIT_Y];}
            if f==3{v=potential(state,base)*if side==0{1.0f32}else{-1.0f32};}
            if f==4{v=values[a];}if f==5{v=logp[a];}if f==6{v=reset[a];}
            if f>=7&&f<=12{v=actions[a*6+f-7];}if f==15{v=meta[m*META+1];}if f==16{v=learn[a];}if f==17{v=state[p+UNIT_HEALTH];}
            aux[at]=v;
        }
    }
}
#[cube(launch)]
fn record_after(reward:&[f32],done:&[f32],meta:&[f32],finished:&mut [f32],aux:&mut [f32],step:usize,actors:usize){
    let a=ABSOLUTE_POS;if a<actors&&finished[a]==0.0f32{
        let at=(step*actors+a)*AUX;aux[at+13]=reward[a];aux[at+14]=done[a];if done[a]>0.5f32{aux[at+18]=meta[a/2*META+17]*if a%2==0{1.0f32}else{-1.0f32};finished[a]=1.0f32;}
    }
}
pub fn collect(config:Config,root:&Path)->Result<serde_json::Value>{
    ensure!((2..=32).contains(&config.matches),"bounded2..32 match collector");
    ensure!(config.initial_checkpoint.is_some()&&config.opponents.len()==1&&config.initial_checkpoint.as_ref().unwrap()==&config.opponents[0],"both actors must use the same original policy");
    std::fs::create_dir_all(root)?;let lock=std::fs::OpenOptions::new().create(true).truncate(false).read(true).write(true).open(root.join("run.lock"))?;lock.try_lock()?;
    ensure!(!root.join("config.json").exists(),"fresh dataset directory required");crate::artifacts::atomic_json(&root.join("config.json"),&config)?;
    let engine=crate::artifacts::freeze_environment(Path::new(&config.engine),root,false)?;let d=device();d.seed(config.seed);let world=World::load(&engine,config.matches,&d)?;let n=config.matches*2;
    let actor=crate::selfplay::load(Path::new(config.initial_checkpoint.as_ref().unwrap()),&d)?.valid();let opponents=crate::selfplay::Pool::new(&config.opponents,config.matches,&d)?;
    let sample=Sampler::new(n,config.seed as u32,&d);let rng=Sampler::new(config.matches,config.seed as u32+1,&d);
    let capacity=((world.engine.max_seconds+world.engine.pre_game_seconds)*7.5).ceil()as usize+128;
    let features=primitive(Tensor::<3>::zeros([capacity,n,256],&d));let aux=primitive(Tensor::<3>::zeros([capacity,n,AUX],&d));let finished=primitive(Tensor::<1>::zeros([n],&d));
    let mut previous=crate::observe::initial_previous(n,&d);let mut h=Tensor::<2>::zeros([n,256],&d);let mut c=h.clone();let started=Instant::now();let mut steps=0;let mut complete=false;
    for step in 0..capacity{
        let mut obs=world.observe(previous.clone());obs.packed=obs.packed.cast(burn::tensor::DType::F16);let keep:Tensor<2>=1.0f32-obs.reset.clone().unsqueeze_dim::<2>(1);h=h*keep.clone();c=c*keep;
        let ph=h.clone();let pc=c.clone();let b=obs.decode();let encoded=actor.encode(&b);(h,c)=actor.memory.step(encoded.fused,h,c);let heads=actor.heads(&b,h.clone(),encoded.keys,None,&sample);
        let(nh,nc,heads,learn,_)=opponents.decide(&world,&b,ph,pc,h,c,heads,&sample,1);h=nh;c=nc;
        let actions=primitive(heads.actions.clone().float());let values=primitive(heads.value);let logp=primitive(heads.logp);let hidden=primitive(h.clone());let reset=primitive(obs.reset.clone());let learn=primitive(learn.reshape([n]));
        record_before::launch(&world.state.client,CubeCount::Static((n*256).div_ceil(256)as u32,1,1),CubeDim::new_1d(256),argument(&world.state),argument(&world.meta),argument(&hidden),argument(&values),argument(&logp),argument(&actions),argument(&reset),argument(&learn),argument(&finished),argument(&features),argument(&aux),step,n);
        let next=obs.previous(heads.actions.clone(),&world.bases,&world.state);let(reward,done)=world.advance(heads.actions,&obs.refs,&rng);
        previous=next.mask_where(done.clone().greater_elem(0.5).unsqueeze_dim::<2>(1),crate::observe::initial_previous(n,&d));let reward=primitive(reward);let done=primitive(done);
        record_after::launch(&world.state.client,CubeCount::Static(n.div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&reward),argument(&done),argument(&world.meta),argument(&finished),argument(&aux),step,n);steps=step+1;
        if steps%128==0{
            let count=tensor::<1>(finished.clone()).sum().into_scalar::<f32>()as usize;let errors=tensor::<1>(world.meta.clone()).reshape([config.matches,META]).slice([0..config.matches,7..8]).max().into_scalar::<f32>();ensure!(errors==0.,"world capacity/error flag");
            crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"phase":"frozen_episode_collection","pid":std::process::id(),"steps":steps,"completed_matches":count/2,"target_matches":config.matches,"elapsed_seconds":started.elapsed().as_secs_f64()}))?;
            if count==n{complete=true;break;}
            if root.join("stop.json").exists(){break;}
        }
    }
    let write=|path:&Path,values:Vec<f32>|->Result<()>{std::fs::write(path,values.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>())?;Ok(())};
    write(&root.join("features.f32"),tensor::<3>(features).slice([0..steps,0..n,0..256]).into_data().try_to_vec::<f32>()?)?;
    write(&root.join("aux.f32"),tensor::<3>(aux).slice([0..steps,0..n,0..AUX]).into_data().try_to_vec::<f32>()?)?;
    crate::bridge::export(&actor,&root.join("original-tensors"))?;
    let report=serde_json::json!({"complete":complete,"matches":config.matches,"actors":n,"steps":steps,"feature_shape":[steps,n,256],"aux_shape":[steps,n,AUX],"columns":["time","x","y","own_phi","value","logp","reset","action","buy","item","target","x_bin","y_bin","reward","terminal","episode","learn","health","outcome"],"selection":"sampling","observation_storage":"f16","recurrent_hc":"fp32","recurrent_reset_interval":128,"memory_reset_origin_seconds":null,"reset":"global decision counter128 and life/terminal; identical to learner collection","environment_sha256":world.fingerprint,"opponent_sha256":opponents.identities,"initial_checkpoint":config.initial_checkpoint,"config":config,"optimizer_steps":0,"policy_frozen":true,"features_sha256":crate::runtime::file_sha256(&root.join("features.f32"))?,"aux_sha256":crate::runtime::file_sha256(&root.join("aux.f32"))?,"executable_sha256":crate::runtime::file_sha256(&std::env::current_exe()?)?,"wall_seconds":started.elapsed().as_secs_f64(),"scope":"Only each preselected first true completed episode; no truncations relabeled terminal. Both roles are originalMIA2, including byte-identical frozen aliases. Stored recurrent features suffice for head-only fitting; no actor update."});
    crate::artifacts::atomic_json(&root.join("dataset.json"),&report)?;ensure!(complete,"incomplete frozen collection; retained raw artifacts");Ok(report)
}
