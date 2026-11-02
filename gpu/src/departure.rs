//! Short frozen-policy closed-loop prefixes; never optimize or infer strength.
use anyhow::{Result,ensure};
use burn::{module::Module,tensor::Tensor};
use cubecl::prelude::*;
use std::path::Path;
use crate::{world::*,experience::SIGNALS,evaluation::FIELDS,runtime::{device,primitive,tensor,argument},sampling::Sampler};
const TIMELINE_FIELDS:usize=15;
#[cube(launch)]
fn timeline_store(state:&[f32],meta:&[f32],actions:&[f32],reset:&[f32],finished:&[f32],timeline:&mut [f32],step:usize){
    let i=ABSOLUTE_POS;if i<4*TIMELINE_FIELDS{
        let a=i/TIMELINE_FIELDS;let f=i%TIMELINE_FIELDS;let p=(a/2*UNITS+a%2)*STRIDE;
        if finished[a/2*FIELDS]==0.0f32{
            let mut v=0.0f32.runtime();if f==0{v=1.0f32;}if f==1{v=meta[a/2*META]/30.0f32;}if f==2{v=reset[a];}
            if f==3{v=state[p+crate::contract::UNIT_X];}if f==4{v=state[p+crate::contract::UNIT_Y];}if f==5{v=state[p+crate::contract::UNIT_HEALTH];}
            if f>=6&&f<12{v=actions[a*6+f-6];}if f==12{v=state[p+ORDER];}if f==13{v=state[p+DEST_X];}if f==14{v=state[p+DEST_Y];}
            timeline[(step*4+a)*TIMELINE_FIELDS+f]=v;
        }
    }
}

#[cube(launch)]
fn commands(signals:&[f32],finished:&[f32],meta:&[f32],counts:&mut [f32],matches:usize){
    let a=ABSOLUTE_POS;
    if a<matches*2 && finished[a/2*FIELDS]==0.0f32 {let action=signals[a*SIGNALS]as usize;if action<11 {counts[a*22+action]+=1.0f32;if meta[a/2*META]>=0.0f32 {counts[a*22+11+action]+=1.0f32;}}}
}

#[cube(launch)]
fn forward_once(actions:&mut [f32],learn:&[f32],mask:&[f32],meta:&[f32],finished:&[f32],marks:&mut [f32],matches:usize){
    let a=ABSOLUTE_POS;
    if a<matches*2 && a%2==0 && learn[a]>0.5f32 && meta[a/2*META]>=0.0f32 && finished[a/2*FIELDS]==0.0f32 && mask[a*11+1]>0.5f32 && marks[a*3]==0.0f32 {
        marks[a*3]=1.0f32;marks[a*3+1]=meta[a/2*META];marks[a*3+2]=actions[a*6];
        actions[a*6]=1.0f32;actions[a*6+4]=127.0f32;actions[a*6+5]=128.0f32;
    }
}

pub(crate) fn inject_forward_once(world:&World,actions:Tensor<2>,learn:Tensor<2>,mask:Tensor<2>,finished:&burn_cubecl::tensor::CubeTensor,marks:&burn_cubecl::tensor::CubeTensor)->Tensor<2>{
    let raw=primitive(actions);let learn=primitive(learn);let mask=primitive(mask);
    forward_once::launch(&world.state.client,CubeCount::Static(1,1,1),CubeDim::new_1d((world.matches*2)as u32),argument(&raw),argument(&learn),argument(&mask),argument(&world.meta),argument(finished),argument(marks),world.matches);
    tensor(raw)
}

pub fn run(config:crate::learner::Config,root:&Path,argmax:bool,capture_start:bool,reset_origin:Option<f32>,post_mid_seconds:f32,force_first_forward_move:bool)->Result<serde_json::Value>{
    ensure!(config.matches==2,"departure diagnostic uses exactly2balanced-side matches");
    ensure!(config.opponents.len()==1&&config.initial_checkpoint.is_some(),"two frozen policies required");
    ensure!(!capture_start || (argmax&&config.initial_checkpoint.as_ref()==config.opponents.first()),"start capture is restricted to identical frozen policies inargmax");
    ensure!(!force_first_forward_move || (argmax&&!capture_start),"one-command intervention requires argmax and a full bounded prefix");
    std::fs::create_dir_all(root)?;ensure!(!root.join("departure-config.json").exists(),"diagnostic output already used");
    ensure!(post_mid_seconds.is_finite()&&(0.0..=90.0).contains(&post_mid_seconds),"bounded post-mid prefix required");
    let deadline=if capture_start {3.}else{90.};let post_mid=if capture_start {0.}else{post_mid_seconds};
    ensure!(reset_origin.is_none_or(f32::is_finite),"finite reset clock required");
    crate::artifacts::atomic_json(&root.join("departure-config.json"),&serde_json::json!({"config":config,"selection":if argmax{"argmax"}else{"sampling"},"deadline_game_seconds":deadline,"post_mid_seconds":post_mid,"capture_start":capture_start,"memory_reset_origin_seconds":reset_origin,"training_updates":0,"force_first_forward_move":force_first_forward_move}))?;
    let engine=crate::artifacts::freeze_environment(Path::new(&config.engine),root,false)?;let d=device();d.seed(config.seed);
    let world=World::load(&engine,2,&d)?;let policy=crate::selfplay::load(Path::new(config.initial_checkpoint.as_ref().unwrap()),&d)?.valid();
    let pool=crate::selfplay::Pool::evaluation(&config.opponents[0],2,&d)?;
    let sampler=Sampler::new(4,config.seed as u32,&d).with_argmax(argmax);let rng=Sampler::new(2,config.seed as u32+1,&d);
    let results=primitive(Tensor::<2>::zeros([2,FIELDS],&d));let counts=primitive(Tensor::<2>::zeros([4,22],&d));
    let intervention=primitive(Tensor::<2>::zeros([4,3],&d));
    let mut previous=crate::observe::initial_previous(4,&d);let mut h=Tensor::<2>::zeros([4,256],&d);let mut c=h.clone();
    let limit=((world.engine.pre_game_seconds+deadline)*7.5).ceil()as usize+if capture_start {1}else{16};let mut completed=0;let mut steps=0;
    let timeline=primitive(Tensor::<3>::zeros([limit,4,TIMELINE_FIELDS],&d));
    let capture=if capture_start {Some(crate::policy_trace::PolicyTrace::new(limit,&d))}else{None};
    for step in 0..limit {
        let mut observed=world.observe_with_clock(previous.clone(),reset_origin);observed.packed=observed.packed.cast(burn::tensor::DType::F16);
        let keep:Tensor<2>=1.0f32-observed.reset.clone().unsqueeze_dim::<2>(1);let prior_h=h*keep.clone();let prior_c=c*keep;
        let recorded_prior=if capture_start {Some((prior_h.clone(),prior_c.clone()))}else{None};
        let b=observed.decode();let encoded=policy.encode(&b);let recorded_fused=if capture_start {Some(encoded.fused.clone())}else{None};let (nh,nc)=policy.memory.step(encoded.fused,prior_h.clone(),prior_c.clone());
        let heads=policy.heads(&b,nh.clone(),encoded.keys,None,&sampler);
        let (nh,nc,heads,learn,versions)=pool.decide(&world,&b,prior_h,prior_c,nh,nc,heads,&sampler,1);h=nh;c=nc;
        let actions=if force_first_forward_move {inject_forward_once(&world,heads.actions.clone().float(),learn.clone(),b.action_mask.clone().float(),&results,&intervention)}else{heads.actions.clone().float()};
        let applied=if force_first_forward_move {actions.clone().int()}else{heads.actions.clone()};
        if let Some(capture)=capture.as_ref(){
            let (ph,pc)=recorded_prior.unwrap();let fused=recorded_fused.unwrap();let logits=policy.action_head.forward(h.clone());
            let row=|x:Tensor<2>,width|x.slice([0..1,0..width]);
            let tick=tensor::<1>(world.meta.clone()).slice([0..1]).reshape([1,1]);
            let extra=Tensor::cat(vec![tick,observed.reset.clone().slice([0..1]).reshape([1,1]),row(ph,256),row(pc,256),row(h.clone(),256),row(c.clone(),256),row(fused,256),row(logits,11),row(actions.clone(),6)],1);
            capture.record(step,observed.packed.clone().slice([0..1,0..crate::observe::WIDTH]),extra);
        }
        let action_raw=primitive(actions.clone());let reset_raw=primitive(observed.reset.clone());
        timeline_store::launch(&world.state.client,CubeCount::Static(1,1,1),CubeDim::new_1d(64),argument(&world.state),argument(&world.meta),argument(&action_raw),argument(&reset_raw),argument(&results),argument(&timeline),step);
        let next=observed.previous(applied.clone(),&world.bases,&world.state);let (reward,done)=world.advance(applied,&observed.refs,&rng);
        previous=next.mask_where(done.clone().greater_elem(0.5).unsqueeze_dim::<2>(1),crate::observe::initial_previous(4,&d));
        let signals=Tensor::cat(vec![actions,heads.logp.unsqueeze_dim::<2>(1),heads.value.unsqueeze_dim::<2>(1),reward.unsqueeze_dim::<2>(1),done.unsqueeze_dim::<2>(1),observed.reset.unsqueeze_dim::<2>(1),learn,versions],1);
        let raw=primitive(signals.clone());commands::launch(&world.state.client,CubeCount::Static(1,1,1),CubeDim::new_1d(4),argument(&raw),argument(&results),argument(&world.meta),argument(&counts),2);
        crate::evaluation::record_prefix(&world,signals,&results,deadline,post_mid);steps=step+1;
        if step%16==15||step+1==limit {
            d.sync()?;completed=tensor::<2>(results.clone()).slice([0..2,0..1]).sum().into_scalar::<f32>()as usize;
            let error=tensor::<1>(world.meta.clone()).reshape([2,META]).slice([0..2,7..8]).max().into_scalar::<f32>();ensure!(error==0.,"environment capacity/error flag{error}");
            crate::artifacts::atomic_json(&root.join("departure-status.json"),&serde_json::json!({"pid":std::process::id(),"completed":completed,"target":2,"decisions":step+1,"training_updates":0}))?;
            if completed==2 {break;}
        }
    }
    ensure!(completed==2,"prefix failed to stop by deadline");
    if let Some(capture)=capture.as_ref(){capture.save(root,steps)?;}
    let timeline_data=tensor::<3>(timeline).slice([0..steps,0..4,0..TIMELINE_FIELDS]).into_data().try_to_vec::<f32>()?;
    std::fs::write(root.join("action-timeline.f32"),timeline_data.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>())?;
    let values=tensor::<2>(results).into_data().try_to_vec::<f32>()?;let counts=tensor::<2>(counts).into_data().try_to_vec::<f32>()?;let mut matches=vec![];
    for m in 0..2 {let row=&values[m*FIELDS..(m+1)*FIELDS];let mut sides=serde_json::Map::new();
        for side in 0..2 {let b=18+side*8;let at=(m*2+side)*22;sides.insert((side+2).to_string(),serde_json::json!({"first_mid_time":if row[b]>0.{Some(row[b]-world.engine.pre_game_seconds-1.)}else{None},"max_stuck_seconds":row[b+2],"stuck_seconds":row[b+3],"last_hits":row[4+side],"denies":row[6+side],"level":row[8+side],"health":row[10+side],"end_position":[row[b+4],row[b+5]],"actions_all":&counts[at..at+11],"actions_after_zero":&counts[at+11..at+22]}));}
        matches.push(serde_json::json!({"index":m+1,"learner_team":row[2]as usize+2,"end_time":row[3],"true_terminal":row[34]>0.5,"sides":sides}));
    }
    let report=serde_json::json!({"complete":true,"selection":if argmax{"argmax"}else{"sampling"},"same_selection_for_both_actors":true,"recurrent_reset_interval":128,"memory_reset_origin_seconds":reset_origin,"observation_storage":"f16","seed":config.seed,"training_updates":0,"matches":matches,"deadline_game_seconds":deadline,"post_mid_seconds":post_mid,"capture_start":capture_start,"scope":"Closed-loop bounded departure prefix; optional clock reset is diagnostic, not a production memory contract. No win-rate or strength conclusion.","environment_sha256":world.fingerprint});
    let mut report=report;report["action_timeline"]=serde_json::json!({"shape":[steps,4,TIMELINE_FIELDS],"columns":["valid","time","reset","x","y","health","action","buy","item","target","x_bin","y_bin","order","dest_x","dest_y"],"sha256":crate::runtime::file_sha256(&root.join("action-timeline.f32"))?});report["single_forward_intervention"]=serde_json::json!({"enabled":force_first_forward_move,"actor_marks":tensor::<2>(intervention).into_data().try_to_vec::<f32>()?,"mark_fields":["applied","game_tick","original_primary"],"coordinates":[127,128],"scope":"One first legal post-zero MOVE for the current-policy Radiant only; buy output preserved. GPU-only diagnostic command, not training or a production fix. Forced-command logp is not used for learning."});
    crate::artifacts::atomic_json(&root.join("departure.json"),&report)?;Ok(report)
}
