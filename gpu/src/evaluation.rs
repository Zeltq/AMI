//! Frozen-policy evaluation: one preselected first episode per slot, balanced sides.
use anyhow::{Result,ensure};
use burn::{module::Module,tensor::Tensor};
use cubecl::prelude::*;
use std::{path::Path,time::Instant,sync::{Arc,atomic::{AtomicBool,Ordering}}};
use crate::{contract::*,world::*,runtime::{device,primitive,tensor,argument},experience::{HORIZON,SIGNALS},sampling::Sampler};
pub const FIELDS:usize=35;

#[cube(launch)]
fn terminal(state:&[f32],meta:&[f32],signals:&[f32],results:&mut [f32],matches:usize,pregame:f32,deadline:f32,post_mid:f32){
    let m=ABSOLUTE_POS;
    if m<matches {
        let r=m*FIELDS;let a=m*2*SIGNALS;let mb=m*META;let p=m*UNITS*STRIDE;let q=p+STRIDE;
        if results[r]==0.0f32 {
            for side in 0..2 {
                let u=p+side*STRIDE;let b=r+18+side*8;let x=state[u+UNIT_X];let y=state[u+UNIT_Y];
                let time=meta[mb]/30.0f32+pregame;let alive=state[u+ACTIVE]>0.0f32 && state[u+UNIT_HEALTH]>0.0f32;
                if alive && results[b]==0.0f32 && x*x+y*y<4000000.0f32 && (x-y).abs()<1200.0f32 {results[b]=time+1.0f32;}
                if results[b+7]>0.0f32 {
                    let dt=((meta[mb]-results[b+6])/30.0f32).clamp(0.0f32,1.0f32);
                    let dx=x-results[b+4];let dy=y-results[b+5];let gx=state[u+DEST_X]-x;let gy=state[u+DEST_Y]-y;
                    let move_order=state[u+ORDER]==1.0f32 || state[u+ORDER]==2.0f32;
                    let eligible=alive && move_order && gx*gx+gy*gy>1024.0f32 && state[u+CAST_END]<=time && state[u+TP_END]<=time && state[u+FEAR_END]<=time;
                    if eligible && dx*dx+dy*dy<1.0f32 {
                        let previous=results[b+1];let run=previous+dt;results[b+1]=run;results[b+2]=results[b+2].max(run);
                        if run>=10.0f32 {results[b+3]+=if previous<10.0f32 {run}else{dt};}
                    }else{results[b+1]=0.0f32;}
                }
                results[b+4]=x;results[b+5]=y;results[b+6]=meta[mb];results[b+7]=1.0f32;
            }
        }
        let time=meta[mb]/30.0f32;
        let arrived=post_mid>0.0f32 && results[r+18]>0.0f32 && results[r+26]>0.0f32 && time+pregame+1.0f32>=results[r+18].max(results[r+26])+post_mid;
        if results[r]==0.0f32 && (signals[a+9]>0.0f32 || (deadline>0.0f32 && time>=deadline) || arrived) {
            results[r]=1.0f32;results[r+1]=meta[mb+17];results[r+2]=if signals[a+11]>0.5f32{0.0f32}else{1.0f32};
            results[r+3]=if deadline>0.0f32 {meta[mb]/30.0f32}else{f32::max(0.0f32,meta[mb])/30.0f32};
            results[r+4]=state[p+UNIT_LAST_HITS];results[r+5]=state[q+UNIT_LAST_HITS];
            results[r+6]=state[p+UNIT_DENIES];results[r+7]=state[q+UNIT_DENIES];
            results[r+8]=state[p+UNIT_LEVEL];results[r+9]=state[q+UNIT_LEVEL];
            results[r+10]=state[p+UNIT_HEALTH];results[r+11]=state[q+UNIT_HEALTH];
            results[r+12]=state[p+2*STRIDE+UNIT_HEALTH];results[r+13]=state[p+4*STRIDE+UNIT_HEALTH];
            results[r+14]=meta[mb+3]/30.0f32;results[r+15]=m as f32;
            results[r+16]=state[p+DEATHS];results[r+17]=state[q+DEATHS];
            results[r+34]=if signals[a+9]>0.0f32 {1.0f32}else{0.0f32};
        }
    }
}

pub fn record(world:&World,signals:Tensor<2>,results:&burn_cubecl::tensor::CubeTensor){
    record_prefix(world,signals,results,0.,0.);
}
pub fn record_prefix(world:&World,signals:Tensor<2>,results:&burn_cubecl::tensor::CubeTensor,deadline:f32,post_mid:f32){
    let signals=primitive(signals);
    terminal::launch(&world.state.client,CubeCount::Static(world.matches.div_ceil(32)as u32,1,1),CubeDim::new_1d(32),
        argument(&world.state),argument(&world.meta),argument(&signals),argument(results),world.matches,world.engine.pre_game_seconds,deadline,post_mid);
}

fn wilson(wins:usize,n:usize)->[f64;2]{
    let z=1.959963984540054;let p=wins as f64/n as f64;let denominator=1.+z*z/n as f64;
    let center=(p+z*z/(2.*n as f64))/denominator;let radius=z*(p*(1.-p)/n as f64+z*z/(4.*(n*n)as f64)).sqrt()/denominator;
    [(center-radius).max(0.),(center+radius).min(1.)]
}

pub fn run(config:crate::learner::Config,root:&Path,traces:bool,reset_origin:Option<f32>,argmax:bool)->Result<serde_json::Value>{
    ensure!(reset_origin.is_none_or(f32::is_finite),"finite reset clock required");
    ensure!(config.matches>=2 && config.matches%2==0,"use an even match count for balanced evaluation");
    ensure!(config.opponents.len()==1 && config.initial_checkpoint.is_some(),"evaluation needs one frozen model and one frozen opponent");
    std::fs::create_dir_all(root)?;
    let lock=std::fs::OpenOptions::new().create(true).truncate(false).read(true).write(true).open(root.join("evaluation.lock"))?;
    lock.try_lock().map_err(|e|anyhow::anyhow!("evaluation already locked: {e}"))?;
    ensure!(!root.join("evaluation-config.json").exists(),"evaluation output already used; select a new cohort directory");
    crate::artifacts::atomic_json(&root.join("evaluation-config.json"),&config)?;
    crate::artifacts::atomic_json(&root.join("evaluation-runtime.json"),&serde_json::json!({"selection":if argmax{"argmax"}else{"sampling"},"memory_reset_origin_seconds":reset_origin,"recurrent_reset_interval":128,"observation_storage":"f16"}))?;
    let engine=crate::artifacts::freeze_environment(Path::new(&config.engine),root,false)?;
    let executable=std::env::current_exe()?;let saved_executable=root.join(executable.file_name().unwrap());
    if executable.canonicalize()?!=saved_executable.canonicalize().unwrap_or_default(){std::fs::copy(&executable,saved_executable)?;}
    let d=device();d.seed(config.seed);let world=World::load(&engine,config.matches,&d)?;let n=config.matches*2;
    let policy=crate::selfplay::load(Path::new(config.initial_checkpoint.as_ref().unwrap()),&d)?.valid();
    let pool=crate::selfplay::Pool::evaluation(&config.opponents[0],config.matches,&d)?;
    let sampler=Sampler::new(n,config.seed as u32,&d).with_argmax(argmax);let rng=Sampler::new(config.matches,config.seed as u32+1,&d);
    let results=primitive(Tensor::<2>::zeros([config.matches,FIELDS],&d));
    let mut previous=crate::observe::initial_previous(n,&d);let mut h=Tensor::<2>::zeros([n,256],&d);let mut c=h.clone();
    let writer=crate::artifacts::Writer::new(root)?;let trace=if traces{Some(crate::trace::Trace::new(&world))}else{None};
    let stop=Arc::new(AtomicBool::new(false));let stopped=stop.clone();ctrlc::set_handler(move||stopped.store(true,Ordering::Release))?;
    let started=Instant::now();let limit=((world.engine.max_seconds+world.engine.pre_game_seconds)*7.5).ceil()as usize+HORIZON;
    let mut completed=0;let mut decisions=0;let mut interrupted=false;
    for block in 0..limit.div_ceil(HORIZON) {
        for step in 0..HORIZON {
            let mut observed=world.observe_with_clock(previous.clone(),reset_origin);observed.packed=observed.packed.cast(burn::tensor::DType::F16);
            let keep:Tensor<2>=1.0f32-observed.reset.clone().unsqueeze_dim::<2>(1);let prior_h=h.clone()*keep.clone();let prior_c=c.clone()*keep;
            let b=observed.decode();let encoded=policy.encode(&b);let (next_h,next_c)=policy.memory.step(encoded.fused,prior_h.clone(),prior_c.clone());
            let heads=policy.heads(&b,next_h.clone(),encoded.keys,None,&sampler);
            let (next_h,next_c,heads,learn,versions)=pool.decide(&world,&b,prior_h,prior_c,next_h,next_c,heads,&sampler,1);h=next_h;c=next_c;
            let actions=heads.actions.clone().float();let next_previous=observed.previous(heads.actions.clone(),&world.bases,&world.state);
            let (reward,done)=world.advance(heads.actions,&observed.refs,&rng);
            previous=next_previous.mask_where(done.clone().greater_elem(0.5).unsqueeze_dim::<2>(1),crate::observe::initial_previous(n,&d));
            let signals=Tensor::cat(vec![actions,heads.logp.unsqueeze_dim::<2>(1),heads.value.unsqueeze_dim::<2>(1),reward.unsqueeze_dim::<2>(1),done.unsqueeze_dim::<2>(1),observed.reset.unsqueeze_dim::<2>(1),learn,versions],1);
            record(&world,signals.clone(),&results);if let Some(trace)=trace.as_ref(){trace.record(&world,step,signals);}
        }
        decisions+=HORIZON;d.sync()?;writer.check()?;
        completed=tensor::<2>(results.clone()).slice([0..config.matches,0..1]).sum().into_scalar::<f32>()as usize;
        let error=tensor::<1>(world.meta.clone()).reshape([config.matches,META]).slice([0..config.matches,7..8]).max().into_scalar::<f32>();ensure!(error==0.,"GPU evaluation environment error{error}");
        if let Some(trace)=trace.as_ref(){writer.trace(&trace.buffer,trace.header(&world,block as u64+1),block as u64+1)?;}
        crate::artifacts::atomic_json(&root.join("evaluation-status.json"),&serde_json::json!({"phase":"evaluating","pid":std::process::id(),"completed":completed,"target":config.matches,"decisions":decisions,"wall_seconds":started.elapsed().as_secs_f64(),"artifacts":writer.status()}))?;
        if completed==config.matches {break;}
        if stop.load(Ordering::Acquire)||root.join("stop.json").exists(){interrupted=true;break;}
    }
    let artifacts=writer.status();writer.finish()?;let wall=started.elapsed().as_secs_f64();
    let values=tensor::<2>(results).into_data().try_to_vec::<f32>()?;let mut games=vec![];let mut wins=0;let mut draws=0;let mut radiant_wins=0;let mut side_wins=[0usize;2];let mut side_games=[0usize;2];
    let mut lh=[0f64;2];let mut denies=[0f64;2];let mut duration=0f64;
    for (index,row)in values.chunks_exact(FIELDS).enumerate(){
        if row[0]<0.5 {continue;}
        let learner=row[2]as usize;let winner=if row[1]>0.{Some(2)}else if row[1]<0.{Some(3)}else{None};
        side_games[learner]+=1;let won=winner==Some(learner+2);if won{wins+=1;side_wins[learner]+=1;}if winner.is_none(){draws+=1;}if winner==Some(2){radiant_wins+=1;}
        let reason=if row[16]>0.||row[17]>0.||row[10]<=0.||row[11]<=0.{"death"}else if row[12]<=0.||row[13]<=0.{"tower"}else{"last_hits"};
        lh[0]+=row[4+learner]as f64;lh[1]+=row[4+1-learner]as f64;denies[0]+=row[6+learner]as f64;denies[1]+=row[6+1-learner]as f64;duration+=row[3]as f64;
        let mut sides=serde_json::Map::new();
        for side in 0..2 {let b=18+side*8;sides.insert((side+2).to_string(),serde_json::json!({"last_hits":row[4+side],"denies":row[6+side],"level":row[8+side],"deaths":row[16+side],"first_mid_time":if row[b]>0.{Some(row[b]-world.engine.pre_game_seconds-1.)}else{None},"max_stuck_seconds":row[b+2],"stuck_seconds":row[b+3]}));}
        games.push(serde_json::json!({"index":index+1,"model_team":learner+2,"outcome":{"winner":winner,"reason":reason,"time":row[3]},"time":row[3],"sides":sides}));
    }
    let complete=completed==config.matches;let count=completed.max(1)as f64;let score=(wins as f64+draws as f64*0.5)/count;
    ensure!(!complete || side_games==[config.matches/2;2],"evaluation sides are unbalanced");
    let radius=(40f64.ln()/(2.*(config.matches/2)as f64)).sqrt();
    let report=serde_json::json!({"complete":complete,"interrupted":interrupted,"matches":completed,"target_matches":config.matches,"wins":wins,"draws":draws,"losses":completed-wins-draws,"score":if complete{Some(score)}else{None},
        "score_95_percent_interval":if complete{Some([(score-radius).max(0.),(score+radius).min(1.)])}else{None},"interval_method":"Hoeffding bound over independent alternating-side pair scores; no strength estimate for incomplete cohorts",
        "map_sides":{"radiant_wins":radiant_wins,"dire_wins":completed-radiant_wins-draws,"draws":draws,"radiant_win_fraction_95_percent_interval":if completed>0{Some(wilson(radiant_wins,completed))}else{None}},
        "learner_side_wins":side_wins,"learner_side_games":side_games,"mean_last_hits":lh.map(|v|v/count),"mean_denies":denies.map(|v|v/count),"mean_game_seconds":duration/count,
        "games":games,"wall_seconds":wall,"decisions":decisions,"learning_updates":0,"seed":config.seed,"decision_hz":7.5,"simulation_hz":30,"recurrent_reset_interval":128,"memory_reset_origin_seconds":reset_origin,"selection":if argmax{"argmax"}else{"sampling"},"observation_storage":"f16","memory_pool":format!("{:?}",d.memory_pool_usage()),
        "environment_sha256":world.fingerprint,"executable_sha256":crate::runtime::file_sha256(&executable)?,"opponent_sha256":pool.identities,"artifacts_before_drain":artifacts,"trace_export_drained":true,
        "cohort":"Exactly the first episode of each preselected slot. Finished slots may keep simulating, but no later episode enters the evaluation; no length-based exclusion."});
    let mut report=report;
    report["behavior_definition"]=serde_json::json!({"first_mid":"First alive frame within2000 of map origin and abs(x-y)<1200, same rule as native evaluator; negative pregame times allowed","stuck":"Active movement order1/2, destination>32units away, displacement<1unit per decision continuously>=10s; excludes dead/casting/TP/fear. Sampled at7.5Hz; intentionally idle/attacking actors are not called stuck."});
    crate::artifacts::atomic_json(&root.join(if complete{"evaluation.json"}else{"partial-evaluation.json"}),&report)?;
    crate::artifacts::atomic_json(&root.join("evaluation-status.json"),&serde_json::json!({"phase":if complete{"complete"}else if interrupted{"stopped"}else{"failed"},"pid":std::process::id(),"completed":completed,"target":config.matches,"wall_seconds":wall}))?;
    ensure!(complete,"incomplete fixed evaluation: {completed}/{}",config.matches);Ok(report)
}
