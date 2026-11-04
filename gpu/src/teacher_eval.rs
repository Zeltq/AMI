//! Teacher versus teacher evaluation, using the production delayed executor.
use anyhow::{Result,ensure};
use burn::tensor::Tensor;
use std::{path::Path,time::Instant};
use crate::{world::*,observe,runtime::{device,tensor},sampling::Sampler,scripted_teacher::{Teacher,AUDIT_FIELDS},teacher_v10::{Controller,Version},teacher_combat as combat};

pub fn run(engine:&Path,root:&Path,matches:usize,seconds:f32,seed:u32,left:Version,right:Version,swap:bool,trace:bool)->Result<serde_json::Value>{
    ensure!(matches>0 && seconds>0. && seconds<=600.,"positive matches and seconds <=600 required");
    ensure!(!swap || matches%2==0,"alternating sides requires even matches");
    if left==Version::V10 || right==Version::V10 {crate::teacher_v10::check_pin()?;}
    std::fs::create_dir_all(root)?;ensure!(!root.join("config.json").exists(),"choose a fresh evaluation directory");
    crate::artifacts::atomic_json(&root.join("config.json"),&serde_json::json!({"engine":engine,"matches":matches,"seconds":seconds,"seed":seed,"left":left,"right":right,"swap":swap,"trace":trace,"left_sha256":left.sha256(),"right_sha256":right.sha256(),"command":std::env::args().collect::<Vec<_>>()}))?;
    let frozen=crate::artifacts::freeze_environment(engine,root,false)?;
    let d=device();let w=World::load(&frozen,matches,&d)?;let t=Teacher::new(&w);let controller=Controller::new(&w,left,right,swap);let rng=Sampler::new(matches,seed,&d);
    let metrics=crate::ggd_metrics::Metrics::new(&w);metrics.sample(&w,&t);
    let mut potential=crate::potential_reward::Tracker::new(&w,&t,0.5);
    let trace=trace.then(||crate::trace::Trace::new(&w));
    let mut previous=observe::initial_previous(matches*2,&d);let started=Instant::now();let mut steps=0;
    let limit=((w.engine.pre_game_seconds+seconds)*7.5).ceil()as usize+2;
    for step in 0..limit {
        let obs=w.observe(previous.clone());let actions=controller.actions(&t,&w,&obs);
        previous=obs.previous(actions.clone(),&w.bases,&w.state);metrics.commands(&w,&t,actions.clone(),&obs.refs);
        let (reward,done)=w.advance_teacher(actions.clone(),&obs.refs,&rng,&t,true);t.record(&w,seconds);metrics.sample(&w,&t);
        let _=potential.step(&w,&t);
        if let Some(trace)=&trace {
            let signals=Tensor::cat(vec![actions.float(),Tensor::zeros([matches*2,2],&d),reward.unsqueeze_dim::<2>(1),done.unsqueeze_dim::<2>(1),obs.reset.unsqueeze_dim::<2>(1),Tensor::ones([matches*2,2],&d)],1);
            trace.record(&w,step%crate::experience::HORIZON,signals);
        }
        steps=step+1;
        if steps%crate::experience::HORIZON==0 || steps==limit {
            if let Some(trace)=&trace {
                // Write the partial final block explicitly; never export stale tail frames.
                let n=(steps-1)%crate::experience::HORIZON+1;
                let values=tensor::<3>(trace.buffer.clone()).slice([0..n,0..trace.matches,0..trace.width]).into_data().try_to_vec::<f32>()?;
                let mut header=trace.header(&w,steps.div_ceil(crate::experience::HORIZON)as u64);header["frames"]=n.into();
                use std::io::Write;
                let mut out=flate2::write::GzEncoder::new(std::fs::File::create(root.join(format!("matches-{:08}.sfduel.gz",steps.div_ceil(crate::experience::HORIZON))))?,flate2::Compression::fast());
                let h=serde_json::to_vec(&header)?;out.write_all(b"SFDUEL02")?;out.write_all(&(h.len()as u32).to_le_bytes())?;out.write_all(&h)?;
                let bytes:Vec<_>=values.iter().flat_map(|v|v.to_le_bytes()).collect();out.write_all(&bytes)?;out.finish()?;
            }
            let values=t.values();let finished=values.chunks_exact(AUDIT_FIELDS).filter(|v|v[0]>0.).count();
            eprintln!("teacher-eval steps={steps} finished={finished}/{matches} wall={:.2}s",started.elapsed().as_secs_f64());
            crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"pid":std::process::id(),"steps":steps,"finished":finished,"matches":matches,"game_seconds":values[1],"wall_seconds":started.elapsed().as_secs_f64()}))?;
            if finished==matches || root.join("stop.json").exists() {break;}
        }
    }
    let audit=t.values();let mut games=metrics.games(&audit,w.engine.pre_game_seconds,0,None);
    crate::demo_telemetry::enrich(&w,&mut games)?;
    potential.enrich(&mut games)?;
    let events=tensor::<1>(w.teacher_combat.clone()).into_data().try_to_vec::<f32>()?;
    let meta=tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>()?;
    for (m,game) in games.iter_mut().enumerate(){
        game["result"]=serde_json::json!(if game["native_terminal"]==true {Some(meta[m*META+17]as i32)}else{None});
        for side in 0..2 {let selected=if swap && m%2==1 {1-side}else{side};let r=(m*2+side)*combat::FIELDS;
            let row=&mut game["sides"][side];row["controller"]=serde_json::to_value(if selected==0 {left}else{right})?;
            for (key,f) in [("damage_from_hero",combat::HERO_DAMAGE),("damage_from_creeps",combat::CREEP_DAMAGE),("damage_from_tower",combat::TOWER_DAMAGE),("hero_attacks_released",combat::HERO_ATTACKS),("hero_attack_orders_executed",combat::HERO_ORDERS),("hero_orders_causing_creep_aggro",combat::AGGRO_ORDERS),("raze_creep_last_hits",combat::RAZE_KILLS)] {row[key]=serde_json::json!(events[r+f]);}
        }
    }
    ensure!(games.iter().all(|g|g["simulation_error"]==0.),"simulation error");
    let result=serde_json::json!({"matches":matches,"seed":seed,"left":left,"right":right,"swap":swap,"steps":steps,"complete":audit.chunks_exact(AUDIT_FIELDS).all(|v|v[0]>0.),"wall_seconds":started.elapsed().as_secs_f64(),"per_match":games});
    crate::artifacts::atomic_json(&root.join("report.json"),&result)?;Ok(result)
}
