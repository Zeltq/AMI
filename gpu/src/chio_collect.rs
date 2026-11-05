//! CHiO teacher/DAgger collection through the same labels, masks and executor.
use anyhow::{Result,ensure};
use burn::{module::Module,tensor::{Tensor,TensorData,Int,DType}};
use std::{path::Path,time::Instant};
use crate::{world::*,contract::*,observe,chio_observe as co,chio_actions as ca,chio_reward as reward,
    runtime::{device,tensor,primitive},imitation::{CollectConfig,DaggerConfig,DemoWriter},experience::{Experience,HORIZON,SIGNALS},sampling::Sampler};

pub fn eligibility(w:&World,t:&crate::scripted_teacher::Teacher)->Tensor<1>{
    let n=w.matches*2;let state=tensor::<1>(w.state.clone()).reshape([w.matches,UNITS,STRIDE]).slice([0..w.matches,0..2,0..STRIDE]).reshape([n,STRIDE]);
    let alive=state.clone().slice([0..n,ACTIVE..ACTIVE+1]).greater_elem(0.).bool_and(state.slice([0..n,UNIT_HEALTH..UNIT_HEALTH+1]).greater_elem(0.));
    let active=tensor::<1>(t.audit.clone()).reshape([w.matches,crate::scripted_teacher::AUDIT_FIELDS]).slice([0..w.matches,0..1]).equal_elem(0.).expand([w.matches,2]).reshape([n,1]);
    alive.bool_and(active).float().reshape([n])
}

pub fn collect(cfg:CollectConfig,root:&Path,dagger:Option<DaggerConfig>)->Result<serde_json::Value>{
    ensure!(cfg.games>0 && cfg.parallel_matches>0 && cfg.seconds>0. && cfg.seconds<=600.,"invalid CHiO collection dimensions");
    ensure!(cfg.teacher_version==crate::teacher_v10::Version::V10,"CHiO labels require v10");crate::teacher_v10::check_pin()?;
    std::fs::create_dir_all(root)?;ensure!(!root.join("config.json").exists(),"fresh CHiO dataset directory required");
    let lock=std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(root.join("dataset.lock"))?;lock.try_lock()?;
    crate::artifacts::atomic_json(&root.join("config.json"),&cfg)?;
    let engine=crate::artifacts::freeze_environment(Path::new(&cfg.engine),root,false)?;
    let d=device();let zero_sum=cfg.zero_sum.unwrap_or(true);let started=Instant::now();let mut writer=DemoWriter::new(root);
    if let Some(dagger)=&dagger {writer.source=format!("dagger-round-{}",dagger.round);crate::artifacts::atomic_json(&root.join("dagger.json"),dagger)?;}
    let policy=dagger.as_ref().map(|x|crate::chio_bridge::import(Path::new(&x.weights),&d).map(|p|p.valid())).transpose()?;
    let mut games=vec![];let mut block=0;let mut partial=false;
    'cohorts: for offset in (0..cfg.games).step_by(cfg.parallel_matches) {
        let count=cfg.parallel_matches.min(cfg.games-offset);let n=count*2;let w=World::load(&engine,count,&d)?;w.validate_profile("chio-v1")?;
        let mut recorder=reward::Recorder::new(&w,zero_sum);let mut exp=Experience::with_width(n,co::WIDTH,32,&d);
        let rng=Sampler::new(count,cfg.seed.wrapping_add(offset as u32*7919),&d);let sampler=Sampler::new(n,cfg.seed.wrapping_add(offset as u32*7919+1),&d);
        let mixing=Sampler::new(n,cfg.seed.wrapping_add(offset as u32*7919+2),&d);
        let learner=Tensor::<1,Int>::from_data(TensorData::new((0..n).map(|a|i32::from(a%2==(offset+a/2)%2)).collect::<Vec<_>>(),[n]),&d).greater_elem(0).unsqueeze_dim::<2>(1);
        let mut h=Tensor::<2>::zeros([n,256],&d);let mut c=h.clone();
        let limit=((w.engine.pre_game_seconds+cfg.seconds)*w.decision_hz()).ceil()as usize+2;let mut stored=0;
        let mut runtime=crate::chio_runtime::Runtime::new(w,cfg.teacher_version,cfg.teacher_version,false,false,zero_sum)?;
        for step in 0..limit {
            if step%128==0 && step>0 {exp.signals=primitive(Tensor::<3>::zeros([n,HORIZON,SIGNALS],&d));}
            let frame=runtime.observe();let obs=&frame.observation;let valid=frame.valid.clone();let labels=frame.labels.clone();
            let mut chosen=labels.clone();let mut extra=Tensor::<2>::zeros([n,3],&d);
            if let (Some(policy),Some(dagger))=(&policy,&dagger) {
                let keep:Tensor<2>=1.0f32-obs.reset.clone().unsqueeze_dim::<2>(1);h=h*keep.clone();c=c*keep;
                let (batch,model_masks)=co::decode(obs,observe::CREEPS,observe::TREES,observe::PROJECTILES);let encoded=policy.encode(&batch);
                (h,c)=policy.memory.step(encoded.fused,h,c);let predicted=policy.heads(&model_masks,h.clone(),encoded.keys,None,&sampler).actions;
                let student=if dagger.beta==1. {Tensor::<2>::zeros([n,1],&d).greater_elem(0.)}else if dagger.beta==0. {learner.clone()}else{
                    let probabilities=Tensor::cat(vec![Tensor::full([n,1],dagger.beta.ln(),&d),Tensor::full([n,1],(1.-dagger.beta).ln(),&d)],1);
                    learner.clone().bool_and(mixing.draw(probabilities).greater_elem(0).unsqueeze_dim::<2>(1))};
                let disagreement=predicted.clone().not_equal(labels.clone()).int().sum_dim(1).greater_elem(0).float();
                chosen=chosen.mask_where(student.clone(),predicted);extra=Tensor::cat(vec![student.bool_not().float(),disagreement,learner.clone().float()],1);
            }
            let transition=runtime.advance(&frame,chosen,&rng,cfg.seconds);
            recorder.store(&transition.before,&transition.after,&transition.reward,valid.clone(),step%128);
            let signals=Tensor::cat(vec![labels.float(),Tensor::zeros([n,2],&d),extra,transition.done.unsqueeze_dim::<2>(1),obs.reset.clone().unsqueeze_dim::<2>(1),valid.unsqueeze_dim::<2>(1),Tensor::full([n,1],5.,&d)],1);
            exp.store(step%128,obs,signals);stored+=1;
            if stored==128 || step+1==limit {
                writer.store(&exp,offset,block)?;recorder.flush(root,block,offset,stored)?;block+=1;stored=0;
                let audit=runtime.teacher.values();let finished=audit.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).filter(|v|v[0]>0.).count();
                ensure!(audit.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).all(|v|v[39]==0.),"CHiO simulation/label error");
                eprintln!("CHiO block={block} cohort={offset} finished={finished}/{count} game_seconds={} wall_seconds={:.3}",audit[1],started.elapsed().as_secs_f64());
                crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"stage":"collecting-chio","pid":std::process::id(),"blocks":block,"cohort_offset":offset,"finished":finished,"cohort_games":count,"game_seconds":audit[1],"wall_seconds":started.elapsed().as_secs_f64()}))?;
                if finished==count {
                    let cohort=runtime.games(offset)?;
                    for mut game in cohort {game["split"]=serde_json::json!(crate::imitation::split(game["game"].as_u64().unwrap()as usize));games.push(game);}break;
                }
                if cfg.max_blocks.is_some_and(|max|block>=max)||root.join("stop.json").exists(){partial=true;break 'cohorts;}
            }
        }
    }
    let files=writer.finish()?;let frames=files.iter().map(|e|e.valid_frames).sum::<usize>();let completed=games.iter().filter(|g|g["native_terminal"]==true).count();
    let dataset=serde_json::json!({"family":"CHiO","observation_profile":"chio-v1","teacher_sha256":cfg.teacher_version.sha256(),"source_sha256":crate::runtime::bytes_sha256(include_bytes!("chio_observe.rs")),
        "config":cfg,"partial":partial,"completed_matches":completed,"elapsed_seconds":started.elapsed().as_secs_f64(),"files":files,"game_metrics":games,
        "decision_ticks":6,"offset_ticks":6,"reward_mode":"lastorder-fb-v1","zero_sum":zero_sum,"gamma":reward::GAMMA,"gae_lambda":reward::LAMBDA,"reward_record_format":"chio-reward-v1"});
    crate::artifacts::atomic_json(&root.join("dataset.json"),&dataset)?;crate::artifacts::atomic_json(&root.join("game-metrics.json"),&games)?;
    let result=serde_json::json!({"family":"CHiO","games_requested":cfg.games,"games_recorded":games.len(),"native_completed":completed,"valid_decisions":frames,"blocks":block,"partial":partial,"wall_seconds":started.elapsed().as_secs_f64()});
    crate::artifacts::atomic_json(&root.join("report.json"),&result)?;Ok(result)
}
