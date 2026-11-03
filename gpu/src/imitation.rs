//! GGD demonstration recording and recurrent behavior cloning on the existing
//! GPU pipeline. Collection uses Vulkan/HIP; recurrent IL uses PyTorch ROCm/CUDA.
//! Only block snapshots and aggregate simulation metrics reach CPU.
use anyhow::{Result,ensure};
use burn::tensor::{Tensor,TensorData,Int,DType};
use burn::module::Module;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use serde::{Serialize,Deserialize};
use std::{path::Path,io::Write,time::Instant,sync::mpsc::{sync_channel,SyncSender},thread::JoinHandle};
use crate::{contract::*,observe,world::*,experience::{Experience,HORIZON,SIGNALS},runtime::{device,primitive,tensor,argument},sampling::Sampler};

const MAGIC:&[u8;8]=b"GGDDEMO2";
#[derive(Clone,Serialize,Deserialize,Debug)]
#[serde(default,deny_unknown_fields)]
pub struct CollectConfig {
    pub teacher_version:crate::teacher_v10::Version,pub engine:String,pub games:usize,pub parallel_matches:usize,pub seconds:f32,pub seed:u32,pub max_blocks:Option<usize>,
}
impl Default for CollectConfig {fn default()->Self{Self{teacher_version:crate::teacher_v10::Version::V9,engine:"runs/gpu/dota-calibration-003/engine.json".into(),games:1024,parallel_matches:100,seconds:600.,seed:20261102,max_blocks:None}}}
/// Both sides of a game always share one split: 75% / 12.5% / 12.5%.
pub fn split(game:usize)->&'static str {match game%8 {6=>"validation",7=>"test",_=>"train"}}

#[derive(Clone,Serialize,Deserialize)]
struct Header {format_version:usize,source:String,version:usize,actors:Vec<usize>,columns:Vec<i32>,buckets:[usize;3],horizon:usize,signal_fields:usize,observation_dtype:String}
#[derive(Clone,Serialize,Deserialize)]
struct Entry {file:String,sha256:String,bytes:u64,valid_frames:usize,attacks:usize,actors:Vec<usize>}
#[derive(Serialize,Deserialize)]
struct Dataset {family:String,observation_profile:String,teacher_sha256:String,source_sha256:String,config:CollectConfig,partial:bool,
    completed_matches:usize,elapsed_seconds:f64,files:Vec<Entry>,game_metrics:Vec<serde_json::Value>}

pub fn columns(pc:usize,pt:usize,pp:usize)->Vec<i32>{
    use observe::*;
    assert!(pc<=CREEPS&&pt<=TREES&&pp<=PROJECTILES);
    let mut out=vec![];let mut add=|start:usize,len:usize|out.extend((start..start+len).map(|v|v as i32));
    add(HERO_OFFSET,4*UNIT_COUNT);add(CREEP_OFFSET,pc*2*UNIT_COUNT);add(TOWER_OFFSET,8*UNIT_COUNT);add(TREE_OFFSET,pt*2*TREE_COUNT);
    add(PROJECTILE_OFFSET,pp*2*PROJECTILE_COUNT);add(SKILL_OFFSET,CREEP_VALID-SKILL_OFFSET);
    add(CREEP_VALID,pc);add(CREEP_ALLY,pc);add(TREE_VALID,pt);add(PROJECTILE_VALID,pp);add(ACTION_MASK,ATTACK_MASK-ACTION_MASK);
    for at in [ATTACK_MASK,TREE_MASK,TP_MASK,SELF_MASK] {add(at,2+pc);add(at+2+CREEPS,4+pt);add(at+TARGETS-1,1);}
    add(COMBAT_UNIT_OFFSET,4*COMBAT_UNIT_COUNT);add(COMBAT_UNIT_OFFSET+4*COMBAT_UNIT_COUNT,pc*2*COMBAT_UNIT_COUNT);
    add(COMBAT_UNIT_OFFSET+(2+CREEPS)*2*COMBAT_UNIT_COUNT,8*COMBAT_UNIT_COUNT);add(COMBAT_PROJECTILE_OFFSET,pp*2*COMBAT_PROJECTILE_COUNT);
    out
}

struct Snapshot {name:String,header:Header,observations:CubeTensor,signals:CubeTensor}
struct DemoWriter {source:String,sender:SyncSender<Snapshot>,worker:JoinHandle<Result<Vec<Entry>>>}
impl DemoWriter {
    fn new(root:&Path)->Self {
        let (sender,receiver)=sync_channel::<Snapshot>(2);let root=root.to_path_buf();
        let worker=std::thread::spawn(move||->Result<Vec<Entry>>{let mut entries=vec![];
            while let Ok(job)=receiver.recv(){
                let obs=job.observations.client.read_one(job.observations.handle)?;let sig=job.signals.client.read_one(job.signals.handle)?;
                let values:Vec<f32>=sig.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
                let valid_frames=values.chunks_exact(SIGNALS).filter(|v|v[11]>0.5).count();
                if valid_frames==0{continue;}
                let attacks=values.chunks_exact(SIGNALS).filter(|v|v[11]>0.5&&v[0]==3.).count();
                let file=root.join(&job.name);let temporary=file.with_extension("pending");
                let mut zip=flate2::write::GzEncoder::new(std::fs::File::create(&temporary)?,flate2::Compression::fast());
                let header=serde_json::to_vec(&job.header)?;zip.write_all(MAGIC)?;zip.write_all(&(header.len()as u32).to_le_bytes())?;zip.write_all(&header)?;
                zip.write_all(&obs)?;zip.write_all(&sig)?;zip.finish()?;std::fs::rename(temporary,&file)?;
                entries.push(Entry{file:job.name,sha256:crate::runtime::file_sha256(&file)?,bytes:std::fs::metadata(file)?.len(),valid_frames,attacks,actors:job.header.actors});
            }Ok(entries)
        });Self{source:"teacher-demo".into(),sender,worker}
    }
    fn store(&self,experience:&Experience,game_offset:usize,block:usize)->Result<()> {
        let (pc,pt,pp)=experience.buckets();let cols=columns(pc,pt,pp);let d=device();
        let indices=Tensor::<1,Int>::from_data(TensorData::new(cols.clone(),[cols.len()]),&d);let mut actor_offset=0;
        for (index,shard) in experience.observations.iter().enumerate(){let n=shard.meta.shape()[0];
            let observations=primitive(tensor::<3>(shard.clone()).select(2,indices.clone()));
            let signals=primitive(experience.signals().slice([actor_offset..actor_offset+n,0..HORIZON,0..SIGNALS])+0.0f32);
            let header=Header{format_version:3,source:self.source.clone(),version:GGD_SPEC_VERSION,actors:(actor_offset..actor_offset+n).map(|a|game_offset*2+a).collect(),columns:cols.clone(),buckets:[pc,pt,pp],horizon:HORIZON,signal_fields:SIGNALS,observation_dtype:"f16_le".into()};
            self.sender.send(Snapshot{name:format!("block-{block:06}-shard-{index:03}.ggd.gz"),header,observations,signals})?;actor_offset+=n;
        }Ok(())
    }
    fn finish(self)->Result<Vec<Entry>>{drop(self.sender);self.worker.join().map_err(|_|anyhow::anyhow!("demonstration writer panicked"))?}
}

#[cube(launch)]
fn valid(state:&[f32],audit:&[f32],out:&mut [f32],actors:usize){let a=ABSOLUTE_POS;if a<actors{let p=(a/2*UNITS+a%2)*STRIDE;out[a]=if audit[a/2*crate::scripted_teacher::AUDIT_FIELDS]==0.0f32&&state[p+ACTIVE]>0.0f32&&state[p+UNIT_HEALTH]>0.0f32{1.0f32}else{0.0f32};}}

#[derive(Clone,Serialize,Deserialize,Debug)]
#[serde(deny_unknown_fields)]
pub struct DaggerConfig {pub collection:CollectConfig,pub weights:String,pub beta:f32,pub round:usize}

pub fn collect(config:CollectConfig,root:&Path)->Result<serde_json::Value>{collect_impl(config,root,None)}
pub fn collect_dagger(config:DaggerConfig,root:&Path)->Result<serde_json::Value>{
    ensure!((0.0..=1.0).contains(&config.beta)&&config.round>0,"beta in [0,1] and positive round required");
    ensure!(config.collection.games%2==0,"balanced learner sides require even games");
    collect_impl(config.collection.clone(),root,Some(config))
}
fn collect_impl(config:CollectConfig,root:&Path,dagger:Option<DaggerConfig>)->Result<serde_json::Value>{
    ensure!(config.games>0&&config.parallel_matches>0&&config.seconds>0.&&config.seconds<=600.,"positive games/parallel count and 0 < seconds <= 600 required");
    std::fs::create_dir_all(root)?;ensure!(!root.join("config.json").exists(),"choose a fresh dataset directory");
    let lock=std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(root.join("dataset.lock"))?;lock.try_lock()?;
    crate::artifacts::atomic_json(&root.join("config.json"),&config)?;let engine=crate::artifacts::freeze_environment(Path::new(&config.engine),root,false)?;
    if config.teacher_version==crate::teacher_v10::Version::V10 {crate::teacher_v10::check_pin()?;}
    let teacher_sha=config.teacher_version.sha256();let started=Instant::now();let d=device();
    let mut writer=DemoWriter::new(root);
    let policy=if let Some(cfg)=dagger.as_ref(){
        writer.source=format!("dagger-round-{}",cfg.round);crate::artifacts::atomic_json(&root.join("dagger.json"),cfg)?;
        let model=crate::bridge::import(Path::new(&cfg.weights),&d)?.valid();ensure!(model.is_ggd(),"GGD weights required");Some(model)
    }else{None};
    let mut learner_frames=0usize;let mut disagreements=0usize;let mut teacher_executed=0usize;
    let mut block=0usize;let mut game_metrics=vec![];let mut partial=false;let mut completed=0usize;
    'cohorts:for offset in (0..config.games).step_by(config.parallel_matches){
        if config.max_blocks.is_some_and(|limit|block>=limit){partial=true;break;}
        let games=config.parallel_matches.min(config.games-offset);let n=games*2;let w=World::load(&engine,games,&d)?;
        let teacher=crate::scripted_teacher::Teacher::new(&w);let metrics=crate::ggd_metrics::Metrics::new(&w);metrics.sample(&w,&teacher);let observer=crate::combat_observe::CombatObserver::new(&w);let rng=Sampler::new(games,config.seed.wrapping_add(offset as u32*7919),&d);
        let controller=crate::teacher_v10::Controller::new(&w,config.teacher_version,config.teacher_version,false);
        let sampler=Sampler::new(n,config.seed.wrapping_add(offset as u32*7919).wrapping_add(1),&d);
        let mixing=Sampler::new(n,config.seed.wrapping_add(offset as u32*7919).wrapping_add(2),&d);
        let learners=Tensor::<1,Int>::from_data(TensorData::new((0..n).map(|a|i32::from(a%2==(offset+a/2)%2)).collect::<Vec<_>>(),[n]),&d).greater_elem(0).unsqueeze_dim::<2>(1);
        let mut h=Tensor::<2>::zeros([n,256],&d);let mut c=h.clone();
        let mut experience=Experience::new_ggd(n,&d);let mut previous=observe::initial_previous(n,&d);let mask=primitive(Tensor::<1>::zeros([n],&d));let mut stored=0usize;
        let limit=((w.engine.pre_game_seconds+config.seconds)*7.5).ceil()as usize+2;
        for step in 0..limit {
            if step%HORIZON==0&&step>0 {experience.signals=primitive(Tensor::<3>::zeros([n,HORIZON,SIGNALS],&d));}
            let obs=w.observe(previous.clone());let mut extended=observer.enrich(&w,&obs);extended.packed=extended.packed.cast(DType::F16);
            let snapshot=dagger.as_ref().map(|_|teacher.command_snapshot());
            let labels=controller.actions(&teacher,&w,&obs);
            let (actions,extra)=if let (Some(policy),Some(cfg))=(policy.as_ref(),dagger.as_ref()){
                if step%HORIZON==0 {h=Tensor::zeros([n,256],&d);c=h.clone();}
                let keep:Tensor<2>=1.0f32-obs.reset.clone().unsqueeze_dim::<2>(1);h=h*keep.clone();c=c*keep;
                let batch=extended.decode();let encoded=policy.encode(&batch);(h,c)=policy.memory.step(encoded.fused,h,c);
                let predicted=policy.heads(&batch,h.clone(),encoded.keys,None,&sampler).actions;
                let choose=if cfg.beta==1. {Tensor::<2>::ones([n,1],&d).greater_elem(0.)}
                    else if cfg.beta==0. {Tensor::<2>::zeros([n,1],&d).greater_elem(0.)}
                    else {let logits=Tensor::cat(vec![Tensor::full([n,1],(1.-cfg.beta).ln(),&d),Tensor::full([n,1],cfg.beta.ln(),&d)],1);mixing.draw(logits).greater_elem(0).unsqueeze_dim::<2>(1)};
                let use_student=learners.clone().bool_and(choose.clone().bool_not());
                let disagreement=predicted.clone().not_equal(labels.clone()).int().sum_dim(1).greater_elem(0).float();
                let actions=labels.clone().mask_where(use_student.clone(),predicted);
                teacher.reconcile_commands(&w,&obs,snapshot.unwrap(),labels.clone(),actions.clone());
                (actions,Tensor::cat(vec![use_student.bool_not().float(),disagreement,learners.clone().float(),Tensor::zeros([n,1],&d)],1))
            }else{(labels.clone(),Tensor::zeros([n,4],&d))};
            metrics.commands(&w,&teacher,actions.clone(),&obs.refs);
            valid::launch(&w.state.client,CubeCount::Static(n.div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&w.state),argument(&teacher.audit),argument(&mask),n);
            let eligibility=tensor::<1>(mask.clone())+0.0f32;previous=obs.previous(actions.clone(),&w.bases,&w.state);
            let signal=Tensor::cat(vec![labels.float(),extra,obs.reset.clone().unsqueeze_dim::<2>(1),eligibility.unsqueeze_dim::<2>(1),Tensor::full([n,1],GGD_SPEC_VERSION as f32,&d)],1);
            experience.store(step%HORIZON,&extended,signal);stored+=1;
            let _=w.advance_teacher(actions,&obs.refs,&rng,&teacher,true);teacher.record(&w,config.seconds);metrics.sample(&w,&teacher);
            if stored==HORIZON || step+1==limit {
                if dagger.is_some(){
                    let values=experience.signals().into_data().try_to_vec::<f32>().unwrap();
                    for v in values.chunks_exact(SIGNALS){if v[11]>0.5 && v[8]>0.5 {learner_frames+=1;disagreements+=usize::from(v[7]>0.5);teacher_executed+=usize::from(v[6]>0.5);}}
                }
                writer.store(&experience,offset,block)?;block+=1;stored=0;
                let values=teacher.values();let done=values.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).filter(|v|v[0]>0.).count();
                let duration=values[1];eprintln!("GGD collection block={block} cohort={offset} game_time={duration:.2} finished={done}/{games} wall={:.2}s",started.elapsed().as_secs_f64());
                crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"stage":"collecting","pid":std::process::id(),"blocks":block,"cohort_offset":offset,"finished_in_cohort":done,"game_time":duration,"elapsed_seconds":started.elapsed().as_secs_f64()}))?;
                if done==games {
                    for mut game in metrics.games(&values,w.engine.pre_game_seconds,offset,None){completed+=usize::from(game["native_terminal"]==true);game["split"]=serde_json::json!(split(game["game"].as_u64().unwrap()as usize));game_metrics.push(game);}
                    crate::artifacts::atomic_json(&root.join("teacher-summary.json"),&crate::ggd_metrics::summary(&game_metrics))?;
                    crate::artifacts::atomic_json(&root.join("game-metrics.json"),&game_metrics)?;break;
                }
                if config.max_blocks.is_some_and(|limit|block>=limit)||root.join("stop.json").exists(){partial=true;break 'cohorts;}
            }
        }
        // The final block can be shorter than 128 decisions. Storage starts at
        // zero and all trailing frames must remain invalid, never stale data.
        if stored>0 {anyhow::bail!("internal partial block was not finalized");}
    }
    let files=writer.finish()?;let bytes=files.iter().map(|e|e.bytes).sum::<u64>();let frames=files.iter().map(|e|e.valid_frames).sum::<usize>();let attacks=files.iter().map(|e|e.attacks).sum::<usize>();
    let elapsed=started.elapsed().as_secs_f64();
    let dataset=Dataset{family:"GGD".into(),observation_profile:"ggd-v2".into(),teacher_sha256:teacher_sha,source_sha256:crate::runtime::bytes_sha256(concat!(include_str!("combat_observe.rs"),include_str!("../src/contract.rs")).as_bytes()),config:config.clone(),partial,completed_matches:completed,elapsed_seconds:elapsed,files,game_metrics};
    crate::artifacts::atomic_json(&root.join("dataset.json"),&dataset)?;
    crate::artifacts::atomic_json(&root.join("teacher-summary.json"),&crate::ggd_metrics::summary(&dataset.game_metrics))?;
    let report=serde_json::json!({"family":"GGD","dataset":root,"games_requested":config.games,"games_recorded":dataset.game_metrics.len(),"native_completed":completed,"partial":partial,"blocks":block,"valid_decisions":frames,"learner_decisions":learner_frames,"disagreements":disagreements,"teacher_executed":teacher_executed,"dagger":dagger,"format_version":3,"signal_fields_extra":{"6":"teacher_executed","7":"learner_label_disagreement_all_six_fields","8":"learner_actor"},"attack_decisions":attacks,"compressed_bytes":bytes,"wall_seconds":elapsed,"decisions_per_second":frames as f64/elapsed,"train_games":(0..config.games).filter(|g|split(*g)=="train").count(),"validation_games":(0..config.games).filter(|g|split(*g)=="validation").count(),"test_games":(0..config.games).filter(|g|split(*g)=="test").count(),"cpu_boundary":"compact FP16 snapshots every 128 decisions; bounded writer, no dropped samples"});
    crate::artifacts::atomic_json(&root.join("report.json"),&report)?;Ok(report)
}

/// Launch the project's PyTorch policy on ROCm/CUDA. The simulator and
/// observation/label collection stay in the native CubeCL pipeline.
pub fn train(config:&Path,root:&Path,max_updates:Option<usize>)->Result<serde_json::Value>{
    let workspace=Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let absolute=|p:&Path|if p.is_absolute(){p.to_path_buf()}else{std::env::current_dir().unwrap().join(p)};
    let linux=|p:&Path|{let s=p.to_string_lossy().replace('\\',"/");if s.as_bytes().get(1)==Some(&b':'){format!("/mnt/{}/{}",s[..1].to_lowercase(),&s[3..])}else{s}};
    let mut command=if cfg!(windows){
        let mut c=std::process::Command::new("wsl");c.args(["-d","Ubuntu","--","bash"]).arg(linux(&workspace.join("scripts/wsl_python.sh")));c
    }else{std::process::Command::new("python3")};
    command.arg(linux(&workspace.join("gpu/scripts/train_ggd.py"))).arg(linux(&absolute(config))).arg(linux(&absolute(root)));
    if let Some(n)=max_updates{ensure!(n>0,"positive smoke update limit required");command.args(["--max-updates",&n.to_string()]);}
    let status=command.status()?;ensure!(status.success(),"GGD GPU IL failed; see training log");
    Ok(serde_json::from_slice(&std::fs::read(root.join("result.json"))?)?)
}

/// Closed-loop GGD versus the fixed teacher, balanced by learner side. This
/// reports the learner's own LH/deny, so the opponent cannot hide weak play.
pub fn evaluate(engine:&Path,weights:&Path,root:&Path,matches:usize,seconds:f32,mode:&str,seed:u32,trace:bool,record_teacher:bool,argmax:bool,zero_previous:bool,teacher_version:crate::teacher_v10::Version)->Result<serde_json::Value>{
    ensure!(matches>0&&(matches%2==0||record_teacher)&&seconds>0.&&seconds<=600.,"an even positive match count (or teacher recording) and 0 < seconds <= 600 required");
    ensure!(mode=="teacher"||mode=="mirror","mode must be teacher or mirror");
    ensure!(!record_teacher||mode=="teacher","teacher recording requires --mode teacher");
    std::fs::create_dir_all(root)?;ensure!(!root.join("config.json").exists(),"choose a fresh evaluation directory");
    let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(weights.join("manifest.json"))?)?;
    let zero_previous=zero_previous||manifest["zero_previous"].as_bool().unwrap_or(false);
    crate::artifacts::atomic_json(&root.join("config.json"),&serde_json::json!({"family":"GGD","weights":weights,"engine":engine,"matches":matches,"seconds":seconds,"mode":mode,"seed":seed,"argmax":argmax,"zero_previous":zero_previous,"trace":trace,"record_teacher":record_teacher,"teacher_version":teacher_version,"teacher_sha256":teacher_version.sha256()}))?;
    let d=device();let w=World::load(engine,matches,&d)?;let policy=crate::bridge::import(weights,&d)?.valid();ensure!(policy.is_ggd(),"GGD weights required");
    let teacher=crate::scripted_teacher::Teacher::new(&w);let metrics=crate::ggd_metrics::Metrics::new(&w);metrics.sample(&w,&teacher);let observer=crate::combat_observe::CombatObserver::new(&w);let rng=Sampler::new(matches,seed,&d);let sampler=Sampler::new(matches*2,seed.wrapping_add(1),&d).with_argmax(argmax);
    if teacher_version==crate::teacher_v10::Version::V10 {crate::teacher_v10::check_pin()?;}
    let controller=crate::teacher_v10::Controller::new(&w,teacher_version,teacher_version,false);
    let rows=Tensor::<1,Int>::from_data(TensorData::new((0..matches*2).map(|a|i32::from(a%2==(a/2)%2)).collect::<Vec<_>>(),[matches*2]),&d).greater_elem(0).unsqueeze_dim::<2>(1);
    let mut h=Tensor::<2>::zeros([matches*2,256],&d);let mut c=h.clone();let mut previous=observe::initial_previous(matches*2,&d);let started=Instant::now();
    let mut diagnostic=if trace{Some(crate::ggd_diagnostic::Trace::new(&w,root)?)}else{None};
    let recording=root.join("observations");
    let writer=if record_teacher{std::fs::create_dir_all(&recording)?;Some(DemoWriter::new(&recording))}else{None};
    let mut experience=if record_teacher{Some(Experience::new_ggd(matches*2,&d))}else{None};
    let mask=primitive(Tensor::<1>::zeros([matches*2],&d));let mut steps=0;
    let limit=((w.engine.pre_game_seconds+seconds)*7.5).ceil()as usize+2;
    for step in 0..limit {
        if let Some(e)=experience.as_mut(){if step%HORIZON==0&&step>0{e.signals=primitive(Tensor::<3>::zeros([matches*2,HORIZON,SIGNALS],&d));}}
        if step%HORIZON==0 {h=Tensor::zeros([matches*2,256],&d);c=h.clone();}
        let obs=w.observe(previous.clone());let mut extended=observer.enrich(&w,&obs);extended.packed=extended.packed.cast(DType::F16);
        let keep:Tensor<2>=1.0f32-obs.reset.clone().unsqueeze_dim::<2>(1);h=h*keep.clone();c=c*keep;
        let mut b=extended.decode();if zero_previous{b.previous=Tensor::zeros([matches*2,PREVIOUS],&d);}let encoded=policy.encode(&b);(h,c)=policy.memory.step(encoded.fused,h,c);
        let predicted=policy.heads(&b,h.clone(),encoded.keys,None,&sampler).actions;let scripted=controller.actions(&teacher,&w,&obs);
        let actions=if record_teacher{scripted}else if mode=="mirror"{predicted}else{scripted.mask_where(rows.clone(),predicted)};
        if let Some(t)=diagnostic.as_ref(){t.record(&w,&teacher,step%HORIZON,actions.clone(),policy.action_head.forward(h.clone()),b.action_mask.clone(),obs.reset.clone());}
        if let Some(e)=experience.as_mut(){
            valid::launch(&w.state.client,CubeCount::Static((matches*2).div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&w.state),argument(&teacher.audit),argument(&mask),matches*2);
            let signal=Tensor::cat(vec![actions.clone().float(),Tensor::zeros([matches*2,4],&d),obs.reset.clone().unsqueeze_dim::<2>(1),(tensor::<1>(mask.clone())+0.0f32).unsqueeze_dim::<2>(1),Tensor::full([matches*2,1],GGD_SPEC_VERSION as f32,&d)],1);
            e.store(step%HORIZON,&extended,signal);
        }
        previous=obs.previous(actions.clone(),&w.bases,&w.state);metrics.commands(&w,&teacher,actions.clone(),&obs.refs);
        let _=w.advance_teacher(actions,&obs.refs,&rng,&teacher,true);teacher.record(&w,seconds);metrics.sample(&w,&teacher);
        steps=step+1;
        if steps%HORIZON==0||steps==limit {
            if let Some(t)=diagnostic.as_mut(){t.flush((steps-1)%HORIZON+1)?;}
            if let (Some(writer),Some(e))=(writer.as_ref(),experience.as_ref()){writer.store(e,0,step/HORIZON)?;}
            let values=teacher.values();let done=values.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).filter(|v|v[0]>0.).count();
            crate::artifacts::atomic_json(&root.join("status.json"),&serde_json::json!({"stage":"evaluating","pid":std::process::id(),"mode":mode,"decisions":step+1,"game_time":values[1],"finished":done,"matches":matches,"wall_seconds":started.elapsed().as_secs_f64()}))?;if done==matches||root.join("stop.json").exists(){break;}}
    }
    if let Some(writer)=writer{let files=writer.finish()?;crate::artifacts::atomic_json(&recording.join("recording.json"),&serde_json::json!({"files":files,"steps":steps,"seed":seed,"observer":"ggd-v2","dtype":"f16_le","controllers":teacher_version}))?;}
    let values=teacher.values();let games=metrics.games(&values,w.engine.pre_game_seconds,0,if record_teacher{None}else{Some(mode)});
    ensure!(games.iter().all(|g|g["simulation_error"]==0.),"GPU simulation error");
    let report=serde_json::json!({"family":"GGD","matches":matches,"mode":mode,"seed":seed,"complete":values.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).all(|v|v[0]>0.),"wall_seconds":started.elapsed().as_secs_f64(),"native_completed":values.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).filter(|v|v[2]>0.).count(),"estimate":"closed-loop simulator evaluation; not real Dota transfer","summary":crate::ggd_metrics::summary(&games),"per_match":games});
    crate::artifacts::atomic_json(&root.join("report.json"),&report)?;Ok(report)
}
