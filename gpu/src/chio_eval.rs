use anyhow::{Result,ensure};
use burn::{module::Module,tensor::{Tensor,TensorData,Int}};
use serde::{Serialize,Deserialize};
use std::{path::Path,time::Instant,io::Write};
use crate::{world::World,chio_observe as co,teacher_v10::Version,runtime::{device,tensor},sampling::Sampler,experience::HORIZON};

#[derive(Clone,Serialize,Deserialize)]
#[serde(default,deny_unknown_fields)]
pub struct Config {
    pub engine:String,pub matches:usize,pub seconds:f32,pub seed:u32,pub left:Version,pub right:Version,
    pub swap:bool,pub original_v9:bool,pub baseline_005:bool,pub weights:Option<String>,pub opponent_weights:Option<String>,
    pub mirror:bool,pub sampling:bool,pub zero_sum:bool,pub trace:bool,pub direction_diagnostics:bool,
}
impl Default for Config {fn default()->Self{Self{engine:String::new(),matches:200,seconds:600.,seed:20261103,left:Version::V10,right:Version::V10,
    swap:false,original_v9:false,baseline_005:false,weights:None,opponent_weights:None,mirror:false,sampling:true,zero_sum:true,trace:false,direction_diagnostics:false}}}

pub fn run(cfg:Config,root:&Path)->Result<serde_json::Value>{run_impl(cfg,root,false)}
/// Explicit ablation only; production collect/train/evaluate always share purchases.
pub fn diagnose_teacher_purchases(cfg:Config,root:&Path)->Result<serde_json::Value>{
    ensure!(!cfg.baseline_005 && !cfg.original_v9 && cfg.weights.is_none() && cfg.opponent_weights.is_none() && cfg.left==Version::V10 && cfg.right==Version::V10,"purchase ablation requires two v10 teachers");
    run_impl(cfg,root,true)
}
fn run_impl(cfg:Config,root:&Path,diagnostic_teacher_purchases:bool)->Result<serde_json::Value>{
    ensure!(cfg.matches>0 && cfg.seconds>0. && cfg.seconds<=600.,"invalid CHiO evaluation dimensions");
    ensure!(!cfg.swap||cfg.matches%2==0,"swap requires an even number of matches");crate::teacher_v10::check_pin()?;
    if cfg.baseline_005 {
        ensure!(cfg.weights.is_none()&&!cfg.original_v9,"baseline uses original teachers only");
        return crate::teacher_eval::run_with_chio_metrics(Path::new(&cfg.engine),root,cfg.matches,cfg.seconds,cfg.seed,cfg.left,cfg.right,cfg.swap,cfg.trace);
    }
    std::fs::create_dir_all(root)?;ensure!(!root.join("config.json").exists(),"fresh CHiO evaluation directory required");
    crate::artifacts::atomic_json(&root.join("config.json"),&cfg)?;
    let frozen=crate::artifacts::freeze_environment(Path::new(&cfg.engine),root,false)?;let d=device();let w=World::load(&frozen,cfg.matches,&d)?;
    let n=cfg.matches*2;let limit=((w.engine.pre_game_seconds+cfg.seconds)*w.decision_hz()).ceil()as usize+2;
    let mut runtime=crate::chio_runtime::Runtime::new(w,cfg.left,cfg.right,cfg.swap,cfg.original_v9,cfg.zero_sum)?;runtime.diagnostic_teacher_purchases=diagnostic_teacher_purchases;
    let model=cfg.weights.as_ref().map(|p|crate::chio_bridge::import(Path::new(p),&d).map(|p|p.valid())).transpose()?;
    let opponent=cfg.opponent_weights.as_ref().map(|p|crate::chio_bridge::import(Path::new(p),&d).map(|p|p.valid())).transpose()?;
    ensure!(opponent.is_none()||model.is_some(),"opponent weights require learner weights");
    let rng=Sampler::new(cfg.matches,cfg.seed,&d);let sampler=Sampler::new(n,cfg.seed.wrapping_add(1),&d).with_argmax(!cfg.sampling);
    let opponent_sampler=Sampler::new(n,cfg.seed.wrapping_add(2),&d).with_argmax(!cfg.sampling);
    let learner=Tensor::<1,Int>::from_data(TensorData::new((0..n).map(|a|i32::from(cfg.mirror||a%2==(a/2)%2)).collect::<Vec<_>>(),[n]),&d).greater_elem(0).unsqueeze_dim::<2>(1);
    let mut h=Tensor::<2>::zeros([n,256],&d);let mut c=h.clone();let mut oh=h.clone();let mut oc=h.clone();
    let mut diagnostic=if cfg.direction_diagnostics {let mut file=std::fs::File::create(root.join("direction-diagnostics.csv"))?;
        writeln!(file,"match,side,elapsed,game_seconds,x,y,after_x,after_y,action,direction,teacher_action,teacher_direction,order,dest_x,dest_y,queued_action,queued_x,queued_y,move_speed")?;Some(file)}else{None};
    let trace=cfg.trace.then(||crate::trace::Trace::new(&runtime.world));let started=Instant::now();let mut steps=0;
    for step in 0..limit {
        let frame=runtime.observe();let mut chosen=frame.labels.clone();let mut value=Tensor::<1>::zeros([n],&d);let mut logp=value.clone();
        if let Some(model)=&model {
            let keep:Tensor<2>=1.0f32-frame.observation.reset.clone().unsqueeze_dim::<2>(1);h=h*keep.clone();c=c*keep.clone();oh=oh*keep.clone();oc=oc*keep;
            let (b,masks)=co::decode(&frame.observation,crate::observe::CREEPS,crate::observe::TREES,crate::observe::PROJECTILES);
            let encoded=model.encode(&b);(h,c)=model.memory.step(encoded.fused,h,c);let heads=model.heads(&masks,h.clone(),encoded.keys,None,&sampler);
            if let Some(opponent)=&opponent {let encoded=opponent.encode(&b);(oh,oc)=opponent.memory.step(encoded.fused,oh,oc);chosen=opponent.heads(&masks,oh.clone(),encoded.keys,None,&opponent_sampler).actions;}
            chosen=chosen.mask_where(learner.clone(),heads.actions);value=heads.value;logp=heads.logp;
        }
        let diagnosis=if diagnostic.is_some(){use crate::{world::*,contract::*};
            Some((tensor::<1>(runtime.world.state.clone()).reshape([cfg.matches,UNITS,STRIDE]).slice([0..cfg.matches,0..2,0..STRIDE]).into_data().try_to_vec::<f32>()?,
                tensor::<1>(runtime.world.meta.clone()).into_data().try_to_vec::<f32>()?,chosen.clone().into_data().try_to_vec::<i32>()?,frame.labels.clone().into_data().try_to_vec::<i32>()?))}else{None};
        let transition=runtime.advance(&frame,chosen.clone(),&rng,cfg.seconds);steps=step+1;
        if let (Some(file),Some((before,meta,actions,labels)))=(&mut diagnostic,diagnosis){use crate::{world::*,contract::*};
            let after=tensor::<1>(runtime.world.state.clone()).reshape([cfg.matches,UNITS,STRIDE]).slice([0..cfg.matches,0..2,0..STRIDE]).into_data().try_to_vec::<f32>()?;
            let queue=tensor::<1>(runtime.world.commands.clone()).into_data().try_to_vec::<f32>()?;
            for m in 0..cfg.matches.min(3) {let side=m%2;let a=m*2+side;let p=a*STRIDE;let slot=meta[m*META+6]as usize%4;let q=(a*4+slot)*COMMAND_FIELDS;let game=meta[m*META]/30.;
                writeln!(file,"{m},{side},{},{game},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",game+runtime.world.engine.pre_game_seconds,before[p+UNIT_X],before[p+UNIT_Y],after[p+UNIT_X],after[p+UNIT_Y],actions[a*4],actions[a*4+1],labels[a*4],labels[a*4+1],before[p+ORDER],before[p+DEST_X],before[p+DEST_Y],queue[q+1],queue[q+5],queue[q+6],before[p+UNIT_MOVE_SPEED])?;
            }
        }
        if let Some(trace)=&trace {
            let signals=Tensor::cat(vec![chosen.float(),Tensor::zeros([n,2],&d),logp.unsqueeze_dim::<2>(1),value.unsqueeze_dim::<2>(1),
                transition.reward.total.unsqueeze_dim::<2>(1),transition.done.unsqueeze_dim::<2>(1),frame.observation.reset.unsqueeze_dim::<2>(1),frame.valid.unsqueeze_dim::<2>(1),Tensor::zeros([n,1],&d)],1);
            trace.record(&runtime.world,step%128,signals);
        }
        if steps%128==0||steps==limit {
            if let Some(trace)=&trace {
                let count=(steps-1)%128+1;let values=tensor::<3>(trace.buffer.clone()).slice([0..count,0..trace.matches,0..trace.width]).into_data().try_to_vec::<f32>()?;
                let mut header=trace.header(&runtime.world,steps.div_ceil(128)as u64);header["frames"]=count.into();header["observation_profile"]="chio-v1".into();header["action_layout"]=serde_json::json!(["type","direction","enemy_target","ally_target","unused","unused"]);
                let mut out=flate2::write::GzEncoder::new(std::fs::File::create(root.join(format!("matches-{:08}.sfduel.gz",steps.div_ceil(HORIZON))))?,flate2::Compression::fast());
                let bytes=serde_json::to_vec(&header)?;out.write_all(b"SFDUEL02")?;out.write_all(&(bytes.len()as u32).to_le_bytes())?;out.write_all(&bytes)?;for v in values{out.write_all(&v.to_le_bytes())?;}out.finish()?;
            }
            let audit=runtime.teacher.values();let done=audit.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).filter(|v|v[0]>0.).count();
            ensure!(audit.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).all(|v|v[39]==0.),"CHiO simulation/label error");
            let status=serde_json::json!({"stage":"evaluating-chio","pid":std::process::id(),"steps":steps,"finished":done,"matches":cfg.matches,"game_seconds":audit[1],"wall_seconds":started.elapsed().as_secs_f64()});
            crate::artifacts::atomic_json(&root.join("status.json"),&status)?;eprintln!("{status}");if done==cfg.matches||root.join("stop.json").exists(){break;}
        }
    }
    let mut games=runtime.games(0)?;
    for (m,game) in games.iter_mut().enumerate(){for side in 0..2 {
        let selected=if cfg.swap && m%2==1 {1-side}else{side};let version=if selected==0{cfg.left}else{cfg.right};
        let mut name=format!("teacher-{}-chio",if version==Version::V9{"v9"}else{"v10"});
        if cfg.original_v9 && selected==1 {name="teacher-v9-original".into();}
        if model.is_some()&&(cfg.mirror||side==m%2) {name="CHiO".into();}else if opponent.is_some(){name="CHiO-opponent".into();}
        game["sides"][side]["controller"]=name.into();
    }}
    let complete=runtime.teacher.values().chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).all(|v|v[0]>0.);
    let result=serde_json::json!({"family":"CHiO","diagnostic_teacher_purchases":diagnostic_teacher_purchases,"matches":cfg.matches,"seed":cfg.seed,"complete":complete,"steps":steps,"wall_seconds":started.elapsed().as_secs_f64(),
        "scripted_base_to_mid":runtime.world.engine.chio_script_navigation,"decision_ticks":6,"offset_ticks":6,"reward_mode":"lastorder-fb-v1","zero_sum":cfg.zero_sum,"per_match":games});
    crate::artifacts::atomic_json(&root.join("report.json"),&result)?;Ok(result)
}
