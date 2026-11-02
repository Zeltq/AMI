//! Bounded asynchronous export. Snapshots own immutable buffers before the writer sees them.
use anyhow::{Result,ensure};
use burn::{module::Module,optim::ModuleOptimizer};
use burn_cubecl::tensor::CubeTensor;
use cubecl::{prelude::*,server::Handle};
use serde::{Serialize,Deserialize};
use std::{path::Path,sync::{Arc,Mutex,mpsc::{sync_channel,SyncSender,TrySendError},atomic::{AtomicBool,AtomicU64,Ordering}},
    thread::{self,JoinHandle},io::Write};
use crate::{model::Policy,world::World,sampling::Sampler};

#[cube(launch)]
fn copy_words(source:&[u32],target:&mut [u32],words:usize){
    if ABSOLUTE_POS<words {target[ABSOLUTE_POS]=source[ABSOLUTE_POS];}
}
pub fn copy_handle(client:&Client,source:&Handle,bytes:usize)->Handle {
    assert_eq!(bytes%4,0);let target=client.empty(bytes);
    let groups=(bytes/4).div_ceil(256);let x=groups.min(65535).max(1);
    copy_words::launch(client,CubeCount::Static(x as u32,groups.div_ceil(x) as u32,1),CubeDim::new_1d(256),
        unsafe{BufferArg::from_raw_parts(source.clone(),1)},unsafe{BufferArg::from_raw_parts(target.clone(),1)},bytes/4);target
}
pub fn copy(t:&CubeTensor)->CubeTensor {
    let mut result=t.clone();let bytes=t.meta.shape().iter().product::<usize>()*t.dtype.size();
    result.handle=copy_handle(&t.client,&t.handle,bytes);result
}
fn checksum(bytes:&[u8])->String {
    let mut value=0xcbf29ce484222325u64;
    for b in bytes {value=(value^*b as u64).wrapping_mul(0x100000001b3);}
    format!("{value:016x}")
}
pub fn atomic_json(path:&Path,value:&impl Serialize)->Result<()> {
    let temporary=path.with_extension(format!("tmp-{}",std::process::id()));
    std::fs::write(&temporary,serde_json::to_vec_pretty(value)?)?;std::fs::rename(temporary,path)?;Ok(())
}

/// Pin mutable development tables and their two binary assets with each run.
/// The configuration retains the caller's path; resume uses this saved copy.
pub fn freeze_environment(source:&Path,root:&Path,resume:bool)->Result<std::path::PathBuf>{
    let directory=root.join("environment");let target=directory.join("engine.json");
    if resume {return Ok(if target.exists(){target}else{source.to_path_buf()});}
    ensure!(!root.join("latest.json").exists(),"existing run requires --resume");
    std::fs::create_dir_all(&directory)?;
    let mut engine:serde_json::Value=serde_json::from_slice(&std::fs::read(source)?)?;
    let calibration=engine["calibration"].as_str().ok_or_else(||anyhow::anyhow!("missing calibration directory"))?;
    let calibration=if !cfg!(windows) && calibration.as_bytes().get(1)==Some(&b':') {
        std::path::PathBuf::from(format!("/mnt/{}/{}",calibration[..1].to_lowercase(),calibration[3..].replace('\\',"/")))
    }else{calibration.into()};
    for (key,name) in [("geometry","geometry.f32"),("navigation","navigation.bin")] {
        let file=engine[key].as_str().ok_or_else(||anyhow::anyhow!("missing {key}"))?;
        std::fs::copy(calibration.join(file),directory.join(name))?;engine[key]=name.into();
    }
    for (key,name)in[("fow_blockers","fow_blockers.f32"),("movement_geometry","movement-geometry.f32"),("static_vision","static-vision.u32")] {
      if let Some(file)=engine[key].as_str().filter(|v|!v.is_empty()) {
        std::fs::copy(calibration.join(file),directory.join(name))?;engine[key]=name.into();
      }
    }
    engine["calibration"]=directory.canonicalize()?.to_string_lossy().to_string().into();
    atomic_json(&target,&engine)?;Ok(target)
}

pub struct Buffer {name:String,client:Client,handle:Handle,bytes:usize}
impl Buffer {
    fn tensor(name:&str,t:&CubeTensor)->Self{
        let copy=copy(t);Self{name:name.into(),bytes:copy.meta.shape().iter().product::<usize>()*copy.dtype.size(),client:copy.client,handle:copy.handle}
    }
    fn rng(name:&str,rng:&Sampler)->Self{
        Self{name:name.into(),bytes:rng.actors*4,client:rng.client.clone(),handle:copy_handle(&rng.client,&rng.state,rng.actors*4)}
    }
}
pub struct Checkpoint {pub policy:Policy,pub optimizer:ModuleOptimizer,pub metadata:serde_json::Value,pub buffers:Vec<Buffer>}
impl Checkpoint {
    pub fn capture(policy:&Policy,optimizer:&ModuleOptimizer,world:&World,previous:&CubeTensor,action_rng:&Sampler,world_rng:&Sampler,metadata:serde_json::Value)->Self{
        let buffers=vec![Buffer::tensor("state",&world.state),Buffer::tensor("shots",&world.shots),Buffer::tensor("shot_hits",&world.shot_hits),Buffer::tensor("meta",&world.meta),
            Buffer::tensor("trees",&world.tree_alive),Buffer::tensor("commands",&world.commands),Buffer::tensor("previous",previous),
            Buffer::rng("action_rng",action_rng),Buffer::rng("world_rng",world_rng)];
        Self {policy:policy.clone(),optimizer:optimizer.clone(),metadata,buffers}
    }
}
#[derive(Serialize,Deserialize)]
struct BufferEntry {bytes:usize,fnv1a64:String}

enum Job {Checkpoint(Checkpoint),Metrics(String),Trace{handle:Handle,client:Client,header:serde_json::Value,cycle:u64}}
#[derive(Default)]
pub struct Counters {pub pending:AtomicBool,pub completed:AtomicU64,pub skipped:AtomicU64,pub trace_pending:AtomicU64,
    pub trace_skipped:AtomicU64,pub bytes:AtomicU64,pub metric_skipped:AtomicU64}
pub struct Writer {sender:Option<SyncSender<Job>>,worker:Option<JoinHandle<Result<()>>>,pub counters:Arc<Counters>,error:Arc<Mutex<Option<String>>>}
fn retain_traces(root:&Path)->Result<()> {
    let mut paths:Vec<_>=std::fs::read_dir(root)?.filter_map(|v|v.ok()).filter(|v|v.file_type().is_ok_and(|t|t.is_file()))
        .filter(|v|{let name=v.file_name();let s=name.to_string_lossy();s.starts_with("matches-")&&s.ends_with(".sfduel.gz")}).map(|v|v.path()).collect();
    paths.sort();let remove=paths.len().saturating_sub(256);let allowed=root.canonicalize()?;
    for path in paths.into_iter().take(remove){ensure!(path.canonicalize()?.parent()==Some(allowed.as_path()),"trace retention escaped run");std::fs::remove_file(path)?;}Ok(())
}
impl Writer {
    pub fn new(root:&Path)->Result<Self>{
        std::fs::create_dir_all(root)?;let root=root.to_path_buf();
        let (sender,receiver)=sync_channel(8);let counters=Arc::new(Counters::default());let state=counters.clone();let error=Arc::new(Mutex::new(None));let errors=error.clone();
        let worker=thread::spawn(move||{
            let work=||->Result<()> {
                let mut metrics=std::fs::OpenOptions::new().create(true).append(true).open(root.join("metrics.jsonl"))?;
                while let Ok(job)=receiver.recv(){
                    match job {
                        Job::Metrics(line)=>{
                            writeln!(metrics,"{line}")?;metrics.flush()?;
                            if metrics.metadata()?.len()>=8*1024*1024 {
                                drop(metrics);
                                for i in (1..=3).rev(){let source=root.join(format!("metrics-{i}.jsonl"));let target=root.join(format!("metrics-{}.jsonl",i+1));
                                    if source.exists(){if target.exists(){std::fs::remove_file(&target)?;}std::fs::rename(source,target)?;}}
                                std::fs::rename(root.join("metrics.jsonl"),root.join("metrics-1.jsonl"))?;
                                metrics=std::fs::OpenOptions::new().create_new(true).write(true).open(root.join("metrics.jsonl"))?;
                            }
                        }
                        Job::Checkpoint(snapshot)=>{
                            let cycle=snapshot.metadata["cycle"].as_u64().unwrap();let name=format!("checkpoint-{cycle:08}");let temporary=root.join(format!("{name}.pending"));
                            std::fs::create_dir(&temporary)?;snapshot.policy.into_record().save(temporary.join("model.bpk"))?;
                            snapshot.optimizer.save(temporary.join("optimizer.bpk"))?;
                            let mut entries=std::collections::BTreeMap::new();
                            for buffer in snapshot.buffers {
                                let bytes=buffer.client.read_one(buffer.handle)?;ensure!(bytes.len()==buffer.bytes,"snapshot size mismatch");
                                std::fs::write(temporary.join(format!("{}.bin",buffer.name)),&*bytes)?;
                                state.bytes.fetch_add(bytes.len() as u64,Ordering::Relaxed);
                                entries.insert(buffer.name,BufferEntry{bytes:bytes.len(),fnv1a64:checksum(&bytes)});
                            }
                            atomic_json(&temporary.join("buffers.json"),&entries)?;atomic_json(&temporary.join("state.json"),&snapshot.metadata)?;
                            std::fs::rename(&temporary,root.join(&name))?;atomic_json(&root.join("latest.json"),&serde_json::json!({"checkpoint":name}))?;
                            let mut checkpoints:Vec<_>=std::fs::read_dir(&root)?.filter_map(|x|x.ok()).filter(|x|x.file_type().is_ok_and(|t|t.is_dir()) && x.file_name().to_string_lossy().starts_with("checkpoint-") && !x.file_name().to_string_lossy().ends_with(".pending")).map(|x|x.path()).collect();
                            checkpoints.sort();let remove=checkpoints.len().saturating_sub(3);
                            let allowed=root.canonicalize()?;
                            for old in checkpoints.into_iter().take(remove){
                                ensure!(old.canonicalize()?.starts_with(&allowed),"checkpoint retention escaped run directory");
                                std::fs::remove_dir_all(old)?;
                            }
                            state.completed.fetch_add(1,Ordering::Relaxed);state.pending.store(false,Ordering::Release);
                        }
                        Job::Trace{handle,client,header,cycle}=>{
                            let bytes=client.read_one(handle)?;let header=serde_json::to_vec(&header)?;
                            let path=root.join(format!("matches-{cycle:08}.sfduel.gz"));
                            let mut zip=flate2::write::GzEncoder::new(std::fs::File::create(path)?,flate2::Compression::fast());
                            zip.write_all(b"SFDUEL02")?;zip.write_all(&(header.len() as u32).to_le_bytes())?;zip.write_all(&header)?;zip.write_all(&bytes)?;zip.finish()?;
                            retain_traces(&root)?;
                            state.trace_pending.fetch_sub(1,Ordering::Release);
                        }
                    }
                }
                Ok(())
            };
            let result=work();if let Err(e)=&result{*errors.lock().unwrap()=Some(format!("{e:#}"));}result
        });
        Ok(Self{sender:Some(sender),worker:Some(worker),counters,error})
    }
    pub fn check(&self)->Result<()> {if let Some(e)=self.error.lock().unwrap().as_ref(){anyhow::bail!("artifact writer: {e}")}Ok(())}
    pub fn reserve_checkpoint(&self,force:bool)->Result<bool>{
        loop {self.check()?;if self.counters.pending.compare_exchange(false,true,Ordering::AcqRel,Ordering::Acquire).is_ok(){return Ok(true)}
            if !force {self.counters.skipped.fetch_add(1,Ordering::Relaxed);return Ok(false)}
            thread::sleep(std::time::Duration::from_millis(10));}
    }
    pub fn checkpoint(&self,snapshot:Checkpoint,force:bool)->Result<()> {
        if force {self.sender.as_ref().unwrap().send(Job::Checkpoint(snapshot))?;return Ok(())}
        match self.sender.as_ref().unwrap().try_send(Job::Checkpoint(snapshot)){
            Ok(())=>{},Err(TrySendError::Full(_))=>{self.counters.pending.store(false,Ordering::Release);self.counters.skipped.fetch_add(1,Ordering::Relaxed);},
            Err(e)=>return Err(anyhow::anyhow!("{e}")),}Ok(())
    }
    pub fn metrics(&self,row:&serde_json::Value)->Result<()> {
        match self.sender.as_ref().unwrap().try_send(Job::Metrics(serde_json::to_string(row)?)){
            Ok(())=>{},Err(TrySendError::Full(_))=>{self.counters.metric_skipped.fetch_add(1,Ordering::Relaxed);},Err(e)=>return Err(anyhow::anyhow!("{e}")),}Ok(())
    }
    pub fn trace(&self,t:&CubeTensor,header:serde_json::Value,cycle:u64)->Result<()> {
        if self.counters.trace_pending.load(Ordering::Acquire)>=2 {self.counters.trace_skipped.fetch_add(1,Ordering::Relaxed);return Ok(())}
        self.counters.trace_pending.fetch_add(1,Ordering::AcqRel);let snapshot=copy(t);
        match self.sender.as_ref().unwrap().try_send(Job::Trace{handle:snapshot.handle,client:snapshot.client,header,cycle}){
            Ok(())=>{},Err(TrySendError::Full(_))=>{self.counters.trace_pending.fetch_sub(1,Ordering::Release);self.counters.trace_skipped.fetch_add(1,Ordering::Relaxed);},
            Err(e)=>return Err(anyhow::anyhow!("{e}")),}Ok(())
    }
    pub fn status(&self)->serde_json::Value {
        let c=&self.counters;serde_json::json!({"checkpoint_pending":c.pending.load(Ordering::Acquire),"checkpoints_written":c.completed.load(Ordering::Relaxed),
            "checkpoint_requests_coalesced":c.skipped.load(Ordering::Relaxed),"trace_batches_pending":c.trace_pending.load(Ordering::Acquire),
            "trace_batches_skipped":c.trace_skipped.load(Ordering::Relaxed),"metrics_skipped":c.metric_skipped.load(Ordering::Relaxed),"snapshot_bytes_written":c.bytes.load(Ordering::Relaxed)})
    }
    pub fn finish(mut self)->Result<()> {drop(self.sender.take());self.worker.take().unwrap().join().map_err(|_|anyhow::anyhow!("artifact writer panicked"))??;Ok(())}
}
impl Drop for Writer {
    fn drop(&mut self){
        drop(self.sender.take());
        if let Some(worker)=self.worker.take(){match worker.join(){
            Ok(Ok(()))=>{},Ok(Err(e))=>eprintln!("artifact writer shutdown: {e:#}"),Err(_)=>eprintln!("artifact writer panicked during shutdown")
        }}
    }
}

pub fn restore(root:&Path,policy:Policy,optimizer:ModuleOptimizer,world:&mut World,previous:&mut CubeTensor,action_rng:&mut Sampler,world_rng:&mut Sampler)->Result<(Policy,ModuleOptimizer,serde_json::Value)>{
    let latest:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("latest.json"))?)?;
    let name=latest["checkpoint"].as_str().ok_or_else(||anyhow::anyhow!("invalid latest checkpoint"))?;
    ensure!(Path::new(name).components().count()==1,"invalid checkpoint path");let path=root.join(name);
    let metadata:serde_json::Value=serde_json::from_slice(&std::fs::read(path.join("state.json"))?)?;
    ensure!(metadata["matches"].as_u64()==Some(world.matches as u64),"resume match count differs");
    ensure!(metadata["game_build"].as_str()==Some(world.engine.build.as_str()),"Dota build differs from checkpoint");
    ensure!(metadata["state_schema"]==3,"checkpoint state schema differs; this development checkpoint requires its original simulator revision");
    ensure!(metadata["environment_sha256"].as_str()==Some(world.fingerprint.as_str()),"simulator calibration or navigation differs from checkpoint");
    let entries:std::collections::BTreeMap<String,BufferEntry>=serde_json::from_slice(&std::fs::read(path.join("buffers.json"))?)?;
    let read=|name:&str,client:&Client,expected:usize|->Result<Handle>{
        let entry=entries.get(name).ok_or_else(||anyhow::anyhow!("missing {name}"))?;let bytes=std::fs::read(path.join(format!("{name}.bin")))?;
        ensure!(bytes.len()==expected && bytes.len()==entry.bytes && checksum(&bytes)==entry.fnv1a64,"corrupt {name}");Ok(client.create_from_slice(&bytes))
    };
    for (name,t) in [("state",&mut world.state),("shots",&mut world.shots),("shot_hits",&mut world.shot_hits),("meta",&mut world.meta),("trees",&mut world.tree_alive),("commands",&mut world.commands),("previous",previous)]{
        t.handle=read(name,&t.client,t.meta.shape().iter().product::<usize>()*t.dtype.size())?;
    }
    action_rng.state=read("action_rng",&action_rng.client,action_rng.actors*4)?;world_rng.state=read("world_rng",&world_rng.client,world_rng.actors*4)?;
    let policy=policy.try_load_record(burn::store::ModuleRecord::load(path.join("model.bpk"))?)?;
    let optimizer=optimizer.load(path.join("optimizer.bpk"))?;Ok((policy,optimizer,metadata))
}

