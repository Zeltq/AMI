//! One recorded minibatch: export loss, gradients, and a fresh Adam step.
use anyhow::{Result,ensure};
use burn::{module::{Module,ModuleMapper,Param},optim::GradientsParams,tensor::{Tensor,TensorData,f16}};
use std::path::Path;
use crate::{learner::{Config,Learner},runtime::{device,primitive},observe::{Observed,WIDTH},sampling::Sampler,experience::HORIZON};
fn read(path:&Path)->Result<Vec<f32>>{Ok(std::fs::read(path)?.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect())}
pub fn run(audit:&Path,root:&Path)->Result<serde_json::Value>{
    std::fs::create_dir_all(root)?;ensure!(!root.join("result.json").exists(),"fresh output required");
    let cfg:Config=serde_json::from_slice(&std::fs::read(audit.join("config.json"))?)?;
    let d=device();let mut learner=Learner::new(cfg.clone());learner.policy=crate::selfplay::load(Path::new(cfg.initial_checkpoint.as_ref().unwrap()),&d)?.train();
    let folder=audit.join("cycle-01");let n=4;let length=n*HORIZON;
    let raw=std::fs::read(folder.join("inputs.f16"))?;ensure!(raw.len()==length*WIDTH*2,"four recorded actors required");
    let inputs=raw.chunks_exact(2).map(|x|f16::from_bits(u16::from_le_bytes(x.try_into().unwrap()))).collect::<Vec<_>>();
    let packed=Tensor::<2>::from_data(TensorData::new(inputs,[length,WIDTH]),&d);
    let signals=read(&folder.join("signals.f32"))?;let signals=Tensor::<3>::from_data(TensorData::new(signals[..length*13].to_vec(),[n,HORIZON,13]),&d);
    let observed=Observed{packed,refs:primitive(Tensor::<1>::zeros([1],&d)),reset:Tensor::zeros([length],&d)};let b=observed.decode_bucket(16,32,16);
    let encoded=learner.policy.encode(&b);let projected=learner.policy.memory.input.forward(encoded.fused).reshape([n,HORIZON,1024]);
    let mut h=Tensor::<2>::zeros([n,256],&d);let mut c=h.clone();let mut hs=vec![];
    for t in 0..HORIZON{
        let keep:Tensor<2>=1.0f32-signals.clone().slice([0..n,t..t+1,10..11]).reshape([n,1]);h=h*keep.clone();c=c*keep;
        (h,c)=learner.policy.memory.projected(projected.clone().slice([0..n,t..t+1,0..1024]).reshape([n,1024]),h,c);hs.push(h.clone());
    }
    let h=Tensor::stack::<3>(hs,1).reshape([length,256]);let labels=signals.clone().slice([0..n,0..HORIZON,0..6]).reshape([length,6]).int();
    let heads=learner.policy.heads(&b,h,encoded.keys,Some(labels),&Sampler::new(length,1,&d));
    let old=signals.clone().slice([0..n,0..HORIZON,6..7]).reshape([length]);let mask=signals.slice([0..n,0..HORIZON,11..12]).reshape([length]);let count=mask.clone().sum();
    let adv=read(&folder.join("normalized-advantages.f32"))?;let adv=Tensor::<1>::from_data(TensorData::new(adv[..length].to_vec(),[length]),&d);
    let ret=read(&folder.join("returns.f32"))?;let ret=Tensor::<1>::from_data(TensorData::new(ret[..length].to_vec(),[length]),&d);
    let ratio=(heads.logp-old).exp();let actor=(-(ratio.clone()*adv.clone()).min_pair(ratio.clamp(1.-cfg.clip,1.+cfg.clip)*adv)*mask.clone()).sum()/count.clone();
    let critic=((heads.value-ret).square()*cfg.value_coefficient*mask.clone()).sum()/count.clone();let entropy=-(heads.entropy*cfg.entropy_coefficient*mask).sum()/count;
    let scalars=serde_json::json!({"actor":actor.clone().without_autodiff().into_scalar::<f32>(),"critic":critic.clone().without_autodiff().into_scalar::<f32>(),"entropy":entropy.clone().without_autodiff().into_scalar::<f32>()});
    let mut grads=GradientsParams::from_grads((actor+critic+entropy).backward(),&learner.policy);
    struct ExportGrad<'a>{grads:&'a GradientsParams}
    impl ModuleMapper for ExportGrad<'_>{fn map_float<const D:usize>(&mut self,p:Param<Tensor<D>>)->Param<Tensor<D>>{let g=self.grads.get::<D>(p.id).unwrap_or_else(||Tensor::zeros(p.val().dims(),&p.val().device()));p.map(|_|g)}}
    let gradients=learner.policy.clone().map(&mut ExportGrad{grads:&grads});crate::bridge::export(&gradients,&root.join("gradients"))?;
    let norm=crate::gradients::clip(&learner.policy,&mut grads,cfg.gradient_norm).into_scalar::<f32>();
    learner.policy=learner.optimizer.step(cfg.learning_rate,learner.policy,grads);
    crate::bridge::export(&learner.policy.valid(),&root.join("after-step"))?;
    let report=serde_json::json!({"complete":true,"actors":n,"learner_frames":384,"fresh_adam":true,"steps":1,"losses":scalars,"gradient_norm_before_clip":norm,"source_signals_sha256":crate::runtime::file_sha256(&folder.join("signals.f32"))?,"source_inputs_sha256":crate::runtime::file_sha256(&folder.join("inputs.f16"))?,"scope":"One identical recorded first-cycle four-actor minibatch, global saved advantages/returns and learn masks. Diagnostic only; no episode collection or production change."});
    crate::artifacts::atomic_json(&root.join("result.json"),&report)?;Ok(report)
}
