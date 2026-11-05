use anyhow::{Result,ensure};
use burn::{module::Module,tensor::{Tensor,Device}};
use crate::{chio_model::Policy,chio_actions::Masks,bridge::Bundle,runtime::{device,primitive}};
use std::path::Path;

pub fn import(root:&Path,d:&Device)->Result<Policy>{
    let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
    ensure!(manifest["observation_profile"]=="chio-v1" && manifest["action_types"]==13 && manifest["directions"]==36,"not a CHiO policy contract");
    ensure!(manifest["decision_ticks"]==6 && manifest["offset_ticks"]==6 && manifest["recurrent_reset"]==128,"CHiO clock/memory mismatch");
    crate::bridge::import_module(Policy::new(d),root)
}

pub fn export(model:&Policy,root:&Path)->Result<()> {
    crate::bridge::export_module(model,root)?;
    let path=root.join("manifest.json");let mut value:serde_json::Value=serde_json::from_slice(&std::fs::read(&path)?)?;
    for tensor in value["tensors"].as_object_mut().unwrap().values_mut(){tensor["sha256"]=serde_json::json!(crate::runtime::file_sha256(&root.join(tensor["file"].as_str().unwrap()))?);}
    for (key,v) in [("family",serde_json::json!("CHiO")),("model_name",serde_json::json!("CHiO")),("observation_profile",serde_json::json!("chio-v1")),
        ("action_types",serde_json::json!(13)),("directions",serde_json::json!(36)),("decision_ticks",serde_json::json!(6)),("offset_ticks",serde_json::json!(6)),
        ("recurrent_reset",serde_json::json!(128)),("timing_honest",serde_json::json!(true)),("timing_oracle",serde_json::json!(false))] {value[key]=v;}
    crate::artifacts::atomic_json(&path,&value)
}

pub fn parity(weights:&Path,fixture:&Path,output:&Path)->Result<serde_json::Value>{
    let d=device();let model=import(weights,&d)?.valid();let fixture=Bundle::open(fixture)?;let b=fixture.observation(&d)?;
    let n=b.heroes.dims()[0];ensure!(n==128,"CHiO parity requires 128 steps");
    let masks=Masks{packed:primitive(Tensor::<2>::zeros([n,1],&d)),types:b.action_mask.clone(),
        enemy:fixture.bools("enemy_target_mask",&d)?,ally:fixture.bools("ally_target_mask",&d)?};
    let encoded=model.encode(&b);let resets=fixture.read::<1>("resets",&d)?.into_data().try_to_vec::<f32>()?;
    let mut h=Tensor::<2>::zeros([1,256],&d);let mut c=h.clone();let mut hs=vec![];
    for i in 0..n {if i%128==0||resets[i]>0.5 {h=Tensor::zeros([1,256],&d);c=h.clone();}
        (h,c)=model.memory.step(encoded.fused.clone().slice([i..i+1]),h,c);hs.push(h.clone());}
    let h=Tensor::cat(hs,0);let sampler=crate::sampling::Sampler::new(n,7,&d);
    let heads=model.heads(&masks,h.clone(),encoded.keys,Some(fixture.ints("labels",&d)?),&sampler);
    let mut logits=serde_json::Map::new();let mut passed=true;
    for (name,actual) in ["action","direction","enemy_target","ally_target"].into_iter().zip(heads.logits){
        let delta=(actual-fixture.read::<2>(&format!("expected_{name}_logits"),&d)?).abs().max().into_scalar::<f32>();
        passed&=delta<=1e-4;logits.insert(name.into(),serde_json::json!({"max_absolute_error":delta,"tolerance":1e-4,"steps":n}));
    }
    let hidden=(h-fixture.read::<2>("expected_h",&d)?).abs().max().into_scalar::<f32>();
    let value=(heads.value-fixture.read::<1>("expected_value",&d)?).abs().max().into_scalar::<f32>();
    let logp=(heads.logp-fixture.read::<1>("expected_logp",&d)?).abs().max().into_scalar::<f32>();
    passed&=hidden<=1e-4&&value<=1e-4&&logp<=1e-4;
    let roundtrip=output.parent().unwrap().join("roundtrip-weights");export(&model,&roundtrip)?;
    let original=Bundle::open(weights)?;let copied=Bundle::open(&roundtrip)?;let mut equal=0;
    for (name,entry) in &original.tensors {let other=&copied.tensors[name];
        if entry.shape==other.shape && std::fs::read(weights.join(&entry.file))?==std::fs::read(roundtrip.join(&other.file))? {equal+=1;}}
    passed&=equal==original.tensors.len();
    let result=serde_json::json!({"passed":passed,"steps":n,"logits":logits,"hidden_max_error":hidden,"value_max_error":value,"logp_max_error":logp,
        "roundtrip":{"equal":equal,"total":original.tensors.len()},"observation_storage":"FP16 to FP32","recurrent_reset":128,"fixture_life_resets":resets.iter().filter(|v|**v>0.5).count()});
    crate::artifacts::atomic_json(output,&result)?;ensure!(passed,"CHiO parity failed: {result}");Ok(result)
}
