use std::{collections::BTreeMap,path::{Path,PathBuf}};
use anyhow::{Result,Context,ensure};
use burn::{module::{Module,ModuleMapper,Param},tensor::{Device,Tensor,TensorData,Bool,Int}};
use serde::{Deserialize,Serialize};
use crate::model::{Observation,Policy};

#[derive(Deserialize,Serialize)]
pub struct Entry {pub shape:Vec<usize>,pub file:String,#[serde(default)]pub sha256:String}
#[derive(Deserialize,Serialize)]
pub struct Bundle {pub tensors:BTreeMap<String,Entry>,#[serde(skip)]pub root:PathBuf}
impl Bundle {
    pub fn open(root:&Path)->Result<Self>{
        let mut result:Self=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        result.root=root.to_path_buf();Ok(result)
    }
    pub fn read<const D:usize>(&self,name:&str,d:&Device)->Result<Tensor<D>>{
        let entry=self.tensors.get(name).with_context(||format!("missing tensor {name}"))?;
        ensure!(entry.shape.len()==D,"rank mismatch for {name}");
        let bytes=std::fs::read(self.root.join(&entry.file))?;
        ensure!(bytes.len()==4*entry.shape.iter().product::<usize>(),"invalid size for {name}");
        if !entry.sha256.is_empty(){ensure!(crate::runtime::file_sha256(&self.root.join(&entry.file))?==entry.sha256,"corrupt imported tensor {name}");}
        let values:Vec<f32>=bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
        Ok(Tensor::from_data(TensorData::new(values,entry.shape.clone()),d))
    }
    pub fn bools<const D:usize>(&self,name:&str,d:&Device)->Result<Tensor<D,Bool>>{Ok(self.read::<D>(name,d)?.greater_elem(0.5))}
    pub fn ints<const D:usize>(&self,name:&str,d:&Device)->Result<Tensor<D,Int>>{Ok(self.read::<D>(name,d)?.int())}
    pub fn observation(&self,d:&Device)->Result<Observation>{
        let n=self.tensors["heroes"].shape[0];
        let combat=|name:&str,group:&str,fields|->Result<Tensor<3>>{if self.tensors.contains_key(name){self.read(name,d)}else{Ok(Tensor::zeros([n,self.tensors[group].shape[1],fields*2],d))}};
        Ok(Observation {
            combat_profile:self.tensors.contains_key("hero_combat"),
            hero_combat:combat("hero_combat","heroes",crate::contract::COMBAT_UNIT_COUNT)?,
            creep_combat:combat("creep_combat","creeps",crate::contract::COMBAT_UNIT_COUNT)?,
            tower_combat:combat("tower_combat","towers",crate::contract::COMBAT_UNIT_COUNT)?,
            projectile_combat:combat("projectile_combat","projectiles",crate::contract::COMBAT_PROJECTILE_COUNT)?,
            heroes:self.read("heroes",d)?,creeps:self.read("creeps",d)?,towers:self.read("towers",d)?,trees:self.read("trees",d)?,
            projectiles:self.read("projectiles",d)?,skills:self.read("skills",d)?,items:self.read("items",d)?,enemy_items:self.read("enemy_items",d)?,
            terrain:self.read("terrain",d)?,global_features:self.read("global_features",d)?,previous:self.read("previous",d)?,
            hero_valid:self.bools("hero_valid",d)?,creep_valid:self.bools("creep_valid",d)?,creep_ally:self.bools("creep_ally",d)?,
            tree_valid:self.bools("tree_valid",d)?,projectile_valid:self.bools("projectile_valid",d)?,
            action_mask:self.bools("action_mask",d)?,buy_mask:self.bools("buy_mask",d)?,item_mask:self.bools("item_mask",d)?,
            attack_mask:self.bools("attack_mask",d)?,tree_mask:self.bools("tree_mask",d)?,tp_mask:self.bools("tp_mask",d)?,self_mask:self.bools("self_mask",d)?,
        })
    }
}

struct Importer<'a>{bundle:&'a Bundle,path:Vec<String>,error:Option<anyhow::Error>,count:usize}
impl ModuleMapper for Importer<'_>{
    fn enter_module(&mut self,name:&str,_:&str){self.path.push(name.into());}
    fn exit_module(&mut self,_:&str,_:&str){self.path.pop();}
    fn map_float<const D:usize>(&mut self,p:Param<Tensor<D>>)->Param<Tensor<D>>{
        let name=self.path.join(".");
        match self.bundle.read::<D>(&name,&p.val().device()){
            Ok(t) if t.dims()==p.val().dims()=>{self.count+=1;p.map(|_|t)}
            Ok(_)=>{self.error=Some(anyhow::anyhow!("shape mismatch for {name}"));p}
            Err(e)=>{self.error=Some(e);p}
        }
    }
}
pub fn import(root:&Path,d:&Device)->Result<Policy>{
    let bundle=Bundle::open(root)?;
    let mut importer=Importer{bundle:&bundle,path:vec![],error:None,count:0};
    let ggd=bundle.tensors.get("hero.first.weight").is_some_and(|e|e.shape[0]==2*(crate::contract::UNIT_COUNT+crate::contract::COMBAT_UNIT_COUNT));
    let model=if ggd{Policy::new_ggd(d)}else{Policy::new(d)}.map(&mut importer);
    if let Some(e)=importer.error{return Err(e)}
    ensure!(importer.count==bundle.tensors.len(),"unused tensors in import");
    Ok(model)
}

pub fn export(model:&Policy,root:&Path)->Result<()> {
    use burn::module::{ModuleVisitor,Param};
    struct Exporter {root:PathBuf,path:Vec<String>,entries:BTreeMap<String,Entry>,error:Option<anyhow::Error>}
    impl ModuleVisitor for Exporter {
        fn enter_module(&mut self,name:&str,_:&str){self.path.push(name.into());}
        fn exit_module(&mut self,_:&str,_:&str){self.path.pop();}
        fn visit_float<const D:usize>(&mut self,p:&Param<Tensor<D>>){
            let write=||->Result<Entry>{
                let t=p.val().without_autodiff();let shape=t.dims().to_vec();let file=format!("{:03}.f32",self.entries.len());
                let data=t.into_data().try_to_vec::<f32>()?;
                let bytes:Vec<_>=data.iter().flat_map(|x|x.to_le_bytes()).collect();std::fs::write(self.root.join(&file),bytes)?;
                Ok(Entry{shape,file,sha256:String::new()})
            };
            match write(){Ok(entry)=>{self.entries.insert(self.path.join("."),entry);},Err(e)=>self.error=Some(e)}
        }
    }
    std::fs::create_dir_all(root)?;let mut export=Exporter{root:root.into(),path:vec![],entries:BTreeMap::new(),error:None};model.visit(&mut export);
    if let Some(e)=export.error{return Err(e)}
    crate::artifacts::atomic_json(&root.join("manifest.json"),&Bundle{tensors:export.entries,root:root.into()})?;Ok(())
}

pub fn validate(root:&Path,d:&Device)->Result<serde_json::Value>{
    let model=import(&root.join("weights"),d)?;
    let fixture=Bundle::open(&root.join("fixture"))?;
    let b=fixture.observation(d)?;let n=b.heroes.dims()[0];
    let encoded=model.encode(&b);let mut hs=vec![];let mut cs=vec![];
    let mut h=Tensor::zeros([1,256],d);let mut c=h.clone();
    for i in 0..n {
        if i%128==0 {h=Tensor::zeros([1,256],d);c=h.clone();}
        (h,c)=model.memory.step(encoded.fused.clone().slice([i..i+1]),h,c);
        hs.push(h.clone());cs.push(c.clone());
    }
    let h=Tensor::cat(hs,0);let c=Tensor::cat(cs,0);
    let sampler=crate::sampling::Sampler::new(n,7,d);
    let result=model.heads(&b,h.clone(),encoded.keys.clone(),Some(fixture.ints("labels",d)?),&sampler);
    let mut errors=BTreeMap::new();
    macro_rules! check {($name:literal,$actual:expr,$rank:literal)=>{{
        let expected=fixture.read::<$rank>(concat!("expected_",$name),d)?;
        let actual=$actual;
        let difference=(actual-expected.clone()).abs();
        let max=difference.clone().max().into_scalar::<f32>();
        let bad=(difference/(expected.abs()*0.0005+0.0005)).max().into_scalar::<f32>();
        ensure!(bad.is_finite() && bad<=1.,"{} parity failed: max error {}, normalized error {}",$name,max,bad);
        errors.insert($name,max);
    }}}
    check!("fused",encoded.fused,2);check!("keys",encoded.keys,3);check!("h",h,2);check!("c",c,2);
    check!("value",result.value,1);check!("logp",result.logp,1);
    Ok(serde_json::json!({"passed":true,"steps":n,"absolute_errors":errors,"atol":0.0005,"rtol":0.0005}))
}

/// Diagnostic CPU readback after one invocation of the production GPU kernel.
pub fn validate_previous(root:&Path,d:&Device)->Result<serde_json::Value>{
    let fixture=Bundle::open(root)?;let input=fixture.read::<2>("packed",d)?;
    let n=input.dims()[0];ensure!(input.dims()[1]==crate::observe::WIDTH && n%2==0,"invalid fixture layout");
    let bases=crate::runtime::primitive(fixture.read::<1>("bases",d)?);
    let observed=crate::observe::Observed{packed:input.cast(burn::tensor::DType::F16),refs:bases.clone(),reset:Tensor::zeros([n],d)};
    let state=crate::runtime::primitive(fixture.read::<1>("state",d)?);
    let actual=observed.previous(fixture.ints("actions",d)?,&bases,&state).cast(burn::tensor::DType::F16)
        .cast(burn::tensor::DType::F32).into_data().try_to_vec::<f32>()?;
    let expected=fixture.read::<2>("expected",d)?.into_data().try_to_vec::<f32>()?;
    ensure!(actual.len()==n*crate::contract::PREVIOUS && actual.len()==expected.len(),"invalid expected shape");
    let mismatches:Vec<_>=actual.iter().zip(&expected).enumerate().filter_map(|(i,(a,e))|
        (a.to_bits()!=e.to_bits()).then(||serde_json::json!({"actor":i/30,"field":i%30,"actual":a,"expected":e,"absolute_error":(a-e).abs()}))).collect();
    Ok(serde_json::json!({"passed":mismatches.is_empty(),"actors":n,"scalars":actual.len(),"mismatches":mismatches,
        "comparison":"bit identity after FP16 next-observation storage", "training_updates":0,
        "fixture_manifest_sha256":crate::runtime::file_sha256(&root.join("manifest.json"))?}))
}
