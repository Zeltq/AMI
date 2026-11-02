//! Frozen opponents stay resident in VRAM. Routing never reads match state on CPU.
use anyhow::Result;
use burn::{module::Module,tensor::{Tensor,Bool,Device}};
use crate::{model::{Policy,Observation,Heads},world::{World,META},runtime::tensor,sampling::Sampler};
use std::path::Path;
use cubecl::prelude::*;

// Terminal metrics stay on device. Self-play games with two learners are
// excluded; these counters describe only games against frozen opponents.
#[cube(launch)]
fn outcomes(meta:&mut [f32],signals:&[f32],state:&[f32],matches:usize){
    let m=ABSOLUTE_POS;
    if m<matches {
        let a=m*2*crate::experience::SIGNALS;let b=a+crate::experience::SIGNALS;let mb=m*META;
        if signals[a+9]>0.0f32 && signals[a+11]+signals[b+11]==1.0f32 {
            let radiant=signals[a+11];let result=meta[mb+17]*(radiant*2.0f32-1.0f32);
            if result>0.0f32 {meta[mb+18]+=1.0f32;}else if result<0.0f32 {meta[mb+19]+=1.0f32;}else{meta[mb+20]+=1.0f32;}
            let learner=if radiant>0.0f32 {0usize}else{1usize};let opponent=1-learner;
            let p=(m*crate::world::UNITS+learner)*crate::world::STRIDE;let q=(m*crate::world::UNITS+opponent)*crate::world::STRIDE;
            meta[mb+21]+=state[p+crate::contract::UNIT_LAST_HITS];meta[mb+22]+=state[q+crate::contract::UNIT_LAST_HITS];
            meta[mb+23]+=state[p+crate::contract::UNIT_DENIES];meta[mb+24]+=state[q+crate::contract::UNIT_DENIES];
            meta[mb+25]+=f32::max(0.0f32,meta[mb])/30.0f32;
            meta[mb+26]+=state[p+crate::contract::UNIT_LEVEL];meta[mb+27]+=state[q+crate::contract::UNIT_LEVEL];
            meta[mb+28+learner]+=1.0f32;if result>0.0f32 {meta[mb+30+learner]+=1.0f32;}
        }
    }
}
pub fn record_outcomes(world:&World,signals:Tensor<2>){
    let signals=crate::runtime::primitive(signals);
    outcomes::launch(&world.state.client,CubeCount::Static(world.matches.div_ceil(32) as u32,1,1),CubeDim::new_1d(32),
        crate::runtime::argument(&world.meta),crate::runtime::argument(&signals),crate::runtime::argument(&world.state),world.matches);
}
pub fn outcome_report(values:Vec<f32>)->serde_json::Value {
    let games=values[0]+values[1]+values[2];let count=games.max(1.);
    serde_json::json!({"games":games,"wins":values[0],"losses":values[1],"draws":values[2],"score":if games>0.{Some((values[0]+values[2]*0.5)/games)}else{None},
        "learner_mean_last_hits":values[3]/count,"opponent_mean_last_hits":values[4]/count,
        "learner_mean_denies":values[5]/count,"opponent_mean_denies":values[6]/count,"mean_game_seconds":values[7]/count,
        "learner_mean_level":values[8]/count,"opponent_mean_level":values[9]/count,
        "learner_radiant_games":values[10],"learner_dire_games":values[11],"learner_radiant_wins":values[12],"learner_dire_wins":values[13],
        "estimate":"nonstationary training diagnostic; not held-out strength or Dota transfer"})
}

pub fn load(path:&Path,d:&Device)->Result<Policy>{
    if path.is_dir() && path.join("manifest.json").exists(){crate::bridge::import(path,d)}
    else {let file=if path.is_dir(){path.join("model.bpk")}else{path.to_path_buf()};
        Ok(Policy::new(d).try_load_record(burn::store::ModuleRecord::load(file)?)?)}
}
pub struct Pool {models:Vec<Policy>,groups:Vec<Tensor<1,Bool>>,side:Tensor<1>,pub identities:Vec<String>}
impl Pool {
    pub fn evaluation(path:&str,matches:usize,d:&Device)->Result<Self>{
        anyhow::ensure!(matches%2==0,"evaluation needs equal sides");
        let mut pool=Self::new(&[path.to_string()],matches,d)?;
        pool.groups[0]=Tensor::<1>::ones([matches*2],d).greater_elem(0.5);
        let side:Vec<f32>=(0..matches*2).map(|a|((a%2+a/2%2)%2)as f32).collect();
        pool.side=Tensor::from_data(burn::tensor::TensorData::new(side,[matches*2]),d);Ok(pool)
    }
    pub fn new(paths:&[String],matches:usize,d:&Device)->Result<Self>{
        anyhow::ensure!(paths.len()<=4,"at most four resident opponent policies");
        let mut models=vec![];let mut groups=vec![];let mut identities=vec![];
        for (i,path) in paths.iter().enumerate(){
            models.push(load(Path::new(path),d)?.valid());
            let path=Path::new(path);let file=if path.is_dir(){path.join(if path.join("manifest.json").exists(){"manifest.json"}else{"model.bpk"})}else{path.to_path_buf()};
            identities.push(crate::runtime::file_sha256(&file)?);
            let ids:Vec<f32>=(0..matches*2).map(|a|if a/2%(paths.len()+1)==i+1 {1.}else{0.}).collect();
            groups.push(Tensor::<1>::from_data(burn::tensor::TensorData::new(ids,[matches*2]),d).greater_elem(0.5));
        }
        let side:Vec<f32>=(0..matches*2).map(|a|((a%2+a/2/(paths.len()+1))%2) as f32).collect();
        Ok(Self{models,groups,side:Tensor::from_data(burn::tensor::TensorData::new(side,[matches*2]),d),identities})
    }
    pub fn decide(&self,world:&World,b:&Observation,prior_h:Tensor<2>,prior_c:Tensor<2>,mut h:Tensor<2>,mut c:Tensor<2>,mut heads:Heads,sampler:&Sampler,version:usize)
        ->(Tensor<2>,Tensor<2>,Heads,Tensor<2>,Tensor<2>){
        let n=world.matches*2;let d=world.device.clone();let mut learn=Tensor::<1>::ones([n],&d);let mut versions=Tensor::<1>::full([n],version as f32,&d);
        let episodes=tensor::<1>(world.meta.clone()).reshape([world.matches,META]).slice([0..world.matches,1..2]).expand([world.matches,2]).reshape([n]);
        let side=(episodes+self.side.clone()).remainder_scalar(2).equal_elem(0.);
        for (i,model) in self.models.iter().enumerate(){
            let mask=side.clone().bool_and(self.groups[i].clone());let rows=mask.clone().unsqueeze_dim::<2>(1);
            let encoded=model.encode(b);let (next_h,next_c)=model.memory.step(encoded.fused,prior_h.clone(),prior_c.clone());
            let opponent=model.heads(b,next_h.clone(),encoded.keys,None,sampler);
            h=h.mask_where(rows.clone(),next_h);c=c.mask_where(rows.clone(),next_c);
            heads.actions=heads.actions.mask_where(rows,opponent.actions);
            heads.logp=heads.logp.mask_where(mask.clone(),opponent.logp);heads.value=heads.value.mask_where(mask.clone(),opponent.value);
            learn=learn.mask_fill(mask.clone(),0.);versions=versions.mask_fill(mask,-((i+1) as f32));
        }
        (h,c,heads,learn.unsqueeze_dim::<2>(1),versions.unsqueeze_dim::<2>(1))
    }
}

