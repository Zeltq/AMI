//! P16 frozen opponents, assigned once per episode; teachers use the CHiO runtime.
use anyhow::{ensure,Result};
use burn::{module::Module,tensor::{Tensor,TensorData,Int,IndexingUpdateOp}};
use serde::{Serialize,Deserialize};
use crate::{chio_model::Policy,chio_observe as co,observe::Observed,sampling::Sampler,world::World};
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq)]
pub struct Assignment {pub kind:String,pub model:Option<usize>}
fn next(rng:&mut u32)->u32 {*rng^=*rng<<13;*rng^=*rng>>17;*rng^=*rng<<5;*rng}
pub fn choose(rng:&mut u32,count:usize)->Assignment {
    assert!(count>=2);
    let (kind,model)=match next(rng)%100 {0..=44=>("latest",Some(count-1)),45..=79=>("older",Some(next(rng)as usize%(count-1))),80..=89=>("v10",None),_=>("v9",None)};
    Assignment{kind:kind.into(),model}
}
struct Group {model:usize,ids:Tensor<1,Int>,h:Tensor<2>,c:Tensor<2>,sampler:Sampler}
pub struct Pool {pub paths:Vec<String>,pub assignments:Vec<Assignment>,pub draws:[usize;4],models:Vec<Policy>,rng:u32,groups:Vec<Group>}
impl Pool {
    pub fn new(w:&World,paths:Vec<String>,seed:u32)->Result<Self>{
        let mut p=Self{paths:vec![],assignments:vec![],draws:[0;4],models:vec![],rng:seed|1,groups:vec![]};
        for path in paths{p.add(&path,&w.device)?;}ensure!(p.paths.len()>=2,"mixed CHiO pool requires two distinct snapshots");
        for _ in 0..w.matches {let a=p.draw();p.assignments.push(a);}Ok(p)
    }
    fn draw(&mut self)->Assignment{let a=choose(&mut self.rng,self.paths.len());let i=match a.kind.as_str(){"latest"=>0,"older"=>1,"v10"=>2,_=>3};self.draws[i]+=1;a}
    pub fn add(&mut self,path:&str,d:&burn::tensor::Device)->Result<()> {
        if !self.paths.iter().any(|x|x==path){self.models.push(crate::chio_bridge::import(std::path::Path::new(path),d)?.valid());self.paths.push(path.into());}Ok(())
    }
    pub fn reset(&mut self,done:&[bool]){for (m,yes) in done.iter().enumerate(){if *yes{self.assignments[m]=self.draw();}}}
    pub fn begin_block(&mut self,w:&World,sides:&[usize]){
        self.groups.clear();
        for model in 0..self.models.len(){
            let ids:Vec<i32>=self.assignments.iter().enumerate().filter(|(_,a)|a.model==Some(model)).map(|(m,_)|(m*2+1-sides[m])as i32).collect();let n=ids.len();if n==0{continue;}
            let h=Tensor::zeros([n,256],&w.device);self.groups.push(Group{model,ids:Tensor::from_data(TensorData::new(ids,[n]),&w.device),c:h.clone(),h,sampler:Sampler::new(n,next(&mut self.rng),&w.device)});
        }
    }
    pub fn v9_rows(&self,sides:&[usize])->Vec<bool>{(0..sides.len()*2).map(|a|a%2!=sides[a/2]&&self.assignments[a/2].kind=="v9").collect()}
    pub fn actions(&mut self,obs:&Observed,mut actions:Tensor<2,Int>)->Tensor<2,Int>{
        for g in &mut self.groups {
            let selected=Observed{packed:obs.packed.clone().select(0,g.ids.clone()),reset:obs.reset.clone().select(0,g.ids.clone()),refs:obs.refs.clone()};
            let keep:Tensor<2>=1f32-selected.reset.clone().unsqueeze_dim(1);
            let (b,masks)=co::decode(&selected,crate::observe::CREEPS,crate::observe::TREES,crate::observe::PROJECTILES);let p=&self.models[g.model];let e=p.encode(&b);
            (g.h,g.c)=p.memory.step(e.fused,g.h.clone()*keep.clone(),g.c.clone()*keep);
            let predicted=p.heads(&masks,g.h.clone(),e.keys,None,&g.sampler).actions;
            actions=actions.select_assign(0,g.ids.clone(),predicted,IndexingUpdateOp::Assign);
        }actions
    }
    pub fn evidence(&self)->serde_json::Value{serde_json::json!({"paths":self.paths,"assignments":self.assignments,"draws":self.draws,"rng":self.rng})}
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn p16_pool_distribution_and_old_exclusion(){let mut rng=20261103;let mut counts=[0;4];for _ in 0..100000 {let a=choose(&mut rng,7);let i=match a.kind.as_str(){"latest"=>{assert_eq!(a.model,Some(6));0},"older"=>{assert!(a.model.unwrap()<6);1},"v10"=>2,_=>3};counts[i]+=1;}for (n,p) in counts.iter().zip([0.45,0.35,0.10,0.10]){assert!((*n as f64/100000.-p).abs()<0.01);}}
    #[test]fn p16_routed_v9_matches_pure_chio_v9_across_reset(){
        use crate::{chio_runtime::Runtime,teacher_v10::Version,runtime::device};
        let d=device();let path=std::path::Path::new("C:/AMI/runs/p16/environment/engine.json");
        let mut routed=Runtime::new(World::load(path,2,&d).unwrap(),Version::V10,Version::V10,false,false,true).unwrap();
        let mut pure=Runtime::new(World::load(path,2,&d).unwrap(),Version::V9,Version::V9,false,false,true).unwrap();routed.route_v9(&[true;4]);
        let rng=Sampler::new(2,19,&d);let reference=Sampler::new(2,19,&d);
        for step in 0..600 {if step==350{routed.reset(&[true,false]);pure.reset(&[true,false]);}
            let a=routed.observe();let b=pure.observe();assert_eq!(a.labels.clone().into_data().try_to_vec::<i32>().unwrap(),b.labels.clone().into_data().try_to_vec::<i32>().unwrap());
            let _=routed.advance(&a,a.labels.clone(),&rng,600.);let _=pure.advance(&b,b.labels.clone(),&reference,600.);
        }
        assert_eq!(crate::runtime::tensor::<1>(routed.world.state).into_data().try_to_vec::<f32>().unwrap(),crate::runtime::tensor::<1>(pure.world.state).into_data().try_to_vec::<f32>().unwrap());
    }
}
