//! Per-episode opponent routing; frozen teachers and simulator are unchanged.
use anyhow::{ensure,Result};
use burn::tensor::{Tensor,TensorData,Int};
use crate::{model::Policy,observe::Observed,sampling::Sampler,world::World,scripted_teacher::Teacher};
use std::path::Path;

#[derive(Clone,serde::Serialize,serde::Deserialize,Debug,PartialEq)]
pub struct Assignment {pub kind:String,pub model:Option<usize>}
fn next(rng:&mut u32)->u32 {*rng^=*rng<<13;*rng^=*rng>>17;*rng^=*rng<<5;*rng}
pub fn choose(rng:&mut u32,count:usize)->Assignment {
    assert!(count>=2);
    let (kind,model)=match next(rng)%10 {
        0..=1=>("latest",Some(count-1)),
        2..=3=>("older",Some(next(rng)as usize%(count-1))),
        4..=6=>("v10",None),_ =>("v9",None),
    };Assignment{kind:kind.into(),model}
}
struct Group {model:usize,ids:Tensor<1,Int>,h:Tensor<2>,c:Tensor<2>,sampler:Sampler}
pub struct Pool {
    pub paths:Vec<String>,pub assignments:Vec<Assignment>,pub draws:[usize;4],
    models:Vec<Policy>,rng:u32,groups:Vec<Group>,v9:Teacher,
    pending:Option<((burn_cubecl::tensor::CubeTensor,burn_cubecl::tensor::CubeTensor),Tensor<2,Int>)>,
}
impl Pool {
    pub fn new(w:&World,t:&Teacher,paths:Vec<String>,seed:u32)->Result<Self>{
        ensure!(paths.len()>=2,"mixed pool needs at least two snapshots");
        let mut v9=Teacher::new(w);v9.audit=t.audit.clone();
        let mut p=Self{paths:vec![],assignments:vec![],draws:[0;4],models:vec![],rng:seed|1,groups:vec![],v9,pending:None};
        for path in paths{p.add(&path,&w.device)?;}
        for _ in 0..w.matches {let a=p.draw();p.assignments.push(a);}
        Ok(p)
    }
    pub fn add(&mut self,path:&str,d:&burn::tensor::Device)->Result<()> {
        if self.paths.iter().any(|x|x==path){return Ok(());}
        let model=crate::bridge::import(Path::new(path),d)?.valid();ensure!(model.is_ggd_v4(),"pool requires ggd-v4");
        self.models.push(model);self.paths.push(path.into());Ok(())
    }
    fn draw(&mut self)->Assignment {let a=choose(&mut self.rng,self.paths.len());let i=match a.kind.as_str(){"latest"=>0,"older"=>1,"v10"=>2,_=>3};self.draws[i]+=1;a}
    pub fn reset(&mut self,done:&[bool]){for (m,yes) in done.iter().enumerate(){if *yes{self.assignments[m]=self.draw();}}}
    pub fn begin_block(&mut self,w:&World,sides:&[usize]){
        self.groups.clear();
        for model in 0..self.models.len(){
            let ids:Vec<i32>=self.assignments.iter().enumerate().filter(|(_,a)|a.model==Some(model)).map(|(m,_)|(m*2+1-sides[m])as i32).collect();
            let n=ids.len();if n==0{continue;}
            let h=Tensor::zeros([n,256],&w.device);
            self.groups.push(Group{model,ids:Tensor::from_data(TensorData::new(ids,[n]),&w.device),c:h.clone(),h,sampler:Sampler::new(n,next(&mut self.rng),&w.device)});
        }
    }
    pub fn scripted(&mut self,w:&World,obs:&Observed,v10:Tensor<2,Int>)->Tensor<2,Int>{
        let before=self.v9.command_snapshot();let actions=self.v9.actions(w,obs);
        self.pending=Some((before,actions.clone()));
        let mask=Tensor::<1>::from_data(TensorData::new((0..w.matches*2).map(|i|if self.assignments[i/2].kind=="v9"{1f32}else{0.}).collect::<Vec<_>>(),[w.matches*2]),&w.device).greater_elem(0.).unsqueeze_dim::<2>(1);
        v10.mask_where(mask,actions)
    }
    pub fn actions(&mut self,w:&World,obs:&Observed,mut actions:Tensor<2,Int>)->Tensor<2,Int>{
        for g in &mut self.groups {
            let selected=Observed{packed:obs.packed.clone().select(0,g.ids.clone()),reset:obs.reset.clone().select(0,g.ids.clone()),refs:obs.refs.clone()};
            let keep:Tensor<2>=1f32-selected.reset.clone().unsqueeze_dim::<2>(1);
            let b=selected.decode();let p=&self.models[g.model];let e=p.encode(&b);
            (g.h,g.c)=p.memory.step(e.fused,g.h.clone()*keep.clone(),g.c.clone()*keep);
            let predicted=p.heads(&b,g.h.clone(),e.keys,None,&g.sampler).actions;
            actions=actions.select_assign(0,g.ids.clone(),predicted);
        }
        let (before,scripted)=self.pending.take().expect("scripted before actions");
        self.v9.reconcile_commands(w,obs,before,scripted,actions.clone());actions
    }
    pub fn evidence(&self)->serde_json::Value {serde_json::json!({"paths":self.paths,"assignments":self.assignments,"draws":self.draws,"rng":self.rng})}
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn p11_pool_distribution_and_old_exclusion(){
        let mut rng=20261105;let mut counts=[0;4];
        for _ in 0..100000{let a=choose(&mut rng,7);let i=match a.kind.as_str(){"latest"=>{assert_eq!(a.model,Some(6));0},"older"=>{assert!(a.model.unwrap()<6);1},"v10"=>2,_=>3};counts[i]+=1;}
        for (n,p) in counts.iter().zip([0.2,0.2,0.3,0.3]){assert!((*n as f64/100000.-p).abs()<0.01);}
    }
}
