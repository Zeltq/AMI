//! P09b potential reward; uses privileged native state and never policy inputs.
use crate::{world::*,contract::*,runtime::{primitive,tensor,argument}};
use burn::tensor::Tensor;
use cubecl::prelude::*;
pub const NAMES:[&str;5]=["outcome","lh","deny","hp","mana"];
#[cube(launch)]
fn snapshot_kernel(state:&[f32],meta:&[f32],audit:&[f32],out:&mut[f32],actors:usize,mana:f32){
    let a=ABSOLUTE_POS;if a<actors {
        let m=a/2;let s=a%2;let p=m*UNITS*STRIDE+s*STRIDE;let q=m*UNITS*STRIDE+(1-s)*STRIDE;let r=a*6;
        let done=audit[m*crate::scripted_teacher::AUDIT_FIELDS];
        out[r]=if done>0.0f32 {10.0f32*meta[m*META+17]*if s==0{1.0f32}else{-1.0f32}}else{0.0f32};
        out[r+1]=(state[p+UNIT_LAST_HITS]-state[q+UNIT_LAST_HITS])*0.5f32*(1.0f32-done);
        out[r+2]=(state[p+UNIT_DENIES]-state[q+UNIT_DENIES])*0.2f32*(1.0f32-done);
        out[r+3]=(state[p+UNIT_HEALTH]/state[p+UNIT_HEALTH_MAX].max(1.0f32)-state[q+UNIT_HEALTH]/state[q+UNIT_HEALTH_MAX].max(1.0f32))*3.0f32*(1.0f32-done);
        out[r+4]=(state[p+UNIT_MANA]/state[p+UNIT_MANA_MAX].max(1.0f32)-state[q+UNIT_MANA]/state[q+UNIT_MANA_MAX].max(1.0f32))*mana*(1.0f32-done);
        out[r+5]=done;
    }
}
pub fn snapshot(w:&World,t:&crate::scripted_teacher::Teacher,mana:f32)->Tensor<2>{
    let out=primitive(Tensor::<2>::zeros([w.matches*2,6],&w.device));
    snapshot_kernel::launch(&w.state.client,CubeCount::Static((w.matches*2).div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&w.state),argument(&w.meta),argument(&t.audit),argument(&out),w.matches*2,mana);tensor(out)
}
pub fn components(before:Tensor<2>,after:Tensor<2>)->Tensor<2>{
    let n=before.dims()[0];let live=1.0f32-before.clone().slice([0..n,5..6]);
    let outcome=after.clone().slice([0..n,0..1])-before.clone().slice([0..n,0..1]);
    let shaped=after.slice([0..n,1..5])*0.99-before.slice([0..n,1..5]);
    Tensor::cat(vec![outcome,shaped],1)*live
}
/// Tracks both players, including the gamma-corrected telescoping residual.
pub struct Tracker {before:Tensor<2>,pub totals:Tensor<2>,pub residual:Tensor<2>,mana:f32}
impl Tracker {
    pub fn new(w:&World,t:&crate::scripted_teacher::Teacher,mana:f32)->Self{
        let before=snapshot(w,t,mana);let n=w.matches*2;
        Self{residual:before.clone().slice([0..n,1..5]),before,totals:Tensor::zeros([n,5],&w.device),mana}
    }
    pub fn step(&mut self,w:&World,t:&crate::scripted_teacher::Teacher)->Tensor<2>{
        let after=snapshot(w,t,self.mana);let n=w.matches*2;let c=components(self.before.clone(),after.clone());
        let live=1.0f32-self.before.clone().slice([0..n,5..6]);
        self.residual=self.residual.clone()+c.clone().slice([0..n,1..5])+after.clone().slice([0..n,1..5])*0.01*live;
        self.totals=self.totals.clone()+c.clone();self.before=after;c
    }
    pub fn reset(&mut self,w:&World,t:&crate::scripted_teacher::Teacher,mask:Tensor<1,burn::tensor::Bool>){
        let n=w.matches*2;let mask=mask.unsqueeze_dim::<2>(1);self.before=snapshot(w,t,self.mana);
        self.totals=self.totals.clone().mask_fill(mask.clone(),0.);
        self.residual=self.residual.clone().mask_where(mask,self.before.clone().slice([0..n,1..5]));
    }
    pub fn enrich(&self,games:&mut[serde_json::Value])->anyhow::Result<()>{
        let values=self.totals.clone().into_data().try_to_vec::<f32>()?;let residual=self.residual.clone().into_data().try_to_vec::<f32>()?;
        for (m,g) in games.iter_mut().enumerate(){for s in 0..2{let a=m*2+s;
            g["sides"][s]["reward_components"]=serde_json::json!(&values[a*5..a*5+5]);
            g["sides"][s]["reward"]=serde_json::json!(values[a*5..a*5+5].iter().sum::<f32>());
            g["sides"][s]["shaping_telescoping_residual"]=serde_json::json!(&residual[a*4..a*4+4]);
        }}Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;use burn::tensor::TensorData;
    #[test]
    fn p09b_potential_terminal_and_discount(){
        let d=crate::runtime::device();
        let a=Tensor::from_data(TensorData::new(vec![0.,2.,1.,-.5,.2,0.,0.,-2.,-1.,.5,-.2,0.],[2,6]),&d);
        let b=Tensor::from_data(TensorData::new(vec![10.,0.,0.,0.,0.,1.,-10.,0.,0.,0.,0.,1.],[2,6]),&d);
        assert_eq!(components(a,b.clone()).into_data().try_to_vec::<f32>().unwrap(),vec![10.,-2.,-1.,.5,-.2,-10.,2.,1.,-.5,.2]);
        assert!(components(b.clone(),b).into_data().try_to_vec::<f32>().unwrap().iter().all(|v|*v==0.));
    }
}
