//! Event-v2: raw native resource losses, normalized by the pre-step maximum.
use crate::{world::*,contract::*,runtime::{primitive,tensor,argument}};
use burn::tensor::{Tensor,Bool};
use cubecl::prelude::*;
pub const NAMES:[&str;6]=["outcome","lh","deny","own_hp","enemy_hp","mana"];
// Each actor: own LH/deny/HP/maxHP/mana/maxMana, enemy equivalents, done/result.
#[cube(launch)]
fn snapshot_kernel(state:&[f32],meta:&[f32],audit:&[f32],out:&mut[f32],actors:usize){
    let a=ABSOLUTE_POS;if a<actors {
        let m=a/2;let side=a%2;let r=a*14;
        for j in 0..2 {
            let p=m*UNITS*STRIDE+if j==0{side}else{1-side}*STRIDE;let k=r+j*6;
            out[k]=state[p+UNIT_LAST_HITS];out[k+1]=state[p+UNIT_DENIES];
            out[k+2]=state[p+UNIT_HEALTH];out[k+3]=state[p+UNIT_HEALTH_MAX];
            out[k+4]=state[p+UNIT_MANA];out[k+5]=state[p+UNIT_MANA_MAX];
        }
        out[r+12]=audit[m*crate::scripted_teacher::AUDIT_FIELDS];
        out[r+13]=meta[m*META+17]*if side==0{1.0f32}else{-1.0f32};
    }
}
pub fn snapshot(w:&World,t:&crate::scripted_teacher::Teacher)->Tensor<2>{
    let out=primitive(Tensor::<2>::zeros([w.matches*2,14],&w.device));
    snapshot_kernel::launch(&w.state.client,CubeCount::Static((w.matches*2).div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&w.state),argument(&w.meta),argument(&t.audit),argument(&out),w.matches*2);tensor(out)
}
pub fn components(before:Tensor<2>,after:Tensor<2>)->Tensor<2>{
    let n=before.dims()[0];let b=|i|before.clone().slice([0..n,i..i+1]);let a=|i|after.clone().slice([0..n,i..i+1]);
    let live=1.0f32-b(12);
    let loss=|i,max| (b(i)-a(i)).clamp_min(0.)/b(max).clamp_min(1.);
    Tensor::cat(vec![a(13)*a(12)*10.,(a(0)-b(0))*0.5,(a(1)-b(1))*0.2,loss(2,3)*-3.,loss(8,9)*3.,loss(4,5)*-0.5],1)*live
}
pub struct Tracker {before:Tensor<2>,pub totals:Tensor<2>}
impl Tracker {
    pub fn new(w:&World,t:&crate::scripted_teacher::Teacher)->Self{Self{before:snapshot(w,t),totals:Tensor::zeros([w.matches*2,6],&w.device)}}
    pub fn step(&mut self,w:&World,t:&crate::scripted_teacher::Teacher)->Tensor<2>{
        let after=snapshot(w,t);let c=components(self.before.clone(),after.clone());self.totals=self.totals.clone()+c.clone();self.before=after;c
    }
    pub fn reset(&mut self,w:&World,t:&crate::scripted_teacher::Teacher,mask:Tensor<1,Bool>){self.before=snapshot(w,t);self.totals=self.totals.clone().mask_fill(mask.unsqueeze_dim::<2>(1),0.);}
    pub fn enrich(&self,games:&mut[serde_json::Value])->anyhow::Result<()>{
        let values=self.totals.clone().into_data().try_to_vec::<f32>()?;
        for (m,g) in games.iter_mut().enumerate(){for s in 0..2{let a=m*2+s;
            g["sides"][s]["event_v2_components"]=serde_json::json!(&values[a*6..a*6+6]);
            g["sides"][s]["event_v2_reward"]=serde_json::json!(values[a*6..a*6+6].iter().sum::<f32>());
        }}Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;use burn::tensor::TensorData;
    #[test]
    fn p09c_events_raw_losses_growth_terminal_and_padding(){
        let d=crate::runtime::device();
        let before=vec![2f32,1.,500.,1000.,200.,400.,8.,4.,600.,1000.,100.,400.,0.,0.];
        let after=vec![3f32,2.,400.,1200.,150.,500.,9.,5.,400.,1200.,300.,500.,1.,1.];
        let a=Tensor::from_data(TensorData::new(before,[1,14]),&d);let b=Tensor::from_data(TensorData::new(after,[1,14]),&d);
        let v=components(a,b.clone()).into_data().try_to_vec::<f32>().unwrap();
        for (v,e) in v.iter().zip([10.,0.5,0.2,-0.3,0.6,-0.0625]){assert!((v-e).abs()<1e-6);}
        assert!(components(b.clone(),b).into_data().try_to_vec::<f32>().unwrap().iter().all(|v|*v==0.));
        let a=Tensor::from_data(TensorData::new(vec![0f32,0.,500.,1000.,200.,400.,0.,0.,600.,1000.,100.,400.,0.,0.],[1,14]),&d);
        let b=Tensor::from_data(TensorData::new(vec![0f32,0.,550.,1200.,250.,500.,3.,2.,650.,1200.,150.,500.,0.,0.],[1,14]),&d);
        assert!(components(a,b).into_data().try_to_vec::<f32>().unwrap().iter().all(|v|*v==0.));
    }
}
