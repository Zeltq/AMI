//! User-authorized P16 base-to-mid control phase. Uses existing legacy MOVE physics.
use burn::tensor::{Tensor,Int};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use crate::{world::*,contract::*,runtime::{argument,primitive,tensor}};
pub struct Navigation {buffer:CubeTensor}
pub struct Phase {pub ready:Tensor<1>,pub reset:Tensor<1>}
#[cube(launch)]
fn update(state:&[f32],meta:&[f32],buffer:&mut [f32],actors:usize){
    let a=ABSOLUTE_POS;if a<actors {let m=a/2;let p=(m*UNITS+a%2)*STRIDE;let r=a*4;let generation=meta[m*META+1];let counter=meta[m*META+6];
        if buffer[r]!=generation {buffer[r]=generation;buffer[r+1]=0.0f32;buffer[r+2]=-1.0f32;}
        let x=state[p+UNIT_X];let y=state[p+UNIT_Y];
        if buffer[r+1]==0.0f32 && x*x+y*y<4000000.0f32 && (x-y).abs()<1200.0f32 {buffer[r+1]=1.0f32;buffer[r+2]=counter;}
        buffer[r+3]=if buffer[r+1]>0.0f32 && buffer[r+2]==counter {1.0f32}else{0.0f32};
    }
}
#[cube(launch)]
fn replace_command(state:&[f32],meta:&[f32],ready:&[f32],commands:&mut [f32],legacy:&mut [f32],actors:usize){
    let a=ABSOLUTE_POS;if a<actors && ready[a]==0.0f32 {let m=a/2;let side=a%2;let p=(m*UNITS+side)*STRIDE;let tower=(m*UNITS+2+side*2)*STRIDE;
        let tx=state[tower+UNIT_X];let ty=state[tower+UNIT_Y];let length=(tx*tx+ty*ty).sqrt().max(1.0f32);
        // A reachable point on the river side of T1; Radiant's occupied center lies outside radius2000.
        let gx=tx-tx/length*350.0f32;let gy=ty-ty/length*350.0f32;
        let slot=meta[m*META+6]as usize%4;let k=(a*4+slot)*COMMAND_FIELDS;
        commands[k+1]=1.0f32;commands[k+3]=0.0f32;commands[k+4]=-1.0f32;commands[k+5]=gx;commands[k+6]=gy;commands[k+8]=0.0f32;
        let dx=gx-state[p+UNIT_X];let dy=gy-state[p+UNIT_Y];let scale=(dx.abs()/1024.0f32).max(dy.abs()/1024.0f32).max(1.0f32);
        legacy[a*6]=1.0f32;legacy[a*6+2]=0.0f32;legacy[a*6+3]=0.0f32;
        legacy[a*6+4]=(64.0f32+dx/scale/16.0f32).round().clamp(0.0f32,128.0f32);legacy[a*6+5]=(64.0f32+dy/scale/16.0f32).round().clamp(0.0f32,128.0f32);
    }
}
impl Navigation {
    pub fn new(w:&World)->Self{Self{buffer:primitive(Tensor::<2>::zeros([w.matches*2,4],&w.device))}}
    pub fn observe(&self,w:&World)->Phase{
        update::launch(&w.state.client,CubeCount::Static((w.matches*2).div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&w.state),argument(&w.meta),argument(&self.buffer),w.matches*2);
        let data=tensor::<2>(self.buffer.clone());Phase{ready:data.clone().slice([0..w.matches*2,1..2]).reshape([w.matches*2]),reset:data.slice([0..w.matches*2,3..4]).reshape([w.matches*2])}
    }
    pub fn enqueue(&self,w:&World,ready:Tensor<1>,legacy:Tensor<2,Int>)->Tensor<2,Int>{
        let ready=primitive(ready);let result=primitive(legacy.float());replace_command::launch(&w.state.client,CubeCount::Static((w.matches*2).div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&w.state),argument(&w.meta),argument(&ready),argument(&w.commands),argument(&result),w.matches*2);tensor::<2>(result).int()
    }
}
#[cfg(test)] mod tests {
    use super::*;use crate::{chio_runtime::Runtime,teacher_v10::Version,sampling::Sampler};
    fn runtime()->Runtime{let d=crate::runtime::device();let mut w=World::load(std::path::Path::new("C:/AMI/runs/p16/environment/engine.json"),2,&d).unwrap();w.engine.chio_script_navigation=true;Runtime::new(w,Version::V10,Version::V9,false,false,true).unwrap()}
    #[test] fn p16_navigation_masks_loss_uses_legacy_move_and_resets_once_on_handoff(){
        let mut r=runtime();let d=r.world.device.clone();let rng=Sampler::new(2,19,&d);let first=r.observe();assert_eq!(first.policy_valid.clone().sum().into_scalar::<f32>(),0.);
        let _=r.advance(&first,crate::chio_observe::initial_previous(4,&d),&rng,600.);
        let queue=tensor::<1>(r.world.commands.clone()).into_data().try_to_vec::<f32>().unwrap();for a in 0..4{assert_eq!(queue[a*4*COMMAND_FIELDS+1],1.);}
        let mut state=tensor::<1>(r.world.state.clone()).into_data().try_to_vec::<f32>().unwrap();for a in 0..4{let p=(a/2*UNITS+a%2)*STRIDE;state[p+UNIT_X]=if a%2==0{-1200.}else{1200.};state[p+UNIT_Y]=state[p+UNIT_X];}
        r.world.state=primitive(Tensor::<1>::from_data(burn::tensor::TensorData::new(state.clone(),[2*UNITS*STRIDE]),&d));
        let hand=r.observe();assert_eq!(hand.policy_valid.clone().sum().into_scalar::<f32>(),4.);assert_eq!(hand.observation.reset.clone().sum().into_scalar::<f32>(),4.);
        let (b,_)=crate::chio_observe::decode(&hand.observation,16,32,16);let previous=b.previous.into_data().try_to_vec::<f32>().unwrap();for a in 0..4{assert_eq!(previous[a*55+1],1.);assert_eq!(previous[a*55+49],1.);}
        assert_eq!(r.observe().observation.reset.sum().into_scalar::<f32>(),4.);let _=r.advance(&hand,crate::chio_observe::initial_previous(4,&d),&rng,600.);
        assert_eq!(r.observe().observation.reset.sum().into_scalar::<f32>(),0.);
        for a in 0..4{let p=(a/2*UNITS+a%2)*STRIDE;state[p+UNIT_X]=6000.;state[p+UNIT_Y]=6000.;}r.world.state=primitive(Tensor::<1>::from_data(burn::tensor::TensorData::new(state,[2*UNITS*STRIDE]),&d));
        assert_eq!(r.observe().policy_valid.sum().into_scalar::<f32>(),4.);r.reset(&[true,false]);let valid=r.observe().policy_valid.into_data().try_to_vec::<f32>().unwrap();assert_eq!(valid,vec![0.,0.,1.,1.]);
    }
    #[test] fn p16_navigation_brings_both_teams_to_the_original_mid_region(){
        let mut r=runtime();let d=r.world.device.clone();let rng=Sampler::new(2,29,&d);let mut arrivals=[None;4];
        for step in 0..400{let f=r.observe();let ready=f.policy_valid.clone().into_data().try_to_vec::<f32>().unwrap();for a in 0..4{if ready[a]>0.&&arrivals[a].is_none(){arrivals[a]=Some(step);}}
            if arrivals.iter().all(Option::is_some){break;}let _=r.advance(&f,crate::chio_observe::initial_previous(4,&d),&rng,600.);
        }
        assert!(arrivals.iter().all(Option::is_some),"{arrivals:?}");println!("scripted navigation arrival decisions: {arrivals:?}");
    }
}
