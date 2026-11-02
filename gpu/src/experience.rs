use burn::tensor::f16;
use burn::tensor::{Tensor,DType};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use crate::{observe::{Observed,WIDTH},runtime::{primitive,tensor,argument}};

pub const HORIZON:usize=128;
pub const SIGNALS:usize=13;
pub struct Experience {pub observations:Vec<CubeTensor>,pub signals:CubeTensor,pub actors:usize}

#[cube(launch)]
fn store_observation(input:&[f16],output:&mut [f16],step:usize,actors:usize,offset:usize){
    let i=ABSOLUTE_POS;
    if i<actors*WIDTH {let a=i/WIDTH;let f=i%WIDTH;output[(a*HORIZON+step)*WIDTH+f]=input[offset*WIDTH+i];}
}
#[cube(launch)]
fn store_signal(input:&[f32],output:&mut [f32],step:usize,actors:usize){
    let i=ABSOLUTE_POS;
    if i<actors*SIGNALS {let a=i/SIGNALS;let f=i%SIGNALS;output[(a*HORIZON+step)*SIGNALS+f]=input[i];}
}
#[cube(launch)]
fn advantages(signals:&[f32],bootstrap:&[f32],advantages:&mut [f32],returns:&mut [f32],actors:usize){
    let a=ABSOLUTE_POS;
    if a<actors {
        let mut next=bootstrap[a];let mut carry=(0.0f32).runtime();
        for reverse in 0..HORIZON {
            let t=HORIZON-1-reverse;let p=(a*HORIZON+t)*SIGNALS;
            let alive=1.0f32-signals[p+9];let value=signals[p+7];
            let delta=signals[p+8]+next*alive-value;
            carry=delta+alive*carry;
            advantages[a*HORIZON+t]=carry;returns[a*HORIZON+t]=carry+value;next=value;
        }
    }
}

impl Experience {
    pub fn buckets(&self)->(usize,usize,usize){
        use crate::observe::*;
        let peak=|offset,capacity,minimum| {
            let peaks=self.observations.iter().map(|shard|{
                let n=shard.meta.shape()[0];tensor::<3>(shard.clone()).slice([0..n,0..HORIZON,offset..offset+capacity]).cast(DType::F32).sum_dim(2).max().reshape([1])
            }).collect();
            let n=Tensor::cat(peaks,0).max().into_scalar::<f32>() as usize;
            n.max(minimum).next_power_of_two().min(capacity)
        };
        (peak(CREEP_VALID,CREEPS,16),peak(TREE_VALID,TREES,32),peak(PROJECTILE_VALID,PROJECTILES,16))
    }
    pub fn new(actors:usize,d:&burn::tensor::Device)->Self {
        // Vulkan drivers can reject a single multi-gigabyte storage buffer even
        // when total free VRAM suffices. Shards stay on the same GPU and retain
        // complete recurrent sequences; no observation is staged on the host.
        let observations=(0..actors).step_by(128).map(|start|primitive(Tensor::<3>::zeros([128.min(actors-start),HORIZON,WIDTH],(d,DType::F16)))).collect();
        Self {observations,
            signals:primitive(Tensor::<3>::zeros([actors,HORIZON,SIGNALS],d)),actors}
    }
    pub fn store(&self,step:usize,obs:&Observed,signals:Tensor<2>){
        let packed=primitive(obs.packed.clone());let signals=primitive(signals);
        let mut offset=0;
        for shard in &self.observations {
            let actors=shard.meta.shape()[0];
            store_observation::launch(&shard.client,CubeCount::Static((actors*WIDTH).div_ceil(256) as u32,1,1),CubeDim::new_1d(256),
                argument(&packed),argument(shard),step,actors,offset);offset+=actors;
        }
        store_signal::launch(&self.signals.client,CubeCount::Static((self.actors*SIGNALS).div_ceil(256) as u32,1,1),CubeDim::new_1d(256),
            argument(&signals),argument(&self.signals),step,self.actors);
    }
    pub fn signals(&self)->Tensor<3>{tensor(self.signals.clone())}
    pub fn gae(&self,bootstrap:Tensor<1>)->(Tensor<2>,Tensor<2>){
        let bootstrap=primitive(bootstrap);let d=crate::runtime::tensor::<1>(bootstrap.clone()).device();
        let adv=primitive(Tensor::<2>::zeros([self.actors,HORIZON],&d));let ret=primitive(Tensor::<2>::zeros([self.actors,HORIZON],&d));
        advantages::launch(&self.signals.client,CubeCount::Static(self.actors.div_ceil(64) as u32,1,1),CubeDim::new_1d(64),
            argument(&self.signals),argument(&bootstrap),argument(&adv),argument(&ret),self.actors);
        (tensor(adv),tensor(ret))
    }
}


