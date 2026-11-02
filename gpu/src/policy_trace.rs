//! Bounded actor0 input/history capture, downloaded only after the diagnostic.
use anyhow::Result;
use burn::tensor::{Tensor,DType,f16};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use std::path::Path;
use crate::{observe::WIDTH,runtime::{primitive,tensor,argument}};
pub const AUX:usize=1299;
pub struct PolicyTrace {pub observations:CubeTensor,pub aux:CubeTensor,pub capacity:usize}

#[cube(launch)]
fn store(input:&[f16],extra:&[f32],observations:&mut [f16],aux:&mut [f32],step:usize){
    let f=ABSOLUTE_POS;
    if f<WIDTH {observations[step*WIDTH+f]=input[f];}
    if f<AUX {aux[step*AUX+f]=extra[f];}
}
impl PolicyTrace {
    pub fn new(capacity:usize,d:&burn::tensor::Device)->Self {
        Self {observations:primitive(Tensor::<2>::zeros([capacity,WIDTH],(d,DType::F16))),aux:primitive(Tensor::<2>::zeros([capacity,AUX],d)),capacity}
    }
    pub fn record(&self,step:usize,observation:Tensor<2>,extra:Tensor<2>){
        assert!(step<self.capacity);let input=primitive(observation);let extra=primitive(extra);
        store::launch(&self.observations.client,CubeCount::Static(WIDTH.div_ceil(256)as u32,1,1),CubeDim::new_1d(256),argument(&input),argument(&extra),argument(&self.observations),argument(&self.aux),step);
    }
    pub fn save(&self,root:&Path,steps:usize)->Result<()> {
        let observations=tensor::<2>(self.observations.clone()).slice([0..steps,0..WIDTH]).into_data().try_to_vec::<f16>()?;
        let aux=tensor::<2>(self.aux.clone()).slice([0..steps,0..AUX]).into_data().try_to_vec::<f32>()?;
        let input_file=root.join("policy-inputs.f16");let aux_file=root.join("policy-aux.f32");
        std::fs::write(&input_file,observations.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect::<Vec<_>>())?;
        std::fs::write(&aux_file,aux.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
        crate::artifacts::atomic_json(&root.join("policy-trace.json"),&serde_json::json!({"actor":0,"team":2,"steps":steps,"observation_shape":[steps,WIDTH],"aux_shape":[steps,AUX],"observation_dtype":"f16_le","aux_dtype":"f32_le","columns":{"game_tick":[0,1],"reset":[1,2],"before_h":[2,258],"before_c":[258,514],"after_h":[514,770],"after_c":[770,1026],"fused":[1026,1282],"action_logits_unmasked":[1282,1293],"actions":[1293,1299]},"inputs_sha256":crate::runtime::file_sha256(&input_file)?,"aux_sha256":crate::runtime::file_sha256(&aux_file)?,"scope":"ActualGPU actor0 inputs and states before/after policy inference; one final diagnostic download, noCPU dependency per step."}))?;Ok(())
    }
}
