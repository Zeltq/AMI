//! Optional readback of decisions and execution state, outside the policy path.
use anyhow::Result;
use burn::tensor::{Tensor,Int,Bool,activation};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use std::{path::Path,io::{BufWriter,Write}};
use crate::{world::*,contract::*,runtime::{primitive,tensor,argument},experience::HORIZON};

const WIDTH:usize=50;

pub fn parity(weights:&Path,fixture:&Path,output:&Path)->Result<serde_json::Value>{
    use burn::module::Module;
    let d=crate::runtime::device();let policy=crate::bridge::import(weights,&d)?.valid();let fixture=crate::bridge::Bundle::open(fixture)?;
    let b=fixture.observation(&d)?;let n=b.heroes.dims()[0];let resets=fixture.read::<1>("resets",&d)?.into_data().try_to_vec::<f32>()?;
    let encoded=policy.encode(&b);let mut hs=vec![];let mut h=Tensor::<2>::zeros([1,256],&d);let mut c=h.clone();
    for i in 0..n{
        if i%HORIZON==0||resets[i]>0.5{h=Tensor::zeros([1,256],&d);c=h.clone();}
        (h,c)=policy.memory.step(encoded.fused.clone().slice([i..i+1]),h,c);hs.push(h.clone());
    }
    let h=Tensor::cat(hs,0);let sampler=crate::sampling::Sampler::new(n,7,&d);
    let heads=policy.heads(&b,h.clone(),encoded.keys,Some(fixture.ints("labels",&d)?),&sampler);
    let mut results=serde_json::Map::new();
    for (name,actual) in ["action","buy","item","target","x","y"].into_iter().zip(heads.logits){
        let expected=fixture.read::<2>(&format!("expected_{name}_logits"),&d)?;
        let max=(actual.clone()-expected.clone()).abs().max().into_scalar::<f32>();
        let mask=fixture.bools::<2>(&format!("{name}_mask"),&d)?;
        let correct=actual.clone().mask_fill(mask.clone().bool_not(),-1e30).argmax(1).equal(expected.clone().mask_fill(mask.bool_not(),-1e30).argmax(1)).int().sum().into_scalar::<i32>();
        let raw=actual.argmax(1).equal(expected.argmax(1)).int().sum().into_scalar::<i32>();
        results.insert(name.into(),serde_json::json!({"max_absolute_delta":max,"masked_argmax":{"numerator":correct,"denominator":n},"raw_argmax":{"numerator":raw,"denominator":n}}));
    }
    let h_error=(h-fixture.read::<2>("expected_h",&d)?).abs().max().into_scalar::<f32>();
    let value_error=(heads.value-fixture.read::<1>("expected_value",&d)?).abs().max().into_scalar::<f32>();
    let result=serde_json::json!({"steps":n,"heads":results,"hidden_max_absolute_delta":h_error,"value_max_absolute_delta":value_error,"observation_storage":"FP16 dataset decoded to FP32 for both runtimes","recurrent_reset":128});
    crate::artifacts::atomic_json(output,&result)?;Ok(result)
}

pub struct Trace{buffer:CubeTensor,actors:usize,file:BufWriter<std::fs::File>}
impl Trace{
    pub fn new(w:&World,root:&Path)->Result<Self>{
        let columns=["game_tick","x","y","health","active","order","dest_x","dest_y","order_target","last_damage_source","last_damage_team","deaths","own_t1_x","own_t1_y","enemy_t1_x","enemy_t1_y","own_t1_distance","enemy_t1_distance","reset","action","buy","item","target","action_x","action_y","logit_0","logit_1","logit_2","logit_3","logit_4","logit_5","logit_6","logit_7","logit_8","logit_9","logit_10","p_0","p_1","p_2","p_3","p_4","p_5","p_6","p_7","p_8","p_9","p_10","finished","source_kind","source_tier"];
        crate::artifacts::atomic_json(&root.join("trace-header.json"),&serde_json::json!({"columns":columns.to_vec(),"actors":w.matches*2,"decision_ticks":w.engine.decision_ticks,"tick_hz":30,"action_layout":["action","buy","item","target","x","y"],"damage_source":"last hit CREDIT; inspect only with health loss; not a per-hit event log"}))?;
        Ok(Self{buffer:primitive(Tensor::<3>::zeros([HORIZON,w.matches*2,WIDTH],&w.device)),actors:w.matches*2,file:BufWriter::new(std::fs::File::create(root.join("trace.jsonl"))?)})
    }
    pub fn record(&self,w:&World,t:&crate::scripted_teacher::Teacher,step:usize,actions:Tensor<2,Int>,logits:Tensor<2>,mask:Tensor<2,Bool>,reset:Tensor<1>){
        let probabilities=primitive(activation::softmax(logits.clone().mask_fill(mask.bool_not(),-1e30),1));let logits=primitive(logits);let actions=primitive(actions.float());let reset=primitive(reset);
        record::launch(&w.state.client,CubeCount::Static(self.actors.div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&w.state),argument(&w.meta),argument(&t.audit),argument(&actions),argument(&logits),argument(&probabilities),argument(&reset),argument(&self.buffer),self.actors,step);
    }
    pub fn flush(&mut self,steps:usize)->Result<()>{
        let values=tensor::<3>(self.buffer.clone()).slice([0..steps,0..self.actors,0..WIDTH]).into_data().try_to_vec::<f32>()?;
        for (i,row) in values.chunks_exact(WIDTH).enumerate(){serde_json::to_writer(&mut self.file,&serde_json::json!({"actor":i%self.actors,"v":row}))?;self.file.write_all(b"\n")?;}
        self.file.flush()?;Ok(())
    }
}
#[cube(launch)]
fn record(state:&[f32],meta:&[f32],audit:&[f32],actions:&[f32],logits:&[f32],probabilities:&[f32],reset:&[f32],out:&mut [f32],actors:usize,step:usize){
    let a=ABSOLUTE_POS;
    if a<actors{
        let m=a/2;let side=a%2;let base=m*UNITS*STRIDE;let p=base+side*STRIDE;let own=base+(2+side*2)*STRIDE;let enemy=base+(4-side*2)*STRIDE;let o=(step*actors+a)*WIDTH;
        out[o]=meta[m*META];out[o+1]=state[p+UNIT_X];out[o+2]=state[p+UNIT_Y];out[o+3]=state[p+UNIT_HEALTH];out[o+4]=state[p+ACTIVE];out[o+5]=state[p+ORDER];out[o+6]=state[p+DEST_X];out[o+7]=state[p+DEST_Y];out[o+8]=state[p+TARGET];out[o+9]=state[p+CREDIT];out[o+10]=state[p+CREDIT_TEAM];out[o+11]=state[p+DEATHS];
        out[o+12]=state[own+UNIT_X];out[o+13]=state[own+UNIT_Y];out[o+14]=state[enemy+UNIT_X];out[o+15]=state[enemy+UNIT_Y];
        let dx=state[p+UNIT_X]-state[own+UNIT_X];let dy=state[p+UNIT_Y]-state[own+UNIT_Y];out[o+16]=(dx*dx+dy*dy).sqrt();
        let ex=state[p+UNIT_X]-state[enemy+UNIT_X];let ey=state[p+UNIT_Y]-state[enemy+UNIT_Y];out[o+17]=(ex*ex+ey*ey).sqrt();out[o+18]=reset[a];
        for i in 0..6{out[o+19+i]=actions[a*6+i];}
        for i in 0..11{out[o+25+i]=logits[a*11+i];out[o+36+i]=probabilities[a*11+i];}
        out[o+47]=audit[m*crate::scripted_teacher::AUDIT_FIELDS];let source=state[p+CREDIT]as usize;
        out[o+48]= -1.0f32;out[o+49]= -1.0f32;
        if state[p+CREDIT]>=0.0f32&&source<UNITS{out[o+48]=state[base+source*STRIDE+KIND];out[o+49]=state[base+source*STRIDE+TOWER_TIER];}
    }
}
