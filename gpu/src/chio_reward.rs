//! lastorder-fb-v1: exact-state transition reward and replayable critic metadata.
use burn::tensor::Tensor;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use crate::{world::*,contract::*,chio_events as event,runtime::{argument,primitive,tensor}};
use anyhow::Result;
use std::{path::Path,io::Write};

pub const GAMMA:f32=0.997;pub const LAMBDA:f32=0.95;
pub const SNAP:usize=85;pub const COMPONENTS:usize=14;
pub const RECORD:usize=2*SNAP+2*COMPONENTS+2;
pub const NAMES:[&str;COMPONENTS]=["experience","gold_received","mana","health","last_hits","denies","kills","deaths","own_t1","flask","mango","tango","flask_break","outcome"];
pub fn fields()->Vec<String>{
    let mut names:Vec<_>=["game_seconds","gold_received","gold_current","xp_total","hp","hp_max","mana","mana_max","own_t1_hp","own_t1_max","enemy_t1_hp","enemy_t1_max",
        "last_hits","denies","kills","deaths","flask_used","mango_used","tango_used","flask_remaining_seconds","cast_skill","terminal","result","reason","gold_spent","flask_break_count","cast_cancellations","cast_active"].into_iter().map(str::to_string).collect();
    for slot in 0..8 {for field in ["hp_before","hp_max","hp_after","remaining_seconds","episode_seconds","source_kind"] {names.push(format!("flask_break_{slot}_{field}"));}}
    for item in 0..8 {names.push(format!("purchases_{item}"));}names.push("flask_heal_per_second".into());names
}

#[cube(launch)]
fn snapshot_kernel(state:&[f32],meta:&[f32],events:&[f32],xp:&[f32],out:&mut [f32],actors:usize,pregame:f32,flask_rate:f32){
    let a=ABSOLUTE_POS;if a<actors {
        let m=a/2;let side=a%2;let base=m*UNITS*STRIDE;let p=base+side*STRIDE;let r=a*SNAP;let e=a*event::FIELDS;
        let time=(meta[m*META]+pregame*30.0f32)/30.0f32;
        out[r]=meta[m*META]/30.0f32;out[r+1]=events[e+event::GOLD];out[r+2]=state[p+UNIT_GOLD];
        let mut total=state[p+XP];for level in 1..30 {if (level as f32)<state[p+UNIT_LEVEL]{total+=xp[level-1];}}out[r+3]=total;
        out[r+4]=state[p+UNIT_HEALTH];out[r+5]=state[p+UNIT_HEALTH_MAX];out[r+6]=state[p+UNIT_MANA];out[r+7]=state[p+UNIT_MANA_MAX];
        let own=base+(2+side*2)*STRIDE;let enemy=base+(4-side*2)*STRIDE;
        out[r+8]=state[own+UNIT_HEALTH];out[r+9]=state[own+UNIT_HEALTH_MAX];out[r+10]=state[enemy+UNIT_HEALTH];out[r+11]=state[enemy+UNIT_HEALTH_MAX];
        out[r+12]=state[p+UNIT_LAST_HITS];out[r+13]=state[p+UNIT_DENIES];out[r+14]=events[e+event::KILLS];out[r+15]=state[p+DEATHS];
        out[r+16]=events[e+event::FLASK];out[r+17]=events[e+event::MANGO];out[r+18]=events[e+event::TANGO];
        out[r+19]=(state[p+FLASK_END]-time).max(0.0f32);out[r+20]=if state[p+CAST_END]>time {state[p+CAST]}else{0.0f32};
        out[r+21]=meta[m*META+2];out[r+22]=meta[m*META+17]*if side==0 {1.0f32}else{-1.0f32};out[r+23]=crate::demo_telemetry::reason(state,meta,m);
        out[r+24]=events[e+event::SPENT];out[r+25]=events[e+event::BREAKS];out[r+26]=events[e+event::CANCELS];out[r+27]=if state[p+CAST_END]>time {1.0f32}else{0.0f32};
        for i in 0..48 {out[r+28+i]=events[e+event::BREAK_OFFSET+i];}
        for i in 0..8 {out[r+76+i]=events[e+event::PURCHASES+i];}out[r+84]=flask_rate;
    }
}

pub fn snapshot(w:&World)->CubeTensor {
    let n=w.matches*2;let out=primitive(Tensor::<2>::zeros([n,SNAP],&w.device));
    snapshot_kernel::launch(&w.state.client,CubeCount::Static(n.div_ceil(64)as u32,1,1),CubeDim::new_1d(64),
        argument(&w.state),argument(&w.meta),argument(&w.chio_events),argument(&w.experience),argument(&out),n,w.engine.pre_game_seconds,w.engine.items[5][4]);out
}

#[cube]
fn fraction(value:f32,maximum:f32)->f32 {(value/maximum.max(1.0f32)).clamp(0.0f32,1.0f32)}
#[cube]
fn health(h:f32)->f32 {let gap=1.0f32-h;h+1.0f32-gap*gap*gap*gap}
#[cube]
fn tower(h:f32)->f32 {let gap=1.0f32-h;h+1.0f32-gap*gap*gap}
#[cube]
fn component(before:&[f32],after:&[f32],a:usize,c:usize)->f32 {
    let r=a*SNAP;let mut value=0.0f32.runtime();
    if c==0 {value=0.002f32*(after[r+3]-before[r+3]);}
    if c==1 {value=0.006f32*(after[r+1]-before[r+1]).max(0.0f32);}
    if c==2 {value=0.5f32*(fraction(after[r+6],after[r+7])-fraction(before[r+6],before[r+7]));}
    if c==3 {value=health(fraction(after[r+4],after[r+5]))-health(fraction(before[r+4],before[r+5]));}
    if c==4 {value=-0.12f32*(after[r+12]-before[r+12]);}
    if c==5 {value=0.1f32*(after[r+13]-before[r+13]);}
    if c==6 {value=1.5f32*(after[r+14]-before[r+14]);}
    if c==7 {value=-3.0f32*(after[r+15]-before[r+15]);}
    if c==8 {value=1.5f32*(tower(fraction(after[r+8],after[r+9]))-tower(fraction(before[r+8],before[r+9])));}
    if c>=9 && c<=11 {value=(if c==9{-0.3f32}else{-0.1f32})*(after[r+16+c-9]-before[r+16+c-9]);}
    if c==12 {
        let first=before[r+25]as usize;let end=after[r+25]as usize;
        for offset in 0..event::BREAK_SLOTS {if first+offset<end {
            let at=r+28+((first+offset)%event::BREAK_SLOTS)*event::BREAK_FIELDS;
            let projected=fraction(after[at]+after[r+84]*after[at+3],after[at+1]);
            value+=(health(fraction(after[at+2],after[at+1]))-health(projected)).max(-0.4f32);
        }}
    }
    if after[r]>=360.0f32 && c!=6 && c!=7 && c!=13 {value*=0.6f32;}
    if c==13 && before[r+21]<=0.0f32 && after[r+21]>0.0f32 {value=5.0f32*after[r+22];}
    value
}

#[cube(launch)]
fn reward_kernel(before:&[f32],after:&[f32],out:&mut [f32],actors:usize,#[comptime] zero_sum:bool){
    let a=ABSOLUTE_POS;if a<actors {
        let r=a*(2*COMPONENTS+1);let enemy=if a%2==0{a+1}else{a-1};let mut total=0.0f32.runtime();
        for c in 0..COMPONENTS {
            let own=component(before,after,a,c);let opponent=if zero_sum && c!=6 && c!=7 && c!=13 {-component(before,after,enemy,c)}else{0.0f32};
            out[r+c]=own;out[r+COMPONENTS+c]=opponent;total+=own+opponent;
        }out[r+2*COMPONENTS]=total;
    }
}
pub struct Reward {pub components:Tensor<2>,pub total:Tensor<1>}
pub fn transition(before:&CubeTensor,after:&CubeTensor,zero_sum:bool,d:&burn::tensor::Device)->Reward {
    let n=before.meta.shape()[0];assert_eq!(before.meta.shape(),after.meta.shape());assert_eq!(n%2,0);
    let out=primitive(Tensor::<2>::zeros([n,2*COMPONENTS+1],d));
    reward_kernel::launch(&out.client,CubeCount::Static(n.div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(before),argument(after),argument(&out),n,zero_sum);
    let out=tensor::<2>(out);Reward{components:out.clone().slice([0..n,0..2*COMPONENTS]),total:out.slice([0..n,2*COMPONENTS..2*COMPONENTS+1]).squeeze_dim(1)}
}

#[cube(launch)]
fn record_kernel(before:&[f32],after:&[f32],components:&[f32],total:&[f32],valid:&[f32],out:&mut [f32],actors:usize,step:usize){
    let i=ABSOLUTE_POS;if i<actors*RECORD {
        let a=i/RECORD;let f=i%RECORD;let at=(a*128+step)*RECORD+f;
        let value=if f<SNAP {before[a*SNAP+f]}else if f<2*SNAP {after[a*SNAP+f-SNAP]}else if f<2*SNAP+2*COMPONENTS {components[a*2*COMPONENTS+f-2*SNAP]}else if f==RECORD-2 {total[a]}else{valid[a]};out[at]=value;
    }
}
pub struct Recorder {buffer:CubeTensor,actors:usize,device:burn::tensor::Device,zero_sum:bool}
impl Recorder {
    pub fn new(w:&World,zero_sum:bool)->Self {Self{buffer:primitive(Tensor::<3>::zeros([w.matches*2,128,RECORD],&w.device)),actors:w.matches*2,device:w.device.clone(),zero_sum}}
    pub fn store(&self,before:&CubeTensor,after:&CubeTensor,reward:&Reward,valid:Tensor<1>,step:usize){
        let components=primitive(reward.components.clone());let total=primitive(reward.total.clone());let valid=primitive(valid);
        record_kernel::launch(&self.buffer.client,CubeCount::Static((self.actors*RECORD).div_ceil(256)as u32,1,1),CubeDim::new_1d(256),
            argument(before),argument(after),argument(&components),argument(&total),argument(&valid),argument(&self.buffer),self.actors,step);
    }
    pub fn flush(&mut self,root:&Path,block:usize,offset:usize,frames:usize)->Result<()> {
        let values=tensor::<3>(self.buffer.clone()).into_data().try_to_vec::<f32>()?;
        let header=serde_json::to_vec(&serde_json::json!({"format":"chio-reward-v2","shape":[self.actors,128,RECORD],"actors":(offset*2..offset*2+self.actors).collect::<Vec<_>>(),
            "frames":frames,"snapshot_fields":fields(),"components":NAMES,"zero_sum":self.zero_sum,"gamma":GAMMA,"gae_lambda":LAMBDA,
            "layout":"before snapshot, after snapshot, own components, subtracted enemy components, total, valid; opposite hero is adjacent actor id xor 1"}))?;
        let mut out=flate2::write::GzEncoder::new(std::fs::File::create(root.join(format!("chio-reward-{block:06}.f32.gz")))?,flate2::Compression::fast());
        out.write_all(&(header.len()as u32).to_le_bytes())?;out.write_all(&header)?;for value in values {out.write_all(&value.to_le_bytes())?;}out.finish()?;
        self.buffer=primitive(Tensor::<3>::zeros([self.actors,128,RECORD],&self.device));Ok(())
    }
}
