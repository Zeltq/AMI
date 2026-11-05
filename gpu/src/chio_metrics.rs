//! Per-game CHiO accounting, frozen on the final valid decision.
use burn::tensor::Tensor;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use crate::{world::*,contract::*,chio_reward as reward,runtime::{argument,primitive,tensor}};
const FIELDS:usize=8+28+reward::SNAP;
pub struct Metrics {pub buffer:CubeTensor}
impl Metrics {
    pub fn new(w:&World)->Self {
        let buffer=primitive(Tensor::<2>::zeros([w.matches*2,FIELDS],&w.device));
        initialize::launch(&buffer.client,CubeCount::Static((w.matches*2).div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&w.state),argument(&w.meta),argument(&buffer),w.matches*2);
        Self{buffer}
    }
    pub fn sample(&self,w:&World,snapshot:&CubeTensor,r:&reward::Reward,valid:Tensor<1>){
        let parts=primitive(r.components.clone());let total=primitive(r.total.clone());let valid=primitive(valid);
        sample::launch(&self.buffer.client,CubeCount::Static((w.matches*2).div_ceil(64)as u32,1,1),CubeDim::new_1d(64),
            argument(&w.state),argument(&w.meta),argument(snapshot),argument(&parts),argument(&total),argument(&valid),argument(&self.buffer),w.matches*2,w.engine.pre_game_seconds);
    }
    pub fn enrich(&self,games:&mut [serde_json::Value])->anyhow::Result<()> {
        let data=tensor::<2>(self.buffer.clone()).into_data().try_to_vec::<f32>()?;
        for (m,game) in games.iter_mut().enumerate(){for side in 0..2 {
            let row=&mut game["sides"][side];let a=(m*2+side)*FIELDS;let snap=&data[a+36..a+FIELDS];
            row["chio_decisions"]=serde_json::json!(data[a]);row["stuck_decisions"]=serde_json::json!(data[a+1]);
            row["stuck_fraction"]=serde_json::json!({"numerator":data[a+1],"denominator":data[a]});row["longest_stuck_seconds"]=serde_json::json!(data[a+6]);
            row["reward_total"]=serde_json::json!(data[a+7]);row["cancelled_casts"]=serde_json::json!(snap[26]);
            row["gold_received"]=serde_json::json!(snap[1]);row["gold_spent"]=serde_json::json!(snap[24]);
            row["consumable_uses"]=serde_json::json!({"flask":snap[16],"mango":snap[17],"tango":snap[18]});
            let mut purchases=serde_json::Map::new();for (i,name) in ["wraith_band","boots","bracer","tp","mango","flask","tango","clarity"].iter().enumerate(){purchases.insert(name.to_string(),serde_json::json!(snap[76+i]));}
            row["purchases"]=purchases.into();let mut components=serde_json::Map::new();
            for i in 0..14 {components.insert(format!("own_{}",reward::NAMES[i]),serde_json::json!(data[a+8+i]));components.insert(format!("enemy_{}",reward::NAMES[i]),serde_json::json!(data[a+22+i]));}
            row["reward_components"]=components.into();
        }}Ok(())
    }
}
#[cube(launch)]
fn initialize(state:&[f32],meta:&[f32],out:&mut [f32],actors:usize){let a=ABSOLUTE_POS;if a<actors {let p=(a/2*UNITS+a%2)*STRIDE;let r=a*FIELDS;out[r+3]=state[p+UNIT_X];out[r+4]=state[p+UNIT_Y];out[r+5]=meta[a/2*META]/30.0f32;}}
#[cube(launch)]
fn sample(state:&[f32],meta:&[f32],snapshot:&[f32],parts:&[f32],total:&[f32],valid:&[f32],out:&mut [f32],actors:usize,pregame:f32){
    let a=ABSOLUTE_POS;if a<actors && valid[a]>0.0f32 {
        let p=(a/2*UNITS+a%2)*STRIDE;let r=a*FIELDS;let time=meta[a/2*META]/30.0f32;let dt=time-out[r+5];
        let dx=state[p+UNIT_X]-out[r+3];let dy=state[p+UNIT_Y]-out[r+4];
        let gx=state[p+DEST_X]-state[p+UNIT_X];let gy=state[p+DEST_Y]-state[p+UNIT_Y];
        let active=state[p+UNIT_HEALTH]>0.0f32 && (state[p+ORDER]==1.0f32 || state[p+ORDER]==2.0f32) && gx*gx+gy*gy>75.0f32*75.0f32 && state[p+CAST_END]<=time+pregame && state[p+FEAR_END]<=time+pregame;
        if active && dt>0.0f32 && (dx*dx+dy*dy).sqrt()<0.5f32*state[p+UNIT_MOVE_SPEED]*dt {out[r+2]+=dt;}else{out[r+2]=0.0f32;}
        out[r]+=1.0f32;if out[r+2]>1.0001f32 {out[r+1]+=1.0f32;}out[r+6]=out[r+6].max(out[r+2]);
        out[r+3]=state[p+UNIT_X];out[r+4]=state[p+UNIT_Y];out[r+5]=time;out[r+7]+=total[a];
        for i in 0..28 {out[r+8+i]+=parts[a*28+i];}for i in 0..reward::SNAP {out[r+36+i]=snapshot[a*reward::SNAP+i];}
    }
}
