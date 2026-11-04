//! Privileged reward metadata, deliberately separate from policy observations.
use crate::{world::*,contract::*,runtime::{primitive,tensor,argument},teacher_combat as combat};
use burn::tensor::Tensor;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use anyhow::Result;
use std::{path::Path,io::Write};
pub const FIELDS:usize=34;
pub const NAMES:[&str;FIELDS]=["game_seconds","own_lh","own_deny","own_hp_fraction","own_mana_fraction","own_level","own_xp_in_level","own_xp_total","own_souls","own_raze_stacks","own_deaths","enemy_lh","enemy_deny","enemy_hp_fraction","enemy_mana_fraction","enemy_level","enemy_xp_in_level","enemy_xp_total","enemy_souls","enemy_raze_stacks","enemy_deaths","own_t1_hp_fraction","enemy_t1_hp_fraction","terminal","result","reason","own_hp","own_hp_max","own_mana","own_mana_max","enemy_hp","enemy_hp_max","enemy_mana","enemy_mana_max"];

#[cube]
pub fn reason(state:&[f32],meta:&[f32],m:usize)->f32{
    let b=m*UNITS*STRIDE;let mut value=0.0f32.runtime();
    if meta[m*META+2]>0.0f32 {
        value=3.0f32;
        if state[b+2*STRIDE+UNIT_HEALTH]<=0.0f32 || state[b+4*STRIDE+UNIT_HEALTH]<=0.0f32 {value=2.0f32;}
        if state[b+DEATHS]+state[b+STRIDE+DEATHS]>0.0f32 {value=1.0f32;}
    }value
}

#[cube(launch)]
fn record(state:&[f32],meta:&[f32],xp:&[f32],out:&mut [f32],actors:usize,step:usize,phase:usize,pregame:f32){
    let a=ABSOLUTE_POS;if a<actors {
        let m=a/2;let side=a%2;let base=m*UNITS*STRIDE;let r=(a*128+step)*FIELDS*2+phase*FIELDS;
        out[r]=meta[m*META]/30.0f32;
        for j in 0..2 {let p=base+if j==0 {side}else{1-side}*STRIDE;let at=r+1+j*10;
            out[at]=state[p+UNIT_LAST_HITS];out[at+1]=state[p+UNIT_DENIES];
            out[at+2]=state[p+UNIT_HEALTH]/state[p+UNIT_HEALTH_MAX].max(1.0f32);out[at+3]=state[p+UNIT_MANA]/state[p+UNIT_MANA_MAX].max(1.0f32);
            let exact=r+26+j*4;
            out[exact]=state[p+UNIT_HEALTH];out[exact+1]=state[p+UNIT_HEALTH_MAX];
            out[exact+2]=state[p+UNIT_MANA];out[exact+3]=state[p+UNIT_MANA_MAX];
            out[at+4]=state[p+UNIT_LEVEL];out[at+5]=state[p+XP];let mut total=state[p+XP];
            for l in 1..30 {if (l as f32)<state[p+UNIT_LEVEL] {total+=xp[l-1];}}out[at+6]=total;
            out[at+7]=state[p+UNIT_SOULS];out[at+8]=if state[p+STACK_END]>((meta[m*META]+pregame*30.0f32)/30.0f32).max(0.0f32) {state[p+UNIT_RAZE_STACKS]}else{0.0f32};out[at+9]=state[p+DEATHS];
            let t=base+(if j==0 {2+side*2}else{4-side*2})*STRIDE;
            out[r+21+j]=state[t+UNIT_HEALTH]/state[t+UNIT_HEALTH_MAX].max(1.0f32);
        }
        out[r+23]=meta[m*META+2];out[r+24]=meta[m*META+17]*if side==0 {1.0f32}else{-1.0f32};out[r+25]=reason(state,meta,m);
    }
}

pub struct Recorder {buffer:CubeTensor,actors:usize}
impl Recorder {
    pub fn new(w:&World)->Self{Self{buffer:primitive(Tensor::<3>::zeros([w.matches*2,128,FIELDS*2],&w.device)),actors:w.matches*2}}
    pub fn record(&self,w:&World,step:usize,phase:usize){record::launch(&w.state.client,CubeCount::Static(self.actors.div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&w.state),argument(&w.meta),argument(&w.experience),argument(&self.buffer),self.actors,step,phase,w.engine.pre_game_seconds);}
    pub fn flush(&self,root:&Path,block:usize,offset:usize,frames:usize)->Result<()>{
        let values=tensor::<3>(self.buffer.clone()).into_data().try_to_vec::<f32>()?;
        let header=serde_json::to_vec(&serde_json::json!({"shape":[self.actors,128,2,FIELDS],"actors":(offset*2..offset*2+self.actors).collect::<Vec<_>>(),"frames":frames,"fields":NAMES.as_slice(),"phases":["before","after"],"reason_codes":{"0":"ongoing","1":"death","2":"T1","3":"LH600"}}))?;
        let mut out=flate2::write::GzEncoder::new(std::fs::File::create(root.join(format!("telemetry-{block:06}.f32.gz")))?,flate2::Compression::fast());
        out.write_all(&(header.len()as u32).to_le_bytes())?;out.write_all(&header)?;
        for value in values {out.write_all(&value.to_le_bytes())?;}out.finish()?;Ok(())
    }
}

pub fn enrich(w:&World,games:&mut [serde_json::Value])->Result<()>{
    let events=tensor::<1>(w.teacher_combat.clone()).into_data().try_to_vec::<f32>()?;
    for (m,g) in games.iter_mut().enumerate(){
        // observe() resets completed arenas before the cohort finishes.
        // Read the persistent event snapshot, never the final live state.
        let r=m*2*combat::FIELDS;
        g["result"]=serde_json::json!(events[r+combat::OUTCOME_RESULT]);
        g["outcome_reason"]=serde_json::json!((["ongoing","death","T1","LH600"][events[r+combat::OUTCOME_REASON]as usize]));
        g["lane_creeps_spawned"]=serde_json::json!(events[m*2*combat::FIELDS+combat::LANE_SPAWNED]+events[(m*2+1)*combat::FIELDS+combat::LANE_SPAWNED]);
        for s in 0..2 {let r=(m*2+s)*combat::FIELDS;let row=&mut g["sides"][s];
            for (key,f) in [("damage_from_hero",combat::HERO_DAMAGE),("damage_from_creeps",combat::CREEP_DAMAGE),("damage_from_tower",combat::TOWER_DAMAGE),("hero_attacks_released",combat::HERO_ATTACKS),("hero_kills",combat::HERO_KILLS)] {row[key]=serde_json::json!(events[r+f]);}
            row["raze_hero_hits_by_spell"]=serde_json::json!(&events[r+combat::RAZE_HITS..r+combat::RAZE_HITS+3]);
            row["death_causes"]=serde_json::json!((0..5).map(|i|serde_json::json!({"cause":(["hero_attack","raze","creeps","tower","requiem"][i]),"count":events[r+combat::DEATH_COUNTS+i],"own_t1_distance_sum":events[r+combat::DEATH_OWN_DISTANCE+i],"enemy_t1_distance_sum":events[r+combat::DEATH_ENEMY_DISTANCE+i]})).collect::<Vec<_>>());
            row["first_death_location"]=serde_json::json!({"cause_code":events[r+combat::FIRST_SOURCE],"own_t1_distance":events[r+combat::FIRST_OWN_DISTANCE],"enemy_t1_distance":events[r+combat::FIRST_ENEMY_DISTANCE]});
        }
    }Ok(())
}
