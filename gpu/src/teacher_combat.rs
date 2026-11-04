//! Read-only event counters, separate from the observation and v9 audit ABI.
use cubecl::prelude::*;
pub const FIELDS:usize=36;
pub const HERO_DAMAGE:usize=0;
pub const CREEP_DAMAGE:usize=1;
pub const TOWER_DAMAGE:usize=2;
pub const HERO_ATTACKS:usize=3;
pub const AGGRO_ORDERS:usize=4;
pub const RAZE_KILLS:usize=5;
pub const LAST_HERO_DAMAGE:usize=6;
pub const HERO_ORDERS:usize=7;
pub const LAST_SOURCE:usize=10;
pub const DEATH_COUNTS:usize=11; // hero attack, raze, creeps, tower, requiem
pub const DEATH_OWN_DISTANCE:usize=16;
pub const DEATH_ENEMY_DISTANCE:usize=21;
pub const FIRST_SOURCE:usize=26;
pub const FIRST_OWN_DISTANCE:usize=27;
pub const FIRST_ENEMY_DISTANCE:usize=28;
pub const RAZE_HITS:usize=29;
pub const LANE_SPAWNED:usize=32;
pub const HERO_KILLS:usize=33;
pub const OUTCOME_REASON:usize=34;
pub const OUTCOME_RESULT:usize=35;

#[cube]
pub fn finish(out:&mut [f32],state:&mut [f32],m:usize,result:f32){
    use crate::{world::*,contract::*};
    let b=m*UNITS*STRIDE;let mut reason=3.0f32.runtime();
    if state[b+2*STRIDE+UNIT_HEALTH]<=0.0f32 || state[b+4*STRIDE+UNIT_HEALTH]<=0.0f32 {reason=2.0f32;}
    if state[b+DEATHS]+state[b+STRIDE+DEATHS]>0.0f32 {reason=1.0f32;}
    for s in 0..2 {let r=(m*2+s)*FIELDS;out[r+OUTCOME_REASON]=reason;out[r+OUTCOME_RESULT]=result;}
}

#[cube]
pub fn damage(out:&mut [f32],m:usize,target:usize,source_kind:f32,amount:f32,time:f32,spell:usize){
    if target<2 && amount>0.0f32 {
        let r=(m*2+target)*FIELDS;
        let field=if source_kind==0.0f32 {HERO_DAMAGE}else if source_kind==1.0f32 {TOWER_DAMAGE}else{CREEP_DAMAGE};
        out[r+field]+=amount;
        out[r+LAST_SOURCE]=if source_kind==0.0f32 {if spell==0 {1.0f32}else if spell<4 {2.0f32}else{5.0f32}}else if source_kind==1.0f32 {4.0f32}else{3.0f32};
        if source_kind==0.0f32 {out[r+LAST_HERO_DAMAGE]=time+1.0f32;}
    }
}

#[cube]
pub fn death(out:&mut [f32],state:&mut [f32],m:usize,side:usize){
    use crate::{world::*,contract::*};
    let base=m*UNITS*STRIDE;let p=base+side*STRIDE;let r=(m*2+side)*FIELDS;
    let cause=out[r+LAST_SOURCE]as usize;
    let own=base+(2+side*2)*STRIDE;let enemy=base+(4-side*2)*STRIDE;
    let dx=state[p+UNIT_X]-state[own+UNIT_X];let dy=state[p+UNIT_Y]-state[own+UNIT_Y];let od=(dx*dx+dy*dy).sqrt();
    let ex=state[p+UNIT_X]-state[enemy+UNIT_X];let ey=state[p+UNIT_Y]-state[enemy+UNIT_Y];let ed=(ex*ex+ey*ey).sqrt();
    if cause>0 && cause<=5 {out[r+DEATH_COUNTS+cause-1]+=1.0f32;out[r+DEATH_OWN_DISTANCE+cause-1]+=od;out[r+DEATH_ENEMY_DISTANCE+cause-1]+=ed;}
    if out[r+FIRST_SOURCE]==0.0f32 {out[r+FIRST_SOURCE]=cause as f32;out[r+FIRST_OWN_DISTANCE]=od;out[r+FIRST_ENEMY_DISTANCE]=ed;}
    if state[p+CREDIT]==(1-side)as f32 {out[(m*2+1-side)*FIELDS+HERO_KILLS]+=1.0f32;}
}
