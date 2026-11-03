//! Read-only event counters, separate from the observation and v9 audit ABI.
use cubecl::prelude::*;
pub const FIELDS:usize=10;
pub const HERO_DAMAGE:usize=0;
pub const CREEP_DAMAGE:usize=1;
pub const TOWER_DAMAGE:usize=2;
pub const HERO_ATTACKS:usize=3;
pub const AGGRO_ORDERS:usize=4;
pub const RAZE_KILLS:usize=5;
pub const LAST_HERO_DAMAGE:usize=6;
pub const HERO_ORDERS:usize=7;

#[cube]
pub fn damage(out:&mut [f32],m:usize,target:usize,source_kind:f32,amount:f32,time:f32){
    if target<2 && amount>0.0f32 {
        let r=(m*2+target)*FIELDS;
        let field=if source_kind==0.0f32 {HERO_DAMAGE}else if source_kind==1.0f32 {TOWER_DAMAGE}else{CREEP_DAMAGE};
        out[r+field]+=amount;
        if source_kind==0.0f32 {out[r+LAST_HERO_DAMAGE]=time+1.0f32;}
    }
}
