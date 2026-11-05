//! Read-only CHiO event accounting. These buffers never affect physics/actions.
use cubecl::prelude::*;
use crate::{world::*,contract::*};
pub const GOLD:usize=0;
pub const FLASK:usize=1;
pub const MANGO:usize=2;
pub const TANGO:usize=3;
pub const SPENT:usize=4;
pub const BREAKS:usize=5;
pub const CANCELS:usize=6;
pub const KILLS:usize=7;
pub const PURCHASES:usize=8;
pub const BREAK_OFFSET:usize=16;
pub const BREAK_SLOTS:usize=8;
pub const BREAK_FIELDS:usize=6; // hp before, max HP, hp after, remaining seconds, time, source kind
pub const FIELDS:usize=BREAK_OFFSET+BREAK_SLOTS*BREAK_FIELDS;

#[cube]
pub fn flask_break(state:&mut [f32],events:&mut [f32],m:usize,side:usize,damage:f32,time:f32,source_kind:f32){
    let p=(m*UNITS+side)*STRIDE;let r=(m*2+side)*FIELDS;
    let before=(state[p+UNIT_HEALTH]-state[p+DAMAGE]).max(0.0f32);let after=(before-damage).max(0.0f32);
    if state[p+FLASK_END]>time && damage>0.0f32 && after<state[p+UNIT_HEALTH_MAX] {
        let slot=events[r+BREAKS]as usize%BREAK_SLOTS;let at=r+BREAK_OFFSET+slot*BREAK_FIELDS;
        events[at]=before;events[at+1]=state[p+UNIT_HEALTH_MAX];events[at+2]=after;
        events[at+3]=state[p+FLASK_END]-time;events[at+4]=time;events[at+5]=source_kind;
        events[r+BREAKS]+=1.0f32;
    }
}
