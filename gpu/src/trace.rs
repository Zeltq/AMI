use cubecl::prelude::*;
use burn::tensor::Tensor;
use burn_cubecl::tensor::CubeTensor;
use crate::{world::*,contract::*,runtime::{primitive,argument},experience::HORIZON};
pub const UNIT_FIELDS:usize=24;pub const PROJECTILE_FIELDS:usize=6;
pub struct Trace {pub buffer:CubeTensor,pub matches:usize,pub width:usize}
impl Trace {
    pub fn new(world:&World)->Self {
        let matches=world.matches.min(4);let width=META+UNITS*UNIT_FIELDS+SHOTS*PROJECTILE_FIELDS+world.tree_count+2*crate::experience::SIGNALS;
        Self{buffer:primitive(Tensor::<3>::zeros([HORIZON,matches,width],&world.device)),matches,width}
    }
    pub fn record(&self,w:&World,step:usize,signals:Tensor<2>){
        let signals=primitive(signals);
        record::launch(&w.state.client,CubeCount::Static((self.matches*self.width).div_ceil(256) as u32,1,1),CubeDim::new_1d(256),
            argument(&w.state),argument(&w.shots),argument(&w.meta),argument(&w.tree_alive),argument(&signals),argument(&self.buffer),step,self.matches,self.width,w.tree_count);
    }
    pub fn header(&self,w:&World,cycle:u64)->serde_json::Value {
        serde_json::json!({"format":2,"cycle":cycle,"frames":HORIZON,"matches":self.matches,"width":self.width,"decision_ticks":w.engine.decision_ticks,
            "tick_hz":30,"unit_capacity":UNITS,"unit_fields":UNIT_FIELDS,"projectile_capacity":SHOTS,"projectile_fields":PROJECTILE_FIELDS,
            "tree_count":w.tree_count,"signal_fields":crate::experience::SIGNALS,"calibration":w.engine.calibration,"game_build":w.engine.build})
    }
}
#[cube(launch)]
fn record(state:&[f32],shots:&[f32],meta:&[f32],trees:&[f32],signals:&[f32],output:&mut [f32],step:usize,matches:usize,width:usize,tree_count:usize){
    let i=ABSOLUTE_POS;
    if i<matches*width {
        let m=i/width;let f=i%width;let mut value=0.0f32.runtime();
        if f<META {value=meta[m*META+f];}
        else if f<META+UNITS*UNIT_FIELDS {
            let q=f-META;let u=q/UNIT_FIELDS;let field=q%UNIT_FIELDS;let p=(m*UNITS+u)*STRIDE;
            if field==0 {value=state[p+UNIT_X];}if field==1 {value=state[p+UNIT_Y];}if field==2 {value=state[p+UNIT_HEALTH];}
            if field==3 {value=state[p+UNIT_HEALTH_MAX];}if field==4 {value=state[p+FACING];}if field==5 {value=state[p+TEAM];}
            if field==6 {value=state[p+KIND];}if field==7 {value=state[p+ACTIVE];}if field==8 {value=state[p+UNIT_MANA];}
            if field==9 {value=state[p+UNIT_SOULS];}if field==10 {value=state[p+WINDUP];}if field==11 {value=state[p+CAST_END];}
            if field==12 {value=state[p+CAST];}if field==13 {value=state[p+TARGET];}
            if field==14 {value=state[p+UNIT_LEVEL];}if field==15 {value=state[p+UNIT_LAST_HITS];}if field==16 {value=state[p+UNIT_DENIES];}
            if field==17 {value=state[p+UNIT_GOLD];}if field==18 {value=state[p+ORDER];}if field==19 {value=state[p+ATTACK_READY];}if field==20 {value=state[p+UNIT_XP_TO_NEXT];}
            if field==21 {value=state[p+FLAGBEARER];}
            if field==22 {value=state[p+DEATHS];}if field==23 {value=state[p+RESPAWN_AT];}
        } else if f<META+UNITS*UNIT_FIELDS+SHOTS*PROJECTILE_FIELDS {
            let q=f-META-UNITS*UNIT_FIELDS;let s=q/PROJECTILE_FIELDS;let field=q%PROJECTILE_FIELDS;let p=(m*SHOTS+s)*SHOT_FIELDS;
            if field<4 {value=shots[p+field];}if field==4 {value=shots[p+7];}if field==5 {value=shots[p+9];}
        } else if f<META+UNITS*UNIT_FIELDS+SHOTS*PROJECTILE_FIELDS+tree_count {value=trees[m*tree_count+f-META-UNITS*UNIT_FIELDS-SHOTS*PROJECTILE_FIELDS];}
        else {value=signals[m*2*crate::experience::SIGNALS+f-META-UNITS*UNIT_FIELDS-SHOTS*PROJECTILE_FIELDS-tree_count];}
        output[(step*matches+m)*width+f]=value;
    }
}
