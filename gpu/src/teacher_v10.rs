//! Minimal aggression layered over the unmodified v9 decision and DAgger memory.
use anyhow::{Result,ensure};
use burn::tensor::{Tensor,Int};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use serde::{Serialize,Deserialize};
use crate::{contract::*,world::*,observe::{self,Observed},runtime::{argument,primitive,tensor},scripted_teacher::Teacher};

pub const FARM_RAZE_MARGIN:f32=0.25;
pub const HERO_RAZE_MARGIN:f32=0.35;
pub const MANA_RESERVE_RAZES:f32=1.0;
pub const TRADE_WINDOW:f32=2.0;
pub const TRADE_HP_GAP:f32=0.15;
pub const RETREAT_BUFFER:f32=100.0;
pub const TOWER_BUFFER:f32=100.0;
pub const AGGRO_RADIUS:f32=500.0;
pub const EXECUTION_DELAY:f32=7.0/30.0;
pub const TURN_STEP_FACTOR:f32=0.4712389;
pub const AIM_STEP:f32=64.0;
pub const V9_HOME_OFFSET:f32=220.0;
pub const V9_HOME_TOLERANCE:f32=90.0;
#[allow(dead_code)]
pub const V9_RETREAT_HP:f32=0.30; // v9 owns this unchanged hysteresis.
#[allow(dead_code)]
pub const V9_RETURN_HP:f32=0.56;
const MOTION_FIELDS:usize=4;
const MOTION_WORDS:usize=2+UNITS*MOTION_FIELDS;

#[derive(Clone,Copy,Debug,Default,Serialize,Deserialize,clap::ValueEnum,PartialEq,Eq)]
pub enum Version {#[default] #[serde(rename="v9")] V9,#[serde(rename="v10")] V10}
impl Version {
    pub fn sha256(self)->String {crate::runtime::bytes_sha256(match self {Self::V9=>include_bytes!("scripted_teacher.rs"),Self::V10=>include_bytes!("teacher_v10.rs")})}
    fn number(self)->usize {if self==Self::V9 {9}else{10}}
}

#[cube(launch)]
fn motion(state:&[f32],meta:&[f32],out:&mut [f32],matches:usize){
    let m=ABSOLUTE_POS;if m<matches {
        let r=m*MOTION_WORDS;let tick=meta[m*META];let dt=(tick-out[r+1])/30.0f32;
        let valid=out[r]==meta[m*META+1]+1.0f32 && dt>0.0f32;
        for u in 0..UNITS {let p=(m*UNITS+u)*STRIDE;let h=r+2+u*MOTION_FIELDS;
            if valid {out[h+2]=(state[p+UNIT_X]-out[h])/dt;out[h+3]=(state[p+UNIT_Y]-out[h+1])/dt;}
            else {out[h+2]=0.0f32;out[h+3]=0.0f32;}
            out[h]=state[p+UNIT_X];out[h+1]=state[p+UNIT_Y];
        }
        out[r]=meta[m*META+1]+1.0f32;out[r+1]=tick;
    }
}

#[cube(launch)]
fn overlay(state:&mut [f32],meta:&[f32],hero:&[f32],spells:&[f32],commands:&[f32],
    combat:&[f32],motion:&[f32],before:&[f32],memory:&[f32],forecasts:&[f32],audit:&[f32],
    obs:&[f32],actions:&mut [f32],actors:usize,pregame:f32,left:usize,right:usize,#[comptime] swap:bool){
    let a=ABSOLUTE_POS;
    if a<actors {
        let m=a/2;let side=a%2;let selected=if swap && m%2==1 {1-side}else{side};
        let version=if selected==0 {left}else{right};
        let base=m*UNITS*STRIDE;let p=base+side*STRIDE;let enemy=base+(1-side)*STRIDE;
        let time=(meta[m*META]+pregame*30.0f32)/30.0f32;let row=a*observe::WIDTH;let mem=a*16;
        if version==10 && audit[m*crate::scripted_teacher::AUDIT_FIELDS]==0.0f32 && state[p+ACTIVE]>0.0f32 && obs[row+observe::ACTION_MASK+1]>0.0f32 {
            let hp=state[p+UNIT_HEALTH]/state[p+UNIT_HEALTH_MAX];
            let enemy_hp=state[enemy+UNIT_HEALTH]/state[enemy+UNIT_HEALTH_MAX];
            let damage_stamp=combat[a*crate::teacher_combat::FIELDS+crate::teacher_combat::LAST_HERO_DAMAGE];
            let trade=damage_stamp>0.0f32 && time-(damage_stamp-1.0f32)<=TRADE_WINDOW && enemy_hp-hp>TRADE_HP_GAP;
            let mut escape=false;let mut gx=state[p+UNIT_X];let mut gy=state[p+UNIT_Y];
            // All enemy towers, including the background map, use their actual range.
            for u in 2..UNITS {let q=base+u*STRIDE;
                if state[q+KIND]==1.0f32 && state[q+TEAM]!=side as f32 && state[q+ACTIVE]>0.0f32 {
                    let dist=distance(state,p,q);let radius=crate::physics::attack_reach(state,q,p);
                    let mut cover=false;
                    for v in 6..CREEP_END {let c=base+v*STRIDE;
                        if state[c+ACTIVE]>0.0f32 && state[c+UNIT_HEALTH]>0.0f32 && state[c+TEAM]==side as f32 && state[c+KIND]>=2.0f32 && distance(state,c,q)<state[q+UNIT_ATTACK_RANGE] {cover=true;}
                    }
                    let targeting=state[q+TARGET]==side as f32 || (state[q+WINDUP]>time && state[q+WINDUP_TARGET]==side as f32);
                    if !escape && dist<radius+TOWER_BUFFER && (targeting || !cover) {
                        let dx=state[p+UNIT_X]-state[q+UNIT_X];let dy=state[p+UNIT_Y]-state[q+UNIT_Y];
                        gx=state[q+UNIT_X]+dx/dist.max(1.0f32)*(radius+TOWER_BUFFER);gy=state[q+UNIT_Y]+dy/dist.max(1.0f32)*(radius+TOWER_BUFFER);escape=true;
                    }
                }
            }
            if !escape && trade {
                let dist=distance(state,p,enemy);let reach=crate::physics::attack_reach(state,enemy,p)+RETREAT_BUFFER;
                if dist<reach || (state[p+ORDER]==3.0f32 && state[p+TARGET]==(1-side)as f32) {
                    gx=state[enemy+UNIT_X]+(state[p+UNIT_X]-state[enemy+UNIT_X])/dist.max(1.0f32)*reach;
                    gy=state[enemy+UNIT_Y]+(state[p+UNIT_Y]-state[enemy+UNIT_Y])/dist.max(1.0f32)*reach;escape=true;
                }
            }
            let mut action=actions[a*6]as usize;let mut target=actions[a*6+3];let mut change=false;
            if escape {action=1;change=true;}
            else if memory[mem+15]>0.0f32 && action!=10 &&
                ((state[p+ORDER]==3.0f32 && state[p+TARGET]==(1-side)as f32) ||
                 (before[mem]==memory[mem] && before[mem+1]==(1-side)as f32 && time<before[mem+3])) {
                // v9 protects useful farming swings down to 20% HP. A hero swing
                // is not a last hit/deny: do not let that inherited lock postpone
                // the unchanged 30% retreat and 56% return hysteresis.
                let tower=base+(2+side*2)*STRIDE;let sign=if side==0 {-1.0f32}else{1.0f32};
                gx=state[tower+UNIT_X]+sign*V9_HOME_OFFSET;gy=state[tower+UNIT_Y]+sign*V9_HOME_OFFSET;
                let dx=gx-state[p+UNIT_X];let dy=gy-state[p+UNIT_Y];
                action=if dx*dx+dy*dy>V9_HOME_TOLERANCE*V9_HOME_TOLERANCE {1usize}else{4usize};change=true;
            }
            else if memory[mem+15]==0.0f32 && !trade && action!=10 {
                let mut protected=false;
                let current=before[mem+1]as usize;
                if before[mem]==memory[mem] && before[mem+1]>=6.0f32 && current<CREEP_END {
                    let q=base+current*STRIDE;
                    protected=state[q+ACTIVE]>0.0f32 && state[q+GENERATION]==before[mem+2] && time<before[mem+3];
                }
                let winding=state[p+WINDUP_TARGET]as usize;
                if state[p+WINDUP]>time && winding>=6 && winding<CREEP_END {
                    let f=(a*UNITS+winding)*10;
                    if forecasts[f+1]==time && forecasts[f+3]>0.0f32 && forecasts[f+3]<=forecasts[f+4] {protected=true;}
                }
                let mut pending_cast=state[p+CAST_END]>time;
                for slot in 0..4 {let k=(a*4+slot)*COMMAND_FIELDS;
                    if commands[k+7]>0.0f32 && commands[k+1]>=6.0f32 && commands[k+1]<=8.0f32 {pending_cast=true;}
                }
                if pending_cast {action=0;change=true;}
                else if !protected && obs[row+observe::HERO_VALID+1]>0.0f32 && state[enemy+ACTIVE]>0.0f32 {
                    let mh=m*MOTION_WORDS+2;let eh=mh+(1-side)*MOTION_FIELDS;let ph=mh+side*MOTION_FIELDS;
                    let cycle=state[p+UNIT_ATTACK_INTERVAL];let mut farm_soon=false;let mut aggro=false;
                    for u in 6..CREEP_END {let q=base+u*STRIDE;let f=(a*UNITS+u)*10;
                        if state[q+ACTIVE]>0.0f32 && forecasts[f+1]==time && forecasts[f]==state[q+GENERATION] {
                            let delay=(forecasts[f+2]-time).max(1.0f32/30.0f32);
                            let dps=((state[q+UNIT_HEALTH]-forecasts[f+3])/delay).max(0.0f32);
                            let window=(state[q+UNIT_HEALTH]-forecasts[f+4])/dps.max(0.001f32);
                            if window<=cycle+EXECUTION_DELAY || forecasts[f+7]-time<=cycle+EXECUTION_DELAY {farm_soon=true;}
                        }
                    }
                    // Same predicate as physics, with a conservative movement envelope
                    // across the delayed execution. Cooldown is checked at execution time.
                    for u in 2..UNITS {let q=base+u*STRIDE;
                        if state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>0.0f32 && state[q+TEAM]!=side as f32 && state[q+KIND]>=2.0f32 && state[q+AGGRO_READY]<=time+EXECUTION_DELAY {
                            if distance(state,p,q)<AGGRO_RADIUS+(state[q+UNIT_MOVE_SPEED]+state[p+UNIT_MOVE_SPEED])*EXECUTION_DELAY {aggro=true;}
                        }
                    }
                    let mut chosen=(-1i32).runtime();let mut chosen_farm=false;let mut can_aim=false;
                    for spell in 0..3 {
                        let level=crate::physics::spell_level(state[p+UNIT_LEVEL],spell);let k=(spell*4+level-1)*8;
                        if obs[row+observe::ACTION_MASK+6+spell]>0.0f32 {
                            let desired=(state[enemy+UNIT_Y]-state[p+UNIT_Y]).atan2(state[enemy+UNIT_X]-state[p+UNIT_X]);
                            let turn=(wrap_angle(desired-state[p+FACING]).abs()/(hero[6]*TURN_STEP_FACTOR)).ceil()/30.0f32;
                            let delay=EXECUTION_DELAY+spells[k+5]+turn;
                            let ex=state[enemy+UNIT_X]+motion[eh+2]*delay;let ey=state[enemy+UNIT_Y]+motion[eh+3]*delay;
                            let px=state[p+UNIT_X]+motion[ph+2]*delay;let py=state[p+UNIT_Y]+motion[ph+3]*delay;
                            let mut facing=state[p+FACING];
                            if state[p+ORDER]==1.0f32 || state[p+ORDER]==2.0f32 {facing=(state[p+DEST_Y]-py).atan2(state[p+DEST_X]-px);}
                            else if state[p+TARGET]==(1-side)as f32 {facing=(ey-py).atan2(ex-px);}
                            let cx=px+facing.cos()*spells[k+3];let cy=py+facing.sin()*spells[k+3];
                            let dx=ex-cx;let dy=ey-cy;let hero_dist=(dx*dx+dy*dy).sqrt();
                            let mut kills=false;
                            for u in 6..CREEP_END {let q=base+u*STRIDE;let uh=mh+u*MOTION_FIELDS;let f=(a*UNITS+u)*10;
                                if state[q+ACTIVE]>0.0f32 && state[q+TEAM]!=side as f32 && forecasts[f+1]==time && forecasts[f]==state[q+GENERATION] {
                                    let dx=state[q+UNIT_X]+motion[uh+2]*delay-cx;let dy=state[q+UNIT_Y]+motion[uh+3]*delay-cy;
                                    let radius=spells[k+4]+crate::physics::hull(state[q+KIND]);
                                    let rate=(state[q+UNIT_HEALTH]-forecasts[f+3])/(forecasts[f+2]-time).max(1.0f32/30.0f32);
                                    let hp_at_cast=state[q+UNIT_HEALTH]-rate*delay;
                                    let damage=(spells[k]+state[p+UNIT_SOULS]*2.0f32)*(1.0f32-state[q+UNIT_MAGIC_RESIST]);
                                    if dx*dx+dy*dy<=radius*radius && hp_at_cast>0.0f32 && hp_at_cast<=damage {kills=true;}
                                }
                            }
                            let farm=kills && hero_dist<=(spells[k+4]+crate::physics::hull(0.0f32))*(1.0f32-FARM_RAZE_MARGIN);
                            let reserve=state[p+UNIT_MANA]>=spells[k+1]*(1.0f32+MANA_RESERVE_RAZES);
                            let harass=reserve && hero_dist<=(spells[k+4]+crate::physics::hull(0.0f32))*(1.0f32-HERO_RAZE_MARGIN);
                            if (farm || harass) && (chosen<0 || (farm && !chosen_farm)) {chosen=spell as i32;chosen_farm=farm;}
                            let radial=((ex-px)*(ex-px)+(ey-py)*(ey-py)).sqrt();
                            if reserve && (radial-spells[k+3]).abs()<spells[k+4]*(1.0f32-HERO_RAZE_MARGIN) {can_aim=true;}
                        }
                    }
                    if chosen>=0 {action=6+chosen as usize;target=0.0f32;change=true;}
                    else if !farm_soon && !aggro && distance(state,p,enemy)<=crate::physics::attack_reach(state,p,enemy) {
                        // One attack cycle; subsequent decisions may replace it for farming.
                        action=3;target=1.0f32;change=true;
                    }else if !farm_soon && can_aim {
                        let dx=state[enemy+UNIT_X]-state[p+UNIT_X];let dy=state[enemy+UNIT_Y]-state[p+UNIT_Y];let length=(dx*dx+dy*dy).sqrt().max(1.0f32);
                        gx=state[p+UNIT_X]+dx/length*AIM_STEP;gy=state[p+UNIT_Y]+dy/length*AIM_STEP;action=1;change=true;
                    }
                }
            }
            if change {
                actions[a*6]=action as f32;actions[a*6+2]=0.0f32;actions[a*6+3]=target;
                if action==1 {let dx=gx-state[p+UNIT_X];let dy=gy-state[p+UNIT_Y];let scale=(dx.abs()/1024.0f32).max(dy.abs()/1024.0f32).max(1.0f32);
                    actions[a*6+4]=(64.0f32+dx/scale/16.0f32).round().clamp(0.0f32,128.0f32);actions[a*6+5]=(64.0f32+dy/scale/16.0f32).round().clamp(0.0f32,128.0f32);
                }
            }
        }
    }
}

pub struct Controller {motion:CubeTensor,left:Version,right:Version,swap:bool}
impl Controller {
    pub fn new(w:&World,left:Version,right:Version,swap:bool)->Self {Self{motion:primitive(Tensor::<1>::zeros([w.matches*MOTION_WORDS],&w.device)),left,right,swap}}
    pub fn actions(&self,t:&Teacher,w:&World,obs:&Observed)->Tensor<2,Int>{
        if self.left==Version::V9 && self.right==Version::V9 {return t.actions(w,obs);}
        let before=t.command_snapshot();let original=t.actions(w,obs);let actions=primitive(original.clone().float());let packed=primitive(obs.packed.clone());
        motion::launch(&w.state.client,CubeCount::Static(w.matches.div_ceil(32)as u32,1,1),CubeDim::new_1d(32),argument(&w.state),argument(&w.meta),argument(&self.motion),w.matches);
        overlay::launch(&w.state.client,CubeCount::Static((w.matches*2).div_ceil(32)as u32,1,1),CubeDim::new_1d(32),argument(&w.state),argument(&w.meta),argument(&w.hero),argument(&w.spells),argument(&w.commands),argument(&w.teacher_combat),argument(&self.motion),argument(&before.0),argument(&t.memory),argument(&t.forecasts),argument(&t.audit),argument(&packed),argument(&actions),w.matches*2,w.engine.pre_game_seconds,self.left.number(),self.right.number(),self.swap);
        let actions=tensor::<2>(actions).int();t.reconcile_commands(w,obs,before,original,actions.clone());actions
    }
}

pub fn check_pin()->Result<()> {
    ensure!(Version::V10.sha256()==include_str!("teacher_v10.sha256").trim(),"teacher-v10 SHA-256 pin mismatch");Ok(())
}
