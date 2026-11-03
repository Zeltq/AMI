//! Deterministic lane teacher. Decisions and audit stay on the simulation device.
//! Reads no RNG, future state, or rolled projectile damage. See report.json's
//! supplementary_inputs for current-state fields not present in the policy ABI.
use anyhow::{ensure,Result};
use burn::tensor::{Tensor,Int};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use std::{path::Path,time::Instant};
use crate::{contract::*,world::*,observe::{self,Observed,TARGETS,WIDTH},runtime::{argument,primitive,tensor},sampling::Sampler};

pub const AUDIT_HEADER:usize=64;
pub const EXAMPLES:usize=24;
pub const EXAMPLE_FIELDS:usize=30;
pub const AUDIT_FIELDS:usize=AUDIT_HEADER+EXAMPLES*EXAMPLE_FIELDS;
const MEMORY_FIELDS:usize=16;
#[path="teacher_dagger.rs"]
mod dagger;
const FORECAST_FIELDS:usize=10;
const HORIZON:usize=96;
const TRACE_UNITS:usize=64;
const TRACE_WORDS:usize=META+TRACE_UNITS*STRIDE+SHOTS*SHOT_FIELDS+2*MEMORY_FIELDS+2*TRACE_UNITS*FORECAST_FIELDS+12;

#[cube]
fn physical(state:&mut [f32],hero:&[f32],combat:&[f32],base:usize,q:usize,source_kind:usize,damage:f32,time:f32)->f32 {
    let armor=effective_armor(state,hero,base,q,time);
    (damage*combat[source_kind*5+state[q+KIND]as usize]*(1.0f32-0.06f32*armor/(1.0f32+0.06f32*armor.abs()))).ceil()
}

#[cube]
fn point(state:&mut [f32],p:usize)->f32 {
    let base=if state[p+KIND]==0.0f32{1.6f32}else if state[p+KIND]==1.0f32{0.9f32}else if state[p+KIND]==4.0f32{3.0f32}else{1.0f32};
    state[p+UNIT_ATTACK_POINT]*state[p+UNIT_ATTACK_INTERVAL]/base
}

#[cube]
fn flight(state:&mut [f32],hero:&[f32],p:usize,q:usize)->f32 {
    let dist=distance(state,p,q);flight_from(state,hero,p,dist)
}

#[cube]
fn flight_from(state:&mut [f32],hero:&[f32],p:usize,dist:f32)->f32 {
    let speed=state[p+UNIT_PROJECTILE_SPEED];
    let mut length=dist;
    // Current engine uses the two symmetric SF muzzle offsets. No random hand
    // is inspected. Custom animation profiles are reported as an approximation.
    if state[p+KIND]==0.0f32 {length=((dist-hero[24])*(dist-hero[24])+hero[25]*hero[25]).sqrt();}
    if speed>0.0f32 {f32::max(0.0f32,((length-8.0f32)/speed*30.0f32).ceil()-1.0f32)/30.0f32}else{0.0f32}
}

#[cube]
fn own_hit(state:&mut [f32],hero:&[f32],meta:&[f32],base:usize,p:usize,q:usize,m:usize,side:usize,time:f32)->f32 {
    let dist=distance(state,p,q);let reach=crate::physics::attack_reach(state,p,q);
    let approach=(f32::max(0.0f32,dist-reach)/state[p+UNIT_MOVE_SPEED].max(1.0f32)*30.0f32).ceil()/30.0f32;
    let desired=(state[q+UNIT_Y]-state[p+UNIT_Y]).atan2(state[q+UNIT_X]-state[p+UNIT_X]);
    let angle=wrap_angle(desired-state[p+FACING]).abs();
    let turn=(f32::max(0.0f32,angle-0.2f32-hero[6]*0.4712389f32)/hero[6]).ceil()/30.0f32;
    // enqueue stores tick+6; executor checks due+1 <= tick. Command actually
    // runs seven ticks after this observation. Point/flight use 30Hz rounding.
    let execute=time+7.0f32/30.0f32;let mut ready=state[p+ATTACK_READY];
    if state[p+WINDUP]>time && execute>=state[p+WINDUP] {ready=ready.max(state[p+WINDUP]+state[p+UNIT_ATTACK_INTERVAL]-point(state,p));}
    let moving_turn=(f32::max(0.0f32,angle-2.3561945f32)/(hero[6]*0.4712389f32)).ceil()/30.0f32;
    let travel=if approach>0.0f32 {f32::max(turn,approach+moving_turn)}else{turn};
    let start=f32::max(execute+travel,ready);
    let error=meta[m*META+META_ATTACK_POINT_ERROR+side];
    let windup=f32::max(0.0f32,(point(state,p)*30.0f32-error-0.0001f32).ceil())/30.0f32;
    let _unused_base=base;
    start+windup+flight_from(state,hero,p,dist.min(reach))
}

#[cube(launch)]
fn decide(state:&mut [f32],shots:&[f32],meta:&[f32],hero:&[f32],combat:&[f32],proto:&[f32],observations:&[f32],refs:&[f32],
    memory:&mut [f32],forecasts:&mut [f32],seen_attacks:&mut [f32],audit:&[f32],actions:&mut [f32],actors:usize,pregame:f32){
    let a=ABSOLUTE_POS;
    if a<actors {
        let m=a/2;let side=a%2;let base=m*UNITS*STRIDE;let p=base+side*STRIDE;let row=a*WIDTH;let mem=a*MEMORY_FIELDS;
        let time=(meta[m*META]+pregame*30.0f32)/30.0f32;
        for f in 0..6 {actions[a*6+f]=0.0f32;}actions[a*6+4]=64.0f32;actions[a*6+5]=64.0f32;
        if audit[m*AUDIT_FIELDS]==0.0f32 {
            if memory[mem]!=meta[m*META+1]+1.0f32 {
                for f in 0..MEMORY_FIELDS{memory[mem+f]=0.0f32;}memory[mem]=meta[m*META+1]+1.0f32;
                for u in 0..UNITS {seen_attacks[(a*UNITS+u)*2]=0.0f32;seen_attacks[(a*UNITS+u)*2+1]=0.0f32;}
                memory[mem+1]=-1.0f32;memory[mem+10]=-1.0f32;
            }
            // Standard shop automation; the initialized mode already supplies
            // Wraith Band and the free scroll. Buy one tango, then boots/bracer.
            let tango=state[p+ITEM_START+6]+state[p+STASH_START+6]+state[p+COURIER_START+6]+state[p+BACKPACK_START+6];
            let flask=state[p+ITEM_START+5]+state[p+STASH_START+5]+state[p+COURIER_START+5]+state[p+BACKPACK_START+5];
            let missing_hp=state[p+UNIT_HEALTH_MAX]-state[p+UNIT_HEALTH];
            if tango==0.0f32 && meta[m*META]<0.0f32 && observations[row+observe::BUY_MASK+5]>0.0f32 {actions[a*6+1]=5.0f32;}
            else if missing_hp>200.0f32 && flask==0.0f32 && state[p+FLASK_END]<=time && observations[row+observe::BUY_MASK+4]>0.0f32 {actions[a*6+1]=4.0f32;}
            else if observations[row+observe::BUY_MASK+1]>0.0f32 {actions[a*6+1]=1.0f32;}
            if observations[row+observe::ACTION_MASK+1]>0.0f32 && state[p+ACTIVE]>0.0f32 {
                let mut known=Array::<i32>::new(UNITS);
                for u in 0..UNITS {known[u]=-1;}
                known[side]=0;if observations[row+observe::HERO_VALID+1]>0.0f32 {known[1-side]=1;}
                for c in 0..observe::CREEPS {let id=refs[a*TARGETS+2+c];if id>=0.0f32 && id<(UNITS as f32){known[id as usize]=(2+c)as i32;}}
                for t in 0..4 {
                    // Physics reads attack phases only for visible (or own)
                    // towers. is_attacking on an unseen cache is always zero.
                    let at=row+observe::TOWER_OFFSET+t*2*UNIT_COUNT;
                    if state[base+(t+2)*STRIDE+TEAM]==side as f32 || observations[at+UNIT_COUNT+UNIT_IS_ATTACKING]>0.0f32 {known[t+2]=(2+observe::CREEPS+t)as i32;}
                }
                for u in 0..UNITS {if known[u]>=0 {let r=base+u*STRIDE;seen_attacks[(a*UNITS+u)*2]=state[r+UNIT_ATTACK_DAMAGE];seen_attacks[(a*UNITS+u)*2+1]=state[r+KIND];}}
                let mut threats=0usize;
                for u in 2..CREEP_END {let q=base+u*STRIDE;
                    if known[u]>=0 && state[q+ACTIVE]>0.0f32 && state[q+TEAM]!=side as f32 && (state[q+TARGET]==side as f32 || (state[q+WINDUP]>time && state[q+WINDUP_TARGET]==side as f32)) {threats+=1;}
                }
                let hp_fraction=state[p+UNIT_HEALTH]/state[p+UNIT_HEALTH_MAX];
                if hp_fraction<0.30f32 {memory[mem+15]=1.0f32;}
                if hp_fraction>0.56f32 && threats==0 {memory[mem+15]=0.0f32;}
                // Do not extrapolate attacks past a source's already committed
                // lethal hit. Existing projectiles from that source still fly.
                let mut source_death=Array::<f32>::new(UNITS);
                for u in 0..UNITS {source_death[u]=1e20f32;}
                let opponent=1-side;let enemy=base+opponent*STRIDE;
                if known[opponent]>=0 && state[enemy+WINDUP]>time {
                    let target=state[enemy+WINDUP_TARGET]as usize;
                    if target>=6 && target<CREEP_END && known[target]>=0 {let q=base+target*STRIDE;let raw=state[enemy+UNIT_ATTACK_DAMAGE]-hero[27];
                        let dealt=physical(state,hero,combat,base,q,0,raw,time);
                        if state[q+GENERATION]==state[enemy+WINDUP_GENERATION] && state[q+UNIT_HEALTH]<=dealt {source_death[target]=state[enemy+WINDUP]+flight(state,hero,enemy,q);}
                    }
                }
                for s in 0..SHOTS {let k=(m*SHOTS+s)*SHOT_FIELDS;let target=shots[k+3]as usize;let caster=shots[k+7]as usize;
                    if shots[k]==1.0f32 && caster<2 && target>=6 && target<CREEP_END && known[target]>=0 {
                        let q=base+target*STRIDE;let raw=seen_attacks[(a*UNITS+caster)*2]-hero[27]-4.0f32;
                        let dealt=physical(state,hero,combat,base,q,0,raw,time);
                        if raw>0.0f32 && state[q+GENERATION]==shots[k+4] && state[q+UNIT_HEALTH]<=dealt {
                            let dx=state[q+UNIT_X]-shots[k+1];let dy=state[q+UNIT_Y]-shots[k+2];let eta=f32::max(0.0f32,((dx*dx+dy*dy).sqrt()-8.0f32)/shots[k+5].max(1.0f32));
                            source_death[target]=source_death[target].min(time+eta);
                        }
                    }
                }
                let mut best=(-1i32).runtime();let mut best_score=(1e20f32).runtime();let mut best_hit=0.0f32.runtime();let mut best_hp=0.0f32.runtime();
                let mut candidates=0usize;let mut cx=0.0f32.runtime();let mut cy=0.0f32.runtime();let mut weight=0.0f32.runtime();
                let mut closest=(1e20f32).runtime();let mut anchor_x=0.0f32.runtime();let mut anchor_y=0.0f32.runtime();
                // First locate the nearest visible wave. Unengaged allied waves
                // behind the hero must not pull the farming position backwards.
                for u in 6..CREEP_END {let q=base+u*STRIDE;
                    if known[u]>=0 && state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>0.0f32 && state[q+TEAM]!=side as f32 {
                        let dist=distance(state,p,q);if dist<closest {closest=dist;anchor_x=state[q+UNIT_X];anchor_y=state[q+UNIT_Y];}
                    }
                }
                for u in 6..CREEP_END {let q=base+u*STRIDE;
                    let dist=distance(state,p,q);
                    if known[u]>=0 && state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>0.0f32 && dist<2200.0f32 {
                        let ax=state[q+UNIT_X]-anchor_x;let ay=state[q+UNIT_Y]-anchor_y;
                        if closest<2200.0f32 && ax*ax+ay*ay<750.0f32*750.0f32 {
                            let w=1.0f32+2.0f32*(1.0f32-state[q+UNIT_HEALTH]/state[q+UNIT_HEALTH_MAX]);cx+=state[q+UNIT_X]*w;cy+=state[q+UNIT_Y]*w;weight+=w;
                        }
                        if dist<1000.0f32 {
                            let f=(a*UNITS+u)*FORECAST_FIELDS;
                            let mut hit=own_hit(state,hero,meta,base,p,q,m,side,time);
                            if state[p+WINDUP]>time && state[p+WINDUP_TARGET]==u as f32 && state[p+WINDUP_GENERATION]==state[q+GENERATION] {hit=state[p+WINDUP]+flight(state,hero,p,q);}
                            else if memory[mem+1]==u as f32 && memory[mem+2]==state[q+GENERATION] && time<memory[mem+3] {hit=memory[mem+5];}
                            let delay=hit-time;
                            let own_min=state[p+UNIT_ATTACK_DAMAGE]-hero[27];let own_max=state[p+UNIT_ATTACK_DAMAGE]+hero[27];
                            let minimum=physical(state,hero,combat,base,q,0,own_min,time);
                            let maximum=physical(state,hero,combat,base,q,0,own_max,time);
                            let mut damage=Array::<f32>::new(HORIZON);let mut uncertainty=Array::<f32>::new(HORIZON);
                            for t in 0..HORIZON {damage[t]=0.0f32;uncertainty[t]=0.0f32;}
                            let mut own_shot=false;
                            for s in 0..SHOTS {let k=(m*SHOTS+s)*SHOT_FIELDS;
                                if shots[k]==1.0f32 && shots[k+3]==u as f32 && shots[k+4]==state[q+GENERATION] {
                                    let source=shots[k+7]as usize;
                                    if source<UNITS {
                                        let dx=state[q+UNIT_X]-shots[k+1];let dy=state[q+UNIT_Y]-shots[k+2];
                                        // Same-tick projectile ordering cannot be
                                        // guaranteed for a shot not yet created.
                                        let t=f32::max(0.0f32,(((dx*dx+dy*dy).sqrt()-8.0f32)/shots[k+5].max(1.0f32)*30.0f32).ceil()-1.0f32)as usize+1;
                                        if t<HORIZON {
                                            // A projectile survives its caster.
                                            // Use the last observed public damage,
                                            // never the hidden rolled shot amount.
                                            let cache=(a*UNITS+source)*2;let kind=shots[k+9]as usize;
                                            let prototype=if kind==0{0usize}else if kind==1{2usize}else{kind+4};
                                            let amount=if seen_attacks[cache]>0.0f32 && seen_attacks[cache+1]==kind as f32 {seen_attacks[cache]}else{proto[prototype*UNIT_COUNT+UNIT_ATTACK_DAMAGE]};
                                            let d=physical(state,hero,combat,base,q,kind,amount,time);
                                            damage[t]+=d;uncertainty[t]+=if source<2 {hero[27]*hero[27]/3.0f32}else{4.0f32};
                                            if source==side {own_shot=true;}
                                        }
                                    }
                                }
                            }
                            for source in 0..UNITS {
                                if source!=side && known[source]>=0 {let r=base+source*STRIDE;
                                    if state[r+ACTIVE]>0.0f32 && state[r+TARGET]==u as f32 && state[r+TARGET_GENERATION]==state[q+GENERATION] && state[r+UNIT_HEALTH]>0.0f32 {
                                        let reach=crate::physics::attack_reach(state,r,q);let d=distance(state,r,q);
                                        let walk=f32::max(0.0f32,d-reach)/state[r+UNIT_MOVE_SPEED].max(1.0f32);
                                        let release=if state[r+WINDUP]>0.0f32 {state[r+WINDUP]}else{f32::max(time+walk,state[r+ATTACK_READY])+point(state,r)};
                                        let first=f32::max(0.0f32,((release-time+flight(state,hero,r,q))*30.0f32).ceil())+if state[r+UNIT_PROJECTILE_SPEED]>0.0f32{1.0f32}else{0.0f32};
                                        let interval=(state[r+UNIT_ATTACK_INTERVAL]*30.0f32).max(1.0f32);
                                        let kind=state[r+KIND]as usize;let attack_damage=state[r+UNIT_ATTACK_DAMAGE];
                                        let amount=physical(state,hero,combat,base,q,kind,attack_damage,time);
                                        let variance=if source<2 {hero[27]}else if state[r+KIND]==1.0f32{hero[31]}else{hero[26+state[r+KIND]as usize]};
                                        for cycle in 0..4 {let t=(first+cycle as f32*interval).ceil()as usize;
                                            let launch=release+cycle as f32*state[r+UNIT_ATTACK_INTERVAL];
                                            if t<HORIZON && launch<=source_death[source]+1.0f32/30.0f32 && (source>=2 || cycle==0) {damage[t]+=amount;uncertainty[t]+=variance*variance/3.0f32;}
                                        }
                                    }
                                }
                            }
                            let regen=state[q+UNIT_HEALTH_REGEN]+if state[q+FLAG_AURA_END]>time {3.0f32}else{0.0f32};
                            let mut hp=state[q+UNIT_HEALTH];let mut variance=0.0f32.runtime();let mut sigma=0.0f32.runtime();let mut predicted=hp;let mut deadline=(HORIZON as f32).runtime();
                            let landing=(delay*30.0f32).round()as usize;
                            for t in 0..HORIZON {
                                hp+=regen/30.0f32;hp-=damage[t];variance+=uncertainty[t];
                                if t==landing {predicted=hp;sigma=variance.sqrt();}
                                if hp<=0.0f32 && deadline==HORIZON as f32 {deadline=t as f32;}
                            }
                            if forecasts[f]!=state[q+GENERATION] {forecasts[f+8]=0.0f32;forecasts[f+9]=0.0f32;}
                            forecasts[f]=state[q+GENERATION];forecasts[f+1]=time;forecasts[f+2]=hit;forecasts[f+3]=predicted;
                            forecasts[f+4]=minimum;forecasts[f+5]=maximum;forecasts[f+6]=0.0f32;forecasts[f+7]=deadline/30.0f32+time;
                            let deny_ok=state[q+TEAM]!=side as f32 || state[q+UNIT_HEALTH]<=state[q+UNIT_HEALTH_MAX]*0.5f32;
                            // Conservative own roll, mean incoming rolls. Never
                            // invent future random damage or let negative HP wrap.
                            let viable=deny_ok && landing<HORIZON && predicted>0.5f32 && predicted<=minimum && !own_shot;
                            if viable {
                                candidates+=1;forecasts[f+6]=1.0f32;forecasts[f+8]=time;
                                // Urgency beats LH/deny preference; stable unit ID
                                // breaks exact ties. This avoids starving denies.
                                let score=deadline+delay*0.25f32+sigma*0.02f32+if state[q+TEAM]==side as f32 {6.0f32}else{0.0f32};
                                if score<best_score {best=u as i32;best_score=score;best_hit=hit;best_hp=predicted;}
                            }
                        }
                    }
                }
                memory[mem+8]=candidates as f32;
                let current=memory[mem+1]as usize;let mut protected=false;
                if memory[mem+1]>=0.0f32 && current<UNITS {let q=base+current*STRIDE;
                    let live=state[q+ACTIVE]>0.0f32 && state[q+GENERATION]==memory[mem+2] && state[q+UNIT_HEALTH]>0.0f32;
                    // Preserve a useful command while it is travelling through
                    // the queue and while its windup is running. Recompute on
                    // every observation; a dead/reused target releases the lock.
                    protected=live && (time<memory[mem+3] || (state[p+WINDUP]>time && state[p+WINDUP_TARGET]==current as f32));
                    let f=(a*UNITS+current)*FORECAST_FIELDS;
                    let release=if state[p+WINDUP]>time {state[p+WINDUP]}else{memory[mem+3]-1.0f32/30.0f32};
                    let doomed=forecasts[f]==state[q+GENERATION] && forecasts[f+1]==time && forecasts[f+3]<=0.0f32;
                    if protected && doomed && best>=0 && best!=current as i32 && time+8.0f32/30.0f32<release {protected=false;}
                }
                for u in 6..CREEP_END {let f=(a*UNITS+u)*FORECAST_FIELDS;
                    if forecasts[f+1]==time && forecasts[f+6]>0.0f32 && ((protected && u!=current) || (!protected && best>=0 && u!=best as usize)) {forecasts[f+9]=time;}
                }
                let mut action=0usize;let mut target=0usize;let mut gx=state[p+UNIT_X];let mut gy=state[p+UNIT_Y];
                if !protected {
                    if best>=0 {let u=best as usize;let q=base+u*STRIDE;
                        action=3;target=known[u]as usize;memory[mem+1]=u as f32;memory[mem+2]=state[q+GENERATION];
                        memory[mem+3]=best_hit-flight(state,hero,p,q)+1.0f32/30.0f32;memory[mem+4]=time;memory[mem+5]=best_hit;memory[mem+6]=best_hp;
                    }else{
                        let sign=if side==0{-1.0f32}else{1.0f32};
                        gx=if side==0{-1050.0f32}else{100.0f32};gy=if side==0{-950.0f32}else{200.0f32};
                        if weight>0.0f32 {gx=cx/weight+sign*210.0f32;gy=cy/weight+sign*210.0f32;}
                        let enemy_tower=base+if side==0{4*STRIDE}else{2*STRIDE};let mut tanks=0usize;
                        for u in 6..CREEP_END {let q=base+u*STRIDE;
                            if known[u]>=0 && state[q+ACTIVE]>0.0f32 && state[q+TEAM]==side as f32 && distance(state,q,enemy_tower)<850.0f32 {tanks+=1;}
                        }
                        let dx=gx-state[enemy_tower+UNIT_X];let dy=gy-state[enemy_tower+UNIT_Y];let d=(dx*dx+dy*dy).sqrt().max(1.0f32);
                        if state[enemy_tower+ACTIVE]>0.0f32 && tanks<2 && d<1000.0f32 {gx=state[enemy_tower+UNIT_X]+dx/d*1000.0f32;gy=state[enemy_tower+UNIT_Y]+dy/d*1000.0f32;}
                        let dx=gx-state[p+UNIT_X];let dy=gy-state[p+UNIT_Y];let d=(dx*dx+dy*dy).sqrt();
                        if d>90.0f32 {action=1;}else if state[p+ORDER]==3.0f32 || state[p+ORDER]==1.0f32 || state[p+ORDER]==2.0f32 {action=4;}
                        // After release a STOP/MOVE safely prevents automatic
                        // repeated attacks into a target with too much HP.
                        memory[mem+1]=-1.0f32;
                    }
                }
                // Sustain uses the same delayed item/move commands. Finish a
                // useful windup unless health is immediately critical.
                if time<memory[mem+14] {action=0;}
                else if !protected || hp_fraction<0.20f32 {
                    if memory[mem+15]>0.0f32 {
                        let tower=base+if side==0{2*STRIDE}else{4*STRIDE};let sign=if side==0{-1.0f32}else{1.0f32};
                        gx=state[tower+UNIT_X]+sign*220.0f32;gy=state[tower+UNIT_Y]+sign*220.0f32;
                        let dx=gx-state[p+UNIT_X];let dy=gy-state[p+UNIT_Y];
                        action=if dx*dx+dy*dy>90.0f32*90.0f32{1usize}else{4usize};memory[mem+1]=-1.0f32;
                    }
                    if threats==0 && missing_hp>200.0f32 && state[p+FLASK_END]<=time && observations[row+observe::ITEM_MASK+5]>0.0f32 {
                        action=10;target=0;actions[a*6+2]=5.0f32;memory[mem+14]=time+0.4f32;
                    }else if missing_hp>100.0f32 && state[p+TANGO_END]<=time && observations[row+observe::ITEM_MASK+6]>0.0f32 {
                        let mut tree=(-1i32).runtime();let mut nearest=(135.0f32).runtime();
                        for t in 0..observe::TREES {let at=row+observe::TREE_OFFSET+t*8;let d=observations[at+3]*2048.0f32;
                            if observations[row+observe::TREE_VALID+t]>0.0f32 && d<nearest {nearest=d;tree=t as i32;}
                        }
                        if tree>=0 {action=10;target=2+observe::CREEPS+4+tree as usize;actions[a*6+2]=6.0f32;memory[mem+14]=time+0.4f32;}
                    }
                }
                if action!=3 && !protected {memory[mem+1]=-1.0f32;}
                if action!=0 && state[p+WINDUP]>time && (action!=3 || state[p+WINDUP_TARGET]!=best as f32) {
                    memory[mem+9]+=1.0f32;memory[mem+10]=state[p+WINDUP_TARGET];memory[mem+11]=time;
                }
                actions[a*6]=action as f32;actions[a*6+3]=target as f32;
                if action==1 {
                    let dx=gx-state[p+UNIT_X];let dy=gy-state[p+UNIT_Y];let scale=(dx.abs()/1024.0f32).max(dy.abs()/1024.0f32).max(1.0f32);
                    actions[a*6+4]=(64.0f32+dx/scale/16.0f32).round().clamp(0.0f32,128.0f32);
                    actions[a*6+5]=(64.0f32+dy/scale/16.0f32).round().clamp(0.0f32,128.0f32);
                }
                memory[mem+12]=action as f32;memory[mem+13]+=if action>0{1.0f32}else{0.0f32};
            }
        }
    }
}

pub struct Teacher {pub memory:CubeTensor,pub forecasts:CubeTensor,pub audit:CubeTensor,seen_attacks:CubeTensor}
impl Teacher {
    pub fn new(w:&World)->Self {
        let zeros=|n|primitive(Tensor::<1>::zeros([n],&w.device));
        Self{memory:zeros(w.matches*2*MEMORY_FIELDS),forecasts:zeros(w.matches*2*UNITS*FORECAST_FIELDS),audit:zeros(w.matches*AUDIT_FIELDS),seen_attacks:zeros(w.matches*2*UNITS*2)}
    }
    pub fn actions(&self,w:&World,obs:&Observed)->Tensor<2,Int>{
        let actions=primitive(Tensor::<2>::zeros([w.matches*2,6],&w.device));let packed=primitive(obs.packed.clone());
        decide::launch(&w.state.client,CubeCount::Static((w.matches*2).div_ceil(32)as u32,1,1),CubeDim::new_1d(32),
            argument(&w.state),argument(&w.shots),argument(&w.meta),argument(&w.hero),argument(&w.combat),argument(&w.prototypes),argument(&packed),argument(&obs.refs),
            argument(&self.memory),argument(&self.forecasts),argument(&self.seen_attacks),argument(&self.audit),argument(&actions),w.matches*2,w.engine.pre_game_seconds);
        tensor::<2>(actions).int()
    }
    pub fn record(&self,w:&World,deadline:f32){
        progress::launch(&w.state.client,CubeCount::Static(w.matches.div_ceil(32)as u32,1,1),CubeDim::new_1d(32),
            argument(&w.state),argument(&w.meta),argument(&self.memory),argument(&self.audit),w.matches,w.engine.pre_game_seconds,deadline);
    }
    pub fn values(&self)->Vec<f32>{tensor::<1>(self.audit.clone()).into_data().try_to_vec::<f32>().unwrap()}
}

/// Called exactly once per actual lane death, inside the regular physics tick.
/// Audit buffers are never used to choose actions.
#[cube]
pub fn death(state:&mut [f32],shots:&mut [f32],meta:&mut [f32],memory:&[f32],forecasts:&[f32],audit:&mut [f32],m:usize,u:usize,time:f32,previous_hp:f32){
    let r=m*AUDIT_FIELDS;let base=m*UNITS*STRIDE;let p=base+u*STRIDE;
    if audit[r]==0.0f32 {
        audit[r+3]+=1.0f32;let credited=state[p+CREDIT]>=0.0f32 && state[p+CREDIT]<2.0f32;
        let mut accessible=false;let mut ready=false;let mut competing=false;let mut cancelled=false;let mut committed=false;
        for h in 0..2 {let q=base+h*STRIDE;let mem=(m*2+h)*MEMORY_FIELDS;
            let f=((m*2+h)*UNITS+u)*FORECAST_FIELDS;
            if forecasts[f]==state[p+GENERATION] && forecasts[f+9]>0.0f32 && time-forecasts[f+9]<3.0f32 {competing=true;}
            if state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>0.0f32 && distance(state,p,q)<=crate::physics::attack_reach(state,q,p) {accessible=true;
                if state[q+ATTACK_READY]<=time {ready=true;}if memory[mem+8]>=2.0f32 {competing=true;}
            }
            if memory[mem+10]==u as f32 && time-memory[mem+11]<2.0f32 {cancelled=true;}
            if memory[mem+1]==u as f32 && memory[mem+2]==state[p+GENERATION] && time-memory[mem+4]<3.0f32 {committed=true;}
        }
        for s in 0..SHOTS {let k=(m*SHOTS+s)*SHOT_FIELDS;if shots[k]==1.0f32 && shots[k+3]==u as f32 && shots[k+4]==state[p+GENERATION] && shots[k+7]<2.0f32 {committed=true;}}
        if accessible {audit[r+5]+=1.0f32;}if credited {audit[r+4]+=1.0f32;if accessible{audit[r+6]+=1.0f32;}}
        else {
            // Mutually exclusive proximate diagnostic labels, not causal
            // counterfactuals. Raw forecasts and weapon state accompany samples.
            let reason=if !accessible {0usize}else if cancelled {3usize}else if competing {2usize}else if committed {4usize}else if !ready {1usize}else{5usize};
            audit[r+7+reason]+=1.0f32;
            let count=audit[r+40]as usize;
            if count<EXAMPLES {
                let e=r+AUDIT_HEADER+count*EXAMPLE_FIELDS;
                audit[e]=meta[m*META]/30.0f32;audit[e+1]=u as f32;audit[e+2]=state[p+GENERATION];audit[e+3]=state[p+TEAM];audit[e+4]=state[p+KIND];
                audit[e+5]=state[p+UNIT_X];audit[e+6]=state[p+UNIT_Y];audit[e+7]=previous_hp;audit[e+8]=state[p+CREDIT];audit[e+9]=reason as f32;
                for h in 0..2 {let q=base+h*STRIDE;let f=((m*2+h)*UNITS+u)*FORECAST_FIELDS;let b=e+10+h*10;
                    audit[b]=distance(state,p,q);audit[b+1]=state[q+ATTACK_READY]-time;audit[b+2]=state[q+WINDUP]-time;audit[b+3]=state[q+WINDUP_TARGET];
                    audit[b+4]=forecasts[f+1];audit[b+5]=forecasts[f+2];audit[b+6]=forecasts[f+3];audit[b+7]=forecasts[f+4];audit[b+8]=forecasts[f+5];audit[b+9]=forecasts[f+6];
                }
                audit[r+40]+=1.0f32;
            }
        }
    }
}

#[cube(launch)]
fn progress(state:&[f32],meta:&[f32],memory:&[f32],audit:&mut [f32],matches:usize,pregame:f32,deadline:f32){
    let m=ABSOLUTE_POS;if m<matches {let r=m*AUDIT_FIELDS;let base=m*UNITS*STRIDE;let elapsed=meta[m*META]/30.0f32+pregame;
        if audit[r]==0.0f32 {
            for h in 0..2 {let p=base+h*STRIDE;let mem=(m*2+h)*MEMORY_FIELDS;let x=state[p+UNIT_X];let y=state[p+UNIT_Y];
                audit[r+16+h*2]=state[p+UNIT_LAST_HITS];audit[r+17+h*2]=state[p+UNIT_DENIES];
                if audit[r+20+h]==0.0f32 && state[p+ACTIVE]>0.0f32 && x*x+y*y<4000000.0f32 && (x-y).abs()<1200.0f32 {audit[r+20+h]=elapsed+1.0f32;}
                if audit[r+22+h]==0.0f32 && (state[p+ORDER]==1.0f32 || state[p+ORDER]==2.0f32) {audit[r+22+h]=elapsed+1.0f32;}
                let dx=x-audit[r+28+h*2];let dy=y-audit[r+29+h*2];let gx=state[p+DEST_X]-x;let gy=state[p+DEST_Y]-y;
                if state[p+ACTIVE]>0.0f32 && (state[p+ORDER]==1.0f32 || state[p+ORDER]==2.0f32) && gx*gx+gy*gy>1024.0f32 && dx*dx+dy*dy<1.0f32 {audit[r+26+h]+=4.0f32/30.0f32;}else{audit[r+26+h]=0.0f32;}
                audit[r+24+h]=audit[r+24+h].max(audit[r+26+h]);audit[r+28+h*2]=x;audit[r+29+h*2]=y;
                audit[r+33+h]=state[p+DEATHS];audit[r+35+h]=memory[mem+9];audit[r+37+h]=memory[mem+13];
            }
            audit[r+1]=meta[m*META]/30.0f32;audit[r+14]=meta[m*META+3];audit[r+39]=meta[m*META+7];
            if meta[m*META+2]>0.0f32 || meta[m*META]/30.0f32>=deadline {audit[r]=1.0f32;audit[r+2]=meta[m*META+2];}
        }
    }
}

#[cube(launch)]
#[allow(unused_assignments)]
fn trace_first(state:&[f32],shots:&[f32],meta:&[f32],memory:&[f32],forecasts:&[f32],actions:&[f32],out:&mut [f32],step:usize){
    let i=ABSOLUTE_POS;
    if i<TRACE_WORDS {let mut value=0.0f32.runtime();
        if i<META {value=meta[i];}
        else if i<META+TRACE_UNITS*STRIDE {value=state[i-META];}
        else if i<META+TRACE_UNITS*STRIDE+SHOTS*SHOT_FIELDS {value=shots[i-META-TRACE_UNITS*STRIDE];}
        else if i<META+TRACE_UNITS*STRIDE+SHOTS*SHOT_FIELDS+2*MEMORY_FIELDS {value=memory[i-META-TRACE_UNITS*STRIDE-SHOTS*SHOT_FIELDS];}
        else if i<TRACE_WORDS-12 {let f=i-META-TRACE_UNITS*STRIDE-SHOTS*SHOT_FIELDS-2*MEMORY_FIELDS;value=forecasts[(f/(TRACE_UNITS*FORECAST_FIELDS)*UNITS+f/FORECAST_FIELDS%TRACE_UNITS)*FORECAST_FIELDS+f%FORECAST_FIELDS];}
        else {value=actions[i-TRACE_WORDS+12];}
        out[step*TRACE_WORDS+i]=value;
    }
}

pub fn run(engine:&Path,root:&Path,matches:usize,seconds:f32,seed:u32,wall_seconds:f64,trace:bool)->Result<serde_json::Value>{
    let process_started=Instant::now();
    ensure!(matches>0 && seconds>0. && wall_seconds>0.,"positive matches and time limits required");
    std::fs::create_dir_all(root)?;
    ensure!(!root.join("config.json").exists(),"output already used; choose a fresh directory");
    let command=std::env::args().collect::<Vec<_>>();
    crate::artifacts::atomic_json(&root.join("config.json"),&serde_json::json!({"engine":engine,"matches":matches,"seconds":seconds,"seed":seed,"wall_seconds":wall_seconds,"command":command,"teacher_sha256":crate::runtime::bytes_sha256(include_bytes!("scripted_teacher.rs"))}))?;
    let frozen=crate::artifacts::freeze_environment(engine,root,false)?;
    let d=crate::runtime::device();let w=World::load(&frozen,matches,&d)?;let teacher=Teacher::new(&w);let rng=Sampler::new(matches,seed,&d);
    let started=Instant::now();let mut steps=0usize;let mut previous=observe::initial_previous(matches*2,&d);
    let limit=((w.engine.pre_game_seconds+seconds)*7.5).ceil()as usize+2;
    let trace_buffer=if trace{Some(primitive(Tensor::<2>::zeros([limit,TRACE_WORDS],&d)))}else{None};
    for step in 0..limit {
        let obs=w.observe(previous.clone());let actions=teacher.actions(&w,&obs);
        if let Some(buffer)=&trace_buffer {let raw_actions=primitive(actions.clone().float());
            trace_first::launch(&w.state.client,CubeCount::Static(TRACE_WORDS.div_ceil(256)as u32,1,1),CubeDim::new_1d(256),argument(&w.state),argument(&w.shots),argument(&w.meta),argument(&teacher.memory),argument(&teacher.forecasts),argument(&raw_actions),argument(buffer),step);
        }
        previous=obs.previous(actions.clone(),&w.bases,&w.state);
        let _=w.advance_teacher(actions,&obs.refs,&rng,&teacher,true);
        teacher.record(&w,seconds);steps=step+1;
        if steps%128==0 {
            let values=teacher.values();let finished=values.chunks_exact(AUDIT_FIELDS).filter(|v|v[0]>0.).count();
            eprintln!("teacher steps={steps} game_time={:.1} finished={finished}/{matches} wall={:.2}s",values[1],started.elapsed().as_secs_f64());
            crate::artifacts::atomic_json(&root.join("progress.json"),&serde_json::json!({"steps":steps,"finished":finished,"matches":matches,"wall_seconds":started.elapsed().as_secs_f64(),"game_time":values[1],"pid":std::process::id()}))?;
            if finished==matches || started.elapsed().as_secs_f64()>=wall_seconds || root.join("stop.json").exists(){break;}
        }
    }
    d.sync()?;let wall=started.elapsed().as_secs_f64();let values=teacher.values();
    if let Some(buffer)=trace_buffer {
        use std::io::Write;
        let data=tensor::<2>(buffer).reshape([limit*TRACE_WORDS]).into_data().try_to_vec::<f32>()?;
        let mut zip=flate2::write::GzEncoder::new(std::fs::File::create(root.join("first-match.f32.gz"))?,flate2::Compression::fast());
        let bytes:Vec<u8>=data[..steps*TRACE_WORDS].iter().flat_map(|v|v.to_le_bytes()).collect();zip.write_all(&bytes)?;zip.finish()?;
        crate::artifacts::atomic_json(&root.join("trace-layout.json"),&serde_json::json!({"steps":steps,"width":TRACE_WORDS,"meta":META,"units":TRACE_UNITS,"stride":STRIDE,"shots":SHOTS,"shot_fields":SHOT_FIELDS,"memory_fields":MEMORY_FIELDS,"forecast_fields":FORECAST_FIELDS,"order":["meta","state","shots","memory","forecasts","actions"],"phase":"after decision before enqueue"}))?;
    }
    let mut report=report(&values,matches,steps,wall,&w.fingerprint,w.engine.sf_projectile_animation.is_some());
    report["total_wall_seconds_including_setup_and_trace_export"]=serde_json::json!(process_started.elapsed().as_secs_f64());
    crate::artifacts::atomic_json(&root.join("report.json"),&report)?;
    let examples:Vec<_>=values.chunks_exact(AUDIT_FIELDS).enumerate().flat_map(|(m,v)|{
        (0..v[40]as usize).map(move|i|serde_json::json!({"match":m,"fields":&v[AUDIT_HEADER+i*EXAMPLE_FIELDS..AUDIT_HEADER+(i+1)*EXAMPLE_FIELDS]}))
    }).collect();
    crate::artifacts::atomic_json(&root.join("misses.json"),&serde_json::json!({"forecast_clock":"elapsed episode seconds = game_time + pre_game_seconds","pre_game_seconds":w.engine.pre_game_seconds,"fields":["game_time","unit","generation","team","kind","x","y","hp_before_death","killer","reason","radiant_distance","radiant_ready_in","radiant_windup_in","radiant_windup_target","radiant_forecast_time","radiant_hit_time","radiant_predicted_hp","radiant_damage_min","radiant_damage_max","radiant_viable","dire_distance","dire_ready_in","dire_windup_in","dire_windup_target","dire_forecast_time","dire_hit_time","dire_predicted_hp","dire_damage_min","dire_damage_max","dire_viable"],"reason_codes":["distance","cooldown","simultaneous_targets","cancellation","timing","prediction"],"samples_per_match_limit":EXAMPLES,"examples":examples}))?;
    Ok(report)
}

fn report(values:&[f32],matches:usize,steps:usize,wall:f64,fingerprint:&str,animation:bool)->serde_json::Value {
    let rows:Vec<_>=values.chunks_exact(AUDIT_FIELDS).collect();let sum=|i|rows.iter().map(|v|v[i]as f64).sum::<f64>();
    let ratio=|n,d|serde_json::json!({"numerator":sum(n),"denominator":sum(d),"fraction":if sum(d)>0.{Some(sum(n)/sum(d))}else{None}});
    let sides:Vec<_>=(0..2).map(|h|{let arrived:Vec<_>=rows.iter().filter(|v|v[20+h]>0.).map(|v|v[20+h]-1.).collect();
        serde_json::json!({"side":if h==0{"radiant"}else{"dire"},"last_hits":sum(16+h*2),"denies":sum(17+h*2),"arrived":arrived.len(),"matches":matches,
            "arrival_seconds_mean":if arrived.is_empty(){None}else{Some(arrived.iter().sum::<f32>()/arrived.len()as f32)},"arrival_seconds_max":arrived.iter().copied().reduce(f32::max),
            "first_move_seconds_max":rows.iter().filter(|v|v[22+h]>0.).map(|v|v[22+h]-1.).reduce(f32::max),"longest_movement_stall_seconds":rows.iter().map(|v|v[24+h]).reduce(f32::max),"cancellation_commands":sum(35+h),"hero_deaths":sum(33+h)})
    }).collect();
    let per_match:Vec<_>=rows.iter().enumerate().map(|(m,v)|serde_json::json!({"match":m,"finished_prefix":v[0],"native_terminal":v[2],"game_seconds":v[1],"lane_deaths":v[3],"credited":v[4],"accessible_deaths":v[5],"accessible_credited":v[6],"radiant_lh":v[16],"radiant_deny":v[17],"dire_lh":v[18],"dire_deny":v[19],"sim_error":v[39]})).collect();
    serde_json::json!({"matches":matches,"finished_prefixes":sum(0),"completed_matches":sum(2),"steps":steps,"wall_seconds":wall,"aggregate_simulated_seconds":sum(14)/30.,"aggregate_realtime_factor":sum(14)/30./wall,"seconds_per_match_mean":sum(14)/30./matches as f64,
        "game_seconds_min":rows.iter().map(|v|v[1]).reduce(f32::min),"game_seconds_max":rows.iter().map(|v|v[1]).reduce(f32::max),"sides":sides,"all_lane_creeps":ratio(4,3),"within_attack_distance":ratio(6,5),
        "misses":{"distance":sum(7),"cooldown":sum(8),"simultaneous_targets":sum(9),"cancellation":sum(10),"timing":sum(11),"prediction":sum(12),"total":sum(3)-sum(4),"classification":"exclusive proximate heuristics; samples include raw forecasts, no counterfactual causal claim"},
        "per_match":per_match,"environment_fingerprint":fingerprint,"clock":{"hz":30,"decision_ticks":4,"offset_ticks":6,"executor_due_plus_one":true},
        "supplementary_inputs":["visible units: current attack target and its generation, attack-ready time, windup release time","own hero: fractional windup rounding carry (reconstructible from past attacks)","tracking shots to observed targets: caster and target identity/generation, current position and speed; rolled shot damage is NOT read","static calibrated damage variance, type multipliers, hulls, turn and muzzle rules"],
        "limitations":["incoming damage uses expected rolls, own damage uses minimum roll","NPCs are assumed to retain their observed target over the forecast horizon","a doomed windup is switched only when another viable target exists and the delayed command can arrive before release","miss reasons are diagnostics and can overlap causally","only the first episode per GPU slot is measured"],"custom_projectile_animation_approximated":animation})
}

/// Small controlled GPU scenarios. Host setup/readback is confined to checks;
/// normal self-play never downloads per-match state or per-step actions.
pub fn check(engine:&Path,root:&Path)->Result<serde_json::Value>{
    use burn::tensor::TensorData;
    std::fs::create_dir_all(root)?;
    let names=["last_hit","deny","three_attackers","competing_targets","incoming_projectile","turn_cooldown","deny_above_half","contested_attack_switch"];
    let matches=names.len()*2;let d=crate::runtime::device();let mut w=World::load(engine,matches,&d)?;
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>()?;
    let mut metadata=vec![0.;matches*META];let mut shots=vec![0.;matches*SHOTS*SHOT_FIELDS];
    for m in 0..matches {let side=m%2;let case=m/2;let base=m*UNITS*STRIDE;let sign=if side==0{1.}else{-1.};metadata[m*META]=-300.;
        for u in 2..UNITS {state[base+u*STRIDE+ACTIVE]=0.;}
        for h in 0..2 {let p=base+h*STRIDE;state[p+UNIT_X]=if h==side{0.}else if case==7{sign*600.}else{sign*6000.};state[p+UNIT_Y]=0.;state[p+UNIT_Z]=0.;state[p+FACING]=if (side==0)==(h==side || case!=7){0.}else{std::f32::consts::PI};state[p+ATTACK_AIM]=state[p+FACING];}
        let now=w.engine.pre_game_seconds-10.;
        for j in 0..if case==3||case==7{2}else{1} {let u=6+j;let p=base+u*STRIDE;
            for f in 0..STRIDE{state[p+f]=0.;}for f in 0..UNIT_COUNT{state[p+f]=w.engine.prototypes[6][f];}
            state[p+UNIT_X]=sign*(250.+j as f32*40.);state[p+UNIT_Y]=j as f32*100.;state[p+UNIT_Z]=0.;
            state[p+ACTIVE]=1.;state[p+GENERATION]=1.;state[p+TEAM]=if case==1||case==6{side as f32}else{(1-side)as f32};state[p+KIND]=2.;
            state[p+UNIT_HEALTH]=match case{2=>94.,4=>80.,6=>400.,_=>30.};state[p+TARGET]=-1.;state[p+LANE_POSITION]=-1.;state[p+UNIT_MOVE_SPEED]=0.;state[p+BOUNTY]=35.;state[p+XP_BOUNTY]=57.;
        }
        if case==2 || case==3 || case==4 || case==7 {let count=if case==2{3}else if case==3||case==7{2}else{1};
            for j in 0..count {let u=8+j;let p=base+u*STRIDE;
                for f in 0..STRIDE {state[p+f]=0.;}for f in 0..UNIT_COUNT {state[p+f]=w.engine.prototypes[if case==4{7}else{6}][f];}
                state[p+UNIT_X]=sign*300.;state[p+UNIT_Y]=if case==3||case==7{j as f32*100.}else{j as f32*32.};state[p+ACTIVE]=1.;state[p+GENERATION]=1.;state[p+TEAM]=side as f32;state[p+KIND]=if case==4{3.}else{2.};
                state[p+TARGET]=if case==3||case==7{(6+j)as f32}else{6.};state[p+TARGET_GENERATION]=1.;state[p+LANE_POSITION]=-1.;state[p+UNIT_MOVE_SPEED]=0.;
                state[p+WINDUP_TARGET]=state[p+TARGET];state[p+WINDUP_GENERATION]=1.;
                state[p+WINDUP]=now+if case==2{0.4}else{1.2+j as f32*1.2};
                if case==3||case==7 {state[p+UNIT_ATTACK_DAMAGE]=40.;}
                if case==4 {state[p+WINDUP]=0.;state[p+ATTACK_READY]=now+10.;state[p+UNIT_ATTACK_DAMAGE]=40.;
                    let k=m*SHOTS*SHOT_FIELDS;shots[k]=1.;shots[k+1]=sign*650.;shots[k+2]=0.;shots[k+3]=6.;shots[k+4]=1.;shots[k+5]=900.;shots[k+6]=40.;shots[k+7]=u as f32;shots[k+8]=side as f32;shots[k+9]=3.;
                }
            }
        }
        if case==5 {let p=base+side*STRIDE;state[p+FACING]+=std::f32::consts::PI;state[p+ATTACK_AIM]=state[p+FACING];state[p+ATTACK_READY]=now+0.3;}
    }
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[matches*UNITS*STRIDE]),&d));
    w.meta=primitive(Tensor::<1>::from_data(TensorData::new(metadata,[matches*META]),&d));
    w.shots=primitive(Tensor::<1>::from_data(TensorData::new(shots,[matches*SHOTS*SHOT_FIELDS]),&d));
    w.geometry=primitive(Tensor::<1>::ones([512*512*2],&d));w.movement_geometry=None;w.fow_blockers=primitive(Tensor::<1>::zeros([512*512],&d));w.static_vision=primitive(Tensor::<1>::zeros([512*512],&d));w.tree_map=primitive(Tensor::<1>::zeros([512*512],&d));
    let teacher=Teacher::new(&w);let rng=Sampler::new(matches,20261102,&d);let previous=observe::initial_previous(matches*2,&d);
    let mut frames=vec![];let started=Instant::now();let mut deterministic=true;let mut delayed=true;
    for step in 0..24 {let obs=w.observe(previous.clone());let actions=teacher.actions(&w,&obs);
        if step==0 {
            let before=actions.clone().into_data().try_to_vec::<i32>()?;
            let duplicate=Teacher::new(&w).actions(&w,&obs).into_data().try_to_vec::<i32>()?;deterministic=before==duplicate;
        }
        let raw_actions=actions.clone().into_data().try_to_vec::<i32>()?;
        let _=w.advance_teacher(actions,&obs.refs,&rng,&teacher,false);teacher.record(&w,100.);
        let raw=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>()?;
        if step==0 {delayed=(0..matches).all(|m|raw[(m*UNITS+m%2)*STRIDE+ORDER]!=3.);}
        frames.push(serde_json::json!({"step":step,"actions":raw_actions,"cases":(0..matches).map(|m|{let p=(m*UNITS+m%2)*STRIDE;serde_json::json!({"match":m,"lh":raw[p+UNIT_LAST_HITS],"deny":raw[p+UNIT_DENIES],"order":raw[p+ORDER],"windup":raw[p+WINDUP],"target_hp":raw[(m*UNITS+6)*STRIDE+UNIT_HEALTH]})}).collect::<Vec<_>>()}));
    }
    let values=teacher.values();let mut passed=deterministic&&delayed;
    let cases:Vec<_>=(0..matches).map(|m|{let v=&values[m*AUDIT_FIELDS..(m+1)*AUDIT_FIELDS];let side=m%2;let case=m/2;let expected=if case==3||case==7{2.}else if case==6{0.}else{1.};let actual=if case==7{v[4]}else{v[16+side*2]+v[17+side*2]};let ok=actual==expected && (case!=7 || v[35]+v[36]>=1.);passed&=ok;
        serde_json::json!({"name":names[case],"side":if side==0{"radiant"}else{"dire"},"expected":expected,"credited":actual,"passed":ok,"deaths":v[3],"misses":v[3]-v[4],"cancellation_commands":v[35]+v[36]})}).collect();
    let result=serde_json::json!({"passed":passed,"deterministic_same_observation":deterministic,"commands_wait_for_executor":delayed,"wall_seconds":started.elapsed().as_secs_f64(),"cases":cases,"frames":frames});
    crate::artifacts::atomic_json(&root.join("checks.json"),&result)?;
    ensure!(passed,"teacher short checks failed; see {}",root.join("checks.json").display());
    Ok(serde_json::json!({"passed":true,"cases":matches,"deterministic":deterministic,"executor_delay":delayed,"report":root.join("checks.json")}))
}
