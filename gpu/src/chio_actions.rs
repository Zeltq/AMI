//! CHiO action ABI and executor. Legacy six-head actions are not reinterpreted.
use burn::tensor::{Tensor,Int,Bool};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use crate::{world::*,contract::*,observe::{self,Observed},runtime::{argument,primitive,tensor}};

pub const TYPES:usize=13;
pub const DIRECTIONS:usize=36;
pub const HEADS:usize=4; // type, direction, enemy-creep row, allied-creep row
pub const MASK_WIDTH:usize=TYPES+2*observe::CREEPS;

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
#[repr(usize)]
pub enum Action {Move,Stop,AttackEnemyCreep,Deny,AttackHero,AttackTower,RazeNear,RazeMid,RazeFar,Requiem,UseFlask,UseMango,EatTree}

pub struct Masks {pub packed:CubeTensor,pub types:Tensor<2,Bool>,pub enemy:Tensor<2,Bool>,pub ally:Tensor<2,Bool>}

#[cube(launch)]
fn masks(state:&mut [f32],meta:&[f32],commands:&[f32],obs:&[f32],refs:&[f32],out:&mut [f32],
    geometry:&[f32],blockers:&[f32],static_vision:&[u32],tree_map:&[f32],trees:&[f32],alive:&mut [f32],extra_vision:&[f32],
    actors:usize,tree_count:usize,vision_count:usize,pregame:f32){
    let a=ABSOLUTE_POS;if a<actors {
        let m=a/2;let side=a%2;let base=m*UNITS*STRIDE;let p=base+side*STRIDE;
        let row=a*observe::WIDTH;let r=a*MASK_WIDTH;let time=(meta[m*META]+pregame*30.0f32)/30.0f32;
        for i in 0..MASK_WIDTH {out[r+i]=0.0f32;}
        out[r]=1.0f32;out[r+1]=1.0f32;
        for i in 0..observe::CREEPS {
            let id=refs[a*observe::TARGETS+2+i];
            if id>=6.0f32 && id<CREEP_END as f32 && obs[row+observe::CREEP_VALID+i]>0.0f32 {
                let q=base+id as usize*STRIDE;
                if state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>0.0f32 && state[q+KIND]>=2.0f32 && state[q+KIND]<=4.0f32 {
                    if state[q+TEAM]!=side as f32 {out[r+TYPES+i]=1.0f32;out[r+2]=1.0f32;}
                    else if state[q+UNIT_HEALTH]<=0.5f32*state[q+UNIT_HEALTH_MAX] {out[r+TYPES+observe::CREEPS+i]=1.0f32;out[r+3]=1.0f32;}
                }
            }
        }
        out[r+4]=obs[row+observe::HERO_VALID+1];
        let tower=if side==0 {4usize}else{2usize};let q=base+tower*STRIDE;
        let tx=state[q+UNIT_X];let ty=state[q+UNIT_Y];
        if state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>0.0f32 && crate::observe::sight(state,geometry,blockers,static_vision,tree_map,trees,alive,extra_vision,vision_count,m,tree_count,base,side as f32,tx,ty,meta[m*META]/30.0f32) {out[r+5]=1.0f32;}
        for spell in 0..4 {out[r+6+spell]=obs[row+observe::ACTION_MASK+6+spell];}
        if state[p+ITEM_START+5]>0.0f32 && state[p+ITEM_READY+5]<=time && state[p+FLASK_END]<=time {out[r+10]=1.0f32;}
        if state[p+ITEM_START+4]>0.0f32 && state[p+ITEM_READY+4]<=time {out[r+11]=1.0f32;}
        if state[p+ITEM_START+6]>0.0f32 && state[p+ITEM_READY+6]<=time {
            for t in 0..tree_count {if alive[m*tree_count+t]>0.0f32 {out[r+12]=1.0f32;}}
        }
        let mut casting=if state[p+CAST_END]>time {state[p+CAST]as usize}else{0usize};
        // An accepted skill order already restricts the next observation while
        // it is still in the seven-tick execution queue.
        let mut latest=(-1e20f32).runtime();
        for slot in 0..4 {let k=(a*4+slot)*COMMAND_FIELDS;
            // A continuation still in flight after completion cannot start a
            // second cast on cooldown, and must not lock the mask forever.
            if state[p+CAST_END]<=time && commands[k+7]>0.0f32 && commands[k+1]>=6.0f32 && commands[k+1]<=9.0f32 && commands[k]>latest && obs[row+observe::ACTION_MASK+commands[k+1]as usize]>0.0f32 {
                casting=commands[k+1]as usize-5;latest=commands[k];
            }
        }
        if casting>0 && casting<=4 {
            for i in 0..TYPES {out[r+i]=0.0f32;}
            out[r+1]=1.0f32;out[r+5+casting]=1.0f32;
        }
        if state[p+UNIT_HEALTH]<=0.0f32 || state[p+ACTIVE]<=0.0f32 || state[p+FEAR_END]>time {
            for i in 0..TYPES {out[r+i]=0.0f32;}out[r+1]=1.0f32;
        }
    }
}

pub fn observe(w:&World,obs:&Observed)->Masks {
    assert!(w.engine.chio_direct && w.engine.decision_ticks==6);
    assert_eq!(obs.packed.dims()[1],observe::WIDTH,"CHiO masks require the base public observation");
    let n=w.matches*2;let packed=primitive(Tensor::<2>::zeros([n,MASK_WIDTH],&w.device));let base=primitive(obs.packed.clone());
    masks::launch(&w.state.client,CubeCount::Static(n.div_ceil(32)as u32,1,1),CubeDim::new_1d(32),
        argument(&w.state),argument(&w.meta),argument(&w.commands),argument(&base),argument(&obs.refs),argument(&packed),
        argument(&w.geometry),argument(&w.fow_blockers),argument(&w.static_vision),argument(&w.tree_map),argument(&w.trees),argument(&w.tree_alive),argument(&w.extra_vision),n,w.tree_count,w.engine.extra_vision.len(),w.engine.pre_game_seconds);
    let data=tensor::<2>(packed.clone());
    Masks{packed,types:data.clone().slice([0..n,0..TYPES]).greater_elem(0.),
        enemy:data.clone().slice([0..n,TYPES..TYPES+observe::CREEPS]).greater_elem(0.),
        ally:data.slice([0..n,TYPES+observe::CREEPS..MASK_WIDTH]).greater_elem(0.)}
}

#[cube(launch)]
fn enqueue_kernel(chosen:&[f32],mask:&[f32],buys:&[f32],refs:&[f32],state:&[f32],meta:&mut [f32],
    trees:&[f32],alive:&[f32],commands:&mut [f32],legacy:&mut [f32],actors:usize,tree_count:usize){
    let a=ABSOLUTE_POS;if a<actors {
        let m=a/2;let side=a%2;let base=m*UNITS*STRIDE;let p=base+side*STRIDE;
        let mut kind=chosen[a*HEADS]as usize;let direction=chosen[a*HEADS+1]as usize;
        let mut valid=kind<TYPES;
        if valid {valid=mask[a*MASK_WIDTH+kind]>0.0f32;}
        if kind==0 && direction>=DIRECTIONS {valid=false;}
        let target_row=if kind==3 {chosen[a*HEADS+3]as usize}else{chosen[a*HEADS+2]as usize};
        if kind==2 || kind==3 {
            if target_row>=observe::CREEPS {valid=false;}
            else {let offset=if kind==2 {TYPES}else{TYPES+observe::CREEPS};if mask[a*MASK_WIDTH+offset+target_row]<=0.0f32 {valid=false;}}
        }
        if !valid {kind=1;meta[m*META+7]=6.0f32;}
        let action=if kind==0 {2usize}else if kind==1 {4usize}else if kind<=5 {3usize}else if kind<=9 {kind}else{10usize};
        let item=if kind==10 {5usize}else if kind==11 {4usize}else if kind==12 {6usize}else{0usize};
        let mut target=side as f32;let mut legacy_target=0usize.runtime();
        if kind==2 || kind==3 {legacy_target=2+target_row;target=refs[a*observe::TARGETS+legacy_target];}
        if kind==4 {target=(1-side)as f32;legacy_target=1;}
        if kind==5 {let tower=if side==0{4usize}else{2usize};target=tower as f32;legacy_target=observe::CREEPS+tower;}
        if kind==12 {
            let mut best=(1e30f32).runtime();target=-1.0f32;
            for t in 0..tree_count {let dx=trees[t*3]-state[p+UNIT_X];let dy=trees[t*3+1]-state[p+UNIT_Y];let distance=dx*dx+dy*dy;
                if alive[m*tree_count+t]>0.0f32 && distance<best {best=distance;target=(UNITS+t)as f32;}
            }
            for i in 0..observe::TARGETS {if refs[a*observe::TARGETS+i]==target {legacy_target=i;}}
        }
        let slot=meta[m*META+6]as usize%4;let k=(a*4+slot)*COMMAND_FIELDS;
        let angle=direction as f32*0.1745329252f32;
        commands[k]=meta[m*META]+6.0f32;commands[k+1]=action as f32;commands[k+2]=buys[a];commands[k+3]=item as f32;
        commands[k+4]=target;
        commands[k+5]=if kind==0 {(state[p+UNIT_X]+2000.0f32*angle.cos()).clamp(-8192.0f32,8192.0f32)}else{state[p+UNIT_X]};
        commands[k+6]=if kind==0 {(state[p+UNIT_Y]+2000.0f32*angle.sin()).clamp(-8192.0f32,8192.0f32)}else{state[p+UNIT_Y]};
        commands[k+7]=1.0f32;commands[k+8]=if target>=0.0f32 && target<UNITS as f32 {state[base+target as usize*STRIDE+GENERATION]}else{0.0f32};
        // Compatibility representation is for existing telemetry/reconciliation,
        // never the CHiO label or its directional policy input.
        legacy[a*6]=action as f32;legacy[a*6+1]=buys[a];legacy[a*6+2]=item as f32;legacy[a*6+3]=legacy_target as f32;
        legacy[a*6+4]=64.0f32;legacy[a*6+5]=64.0f32;
    }
}

pub fn enqueue(w:&World,obs:&Observed,masks:&Masks,chosen:Tensor<2,Int>,buys:Tensor<1>)->Tensor<2,Int>{
    let n=w.matches*2;assert_eq!(chosen.dims(),[n,HEADS]);assert_eq!(buys.dims(),[n]);
    assert!(w.engine.chio_direct && w.engine.decision_ticks==6);
    let chosen=primitive(chosen.float());let buys=primitive(buys);let legacy=primitive(Tensor::<2>::zeros([n,6],&w.device));
    enqueue_kernel::launch(&w.state.client,CubeCount::Static(n.div_ceil(32)as u32,1,1),CubeDim::new_1d(32),
        argument(&chosen),argument(&masks.packed),argument(&buys),argument(&obs.refs),argument(&w.state),argument(&w.meta),
        argument(&w.trees),argument(&w.tree_alive),argument(&w.commands),argument(&legacy),n,w.tree_count);
    tensor::<2>(legacy).int()
}

/// Evaluation-only original v9 opponent: preserve its combat/movement orders,
/// while using the common purchase script and the arena's six-tick clock.
#[cube(launch)]
fn legacy_right_kernel(raw:&[f32],buys:&[f32],refs:&[f32],state:&[f32],meta:&[f32],commands:&mut [f32],executed:&mut [f32],actors:usize,#[comptime] swap:bool){
    let a=ABSOLUTE_POS;if a<actors {
        let m=a/2;let side=a%2;let selected=if swap && m%2==1 {1-side}else{side};
        if selected==1 {
            let p=(m*UNITS+side)*STRIDE;let slot=meta[m*META+6]as usize%4;let k=(a*4+slot)*COMMAND_FIELDS;
            for i in 0..6 {executed[a*6+i]=raw[a*6+i];commands[k+1+i]=raw[a*6+i];}executed[a*6+1]=buys[a];commands[k+2]=buys[a];
            commands[k]=meta[m*META]+6.0f32;let target=refs[a*observe::TARGETS+raw[a*6+3]as usize];commands[k+4]=target;
            commands[k+5]=state[p+UNIT_X]+(raw[a*6+4]-64.0f32)*16.0f32;commands[k+6]=state[p+UNIT_Y]+(raw[a*6+5]-64.0f32)*16.0f32;
            commands[k+7]=1.0f32;commands[k+8]=if target>=0.0f32 && target<UNITS as f32 {state[(m*UNITS+target as usize)*STRIDE+GENERATION]}else{0.0f32};
        }
    }
}
pub fn original_v9_right(w:&World,obs:&Observed,raw:Tensor<2,Int>,buys:Tensor<1>,executed:Tensor<2,Int>,swap:bool)->Tensor<2,Int>{
    let raw=primitive(raw.float());let buys=primitive(buys);let out=primitive(executed.float());let n=w.matches*2;
    legacy_right_kernel::launch(&w.state.client,CubeCount::Static(n.div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&raw),argument(&buys),argument(&obs.refs),argument(&w.state),argument(&w.meta),argument(&w.commands),argument(&out),n,swap);
    tensor::<2>(out).int()
}
