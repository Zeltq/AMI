//! CHiO causal history and geometry, appended to the unchanged honest v4 snapshot.
use burn::tensor::{Tensor,TensorData,Int};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use crate::{world::*,contract::*,observe::{self,Observed},chio_actions,runtime::{argument,primitive,tensor}};

pub const HISTORY:usize=12;
pub const UNIT_FIELDS:usize=17; // HP history, 3 own razes, attack reach, relative angle/pi
pub const HERO_FIELDS:usize=25; // plus 3 enemy razes and own cast flag/4-way skill
pub const PREVIOUS:usize=13+36+6;
pub const HERO_OFFSET:usize=crate::enemy_observe::WIDTH;
pub const CREEP_OFFSET:usize=HERO_OFFSET+4*HERO_FIELDS;
pub const TOWER_OFFSET:usize=CREEP_OFFSET+observe::CREEPS*2*UNIT_FIELDS;
pub const PREVIOUS_OFFSET:usize=TOWER_OFFSET+4*2*UNIT_FIELDS;
pub const MASK_OFFSET:usize=PREVIOUS_OFFSET+PREVIOUS;
pub const WIDTH:usize=MASK_OFFSET+chio_actions::MASK_WIDTH;
const UNIT_MEMORY:usize=1+2*HISTORY;
const MEMORY:usize=2+UNITS*UNIT_MEMORY;
pub const ITEM_IDS:[i32;6]=[0,1,2,5,4,6];

pub struct Observer {memory:CubeTensor}
impl Observer {
    pub fn new(w:&World)->Self {Self{memory:primitive(Tensor::<2>::zeros([w.matches*2,MEMORY],&w.device))}}
    pub fn extend(&self,w:&World,base:&Observed,v4:Observed,previous:Tensor<2,Int>,masks:&chio_actions::Masks)->Observed {
        assert_eq!(v4.packed.dims()[1],crate::enemy_observe::WIDTH);
        let n=w.matches*2;let extra=primitive(Tensor::<2>::zeros([n,WIDTH-HERO_OFFSET],&w.device));
        let public=primitive(base.packed.clone());let previous=primitive(previous.float());
        extend::launch(&w.state.client,CubeCount::Static(n.div_ceil(32)as u32,1,1),CubeDim::new_1d(32),
            argument(&w.state),argument(&w.meta),argument(&public),argument(&base.refs),argument(&previous),argument(&masks.packed),argument(&w.spells),
            argument(&w.geometry),argument(&w.fow_blockers),argument(&w.static_vision),argument(&w.tree_map),argument(&w.trees),argument(&w.tree_alive),argument(&w.extra_vision),
            argument(&self.memory),argument(&extra),n,w.tree_count,w.engine.extra_vision.len());
        Observed{packed:Tensor::cat(vec![v4.packed,tensor::<2>(extra)],1),refs:v4.refs,reset:v4.reset}
    }
}

#[cube]
fn zone(spells:&[f32],spell:usize,x:f32,y:f32,facing:f32,tx:f32,ty:f32)->f32 {
    let k=spell*4*8;let cx=x+spells[k+3]*facing.cos();let cy=y+spells[k+3]*facing.sin();
    let dx=tx-cx;let dy=ty-cy;if dx*dx+dy*dy<=spells[k+4]*spells[k+4] {1.0f32}else{0.0f32}
}

#[cube(launch)]
fn extend(state:&mut [f32],meta:&[f32],obs:&[f32],refs:&[f32],previous:&[f32],masks:&[f32],spells:&[f32],
    geometry:&[f32],blockers:&[f32],static_vision:&[u32],tree_map:&[f32],trees:&[f32],alive:&mut [f32],vision:&[f32],
    memory:&mut [f32],out:&mut [f32],actors:usize,tree_count:usize,vision_count:usize){
    let a=ABSOLUTE_POS;if a<actors {
        let m=a/2;let side=a%2;let base=m*UNITS*STRIDE;let own=base+side*STRIDE;let r=a*(WIDTH-HERO_OFFSET);let mem=a*MEMORY;
        let epoch=meta[m*META+1]+1.0f32;let counter=meta[m*META+6];
        if memory[mem]!=epoch {for i in 0..MEMORY {memory[mem+i]=0.0f32;}memory[mem]=epoch;}
        let fresh=memory[mem+1]!=counter+1.0f32;
        if fresh {
            for u in 0..UNITS {let h=mem+2+u*UNIT_MEMORY;
                for lag in 1..HISTORY {let i=HISTORY-lag;memory[h+1+i]=memory[h+i];memory[h+1+HISTORY+i]=memory[h+HISTORY+i];}
                memory[h+1]=0.0f32;memory[h+1+HISTORY]=0.0f32;
            }memory[mem+1]=counter+1.0f32;
        }
        for i in 0..WIDTH-HERO_OFFSET {out[r+i]=0.0f32;}
        let ox=state[own+UNIT_X];let oy=state[own+UNIT_Y];let facing=state[own+FACING];
        for slot in 0..2+observe::CREEPS+4 {
            let id=refs[a*observe::TARGETS+slot];
            if id>=0.0f32 && id<UNITS as f32 {
                let u=id as usize;let p=base+u*STRIDE;let h=mem+2+u*UNIT_MEMORY;
                let fields=if slot<2 {HERO_FIELDS}else{UNIT_FIELDS};
                let at=r+if slot<2 {slot*2*HERO_FIELDS}else if slot<2+observe::CREEPS {CREEP_OFFSET-HERO_OFFSET+(slot-2)*2*UNIT_FIELDS}else{TOWER_OFFSET-HERO_OFFSET+(slot-2-observe::CREEPS)*2*UNIT_FIELDS};
                let tx=state[p+UNIT_X];let ty=state[p+UNIT_Y];let mut seen=slot==0;
                if slot==1 {seen=obs[a*observe::WIDTH+observe::HERO_VALID+1]>0.0f32;}
                if slot>=2 && slot<2+observe::CREEPS {seen=obs[a*observe::WIDTH+observe::CREEP_VALID+slot-2]>0.0f32;}
                if slot>=2+observe::CREEPS {seen=state[p+TEAM]==side as f32 || crate::observe::sight(state,geometry,blockers,static_vision,tree_map,trees,alive,vision,vision_count,m,tree_count,base,side as f32,tx,ty,meta[m*META]/30.0f32);}
                if fresh && seen {
                    let generation=state[p+GENERATION]+1.0f32;
                    if memory[h]!=generation {for i in 0..UNIT_MEMORY {memory[h+i]=0.0f32;}memory[h]=generation;}
                    if state[p+UNIT_HEALTH_MAX]>0.0f32 {memory[h+1]=(state[p+UNIT_HEALTH]/state[p+UNIT_HEALTH_MAX]).clamp(0.0f32,1.0f32);memory[h+1+HISTORY]=1.0f32;}
                }
                for lag in 0..HISTORY {out[at+lag]=memory[h+1+lag];out[at+fields+lag]=memory[h+1+HISTORY+lag];}
                if seen {
                    for spell in 0..3 {out[at+12+spell]=zone(spells,spell,ox,oy,facing,tx,ty);out[at+fields+12+spell]=1.0f32;}
                    let dx=tx-ox;let dy=ty-oy;let distance=(dx*dx+dy*dy).sqrt();
                    let reach=state[own+UNIT_ATTACK_RANGE]+crate::physics::hull(0.0f32)+crate::physics::hull(state[p+KIND]);
                    out[at+15]=if distance<=reach {1.0f32}else{0.0f32};out[at+fields+15]=1.0f32;
                    out[at+16]=if distance>0.0f32 {crate::world::wrap_angle(dy.atan2(dx)-facing)/3.141592654f32}else{0.0f32};out[at+fields+16]=1.0f32;
                }
            }
        }
        if obs[a*observe::WIDTH+observe::HERO_VALID+1]>0.0f32 {
            let enemy=base+(1-side)*STRIDE;
            for spell in 0..3 {out[r+17+spell]=zone(spells,spell,state[enemy+UNIT_X],state[enemy+UNIT_Y],state[enemy+FACING],ox,oy);out[r+HERO_FIELDS+17+spell]=1.0f32;}
        }
        for f in 20..25 {out[r+HERO_FIELDS+f]=1.0f32;}
        if masks[a*chio_actions::MASK_WIDTH]==0.0f32 {
            for spell in 0..4 {if masks[a*chio_actions::MASK_WIDTH+6+spell]>0.0f32 {out[r+20]=1.0f32;out[r+21+spell]=1.0f32;}}
        }
        let mut kind=previous[a*chio_actions::HEADS]as usize;
        if counter==0.0f32 || state[own+UNIT_HEALTH]<=0.0f32 || (state[own+RESPAWN_RESET]>0.0f32 && state[own+RESPAWN_RESET]==counter) {kind=1;}
        let at=r+PREVIOUS_OFFSET-HERO_OFFSET;
        if kind<13 {out[at+kind]=1.0f32;}
        if kind==0 {let direction=previous[a*chio_actions::HEADS+1]as usize;if direction<36 {out[at+13+direction]=1.0f32;}}
        let target=if kind>=2 && kind<=5 {kind-1}else if kind==12 {5usize}else{0usize};out[at+49+target]=1.0f32;
        for i in 0..chio_actions::MASK_WIDTH {out[r+MASK_OFFSET-HERO_OFFSET+i]=masks[a*chio_actions::MASK_WIDTH+i];}
    }
}

pub fn initial_previous(actors:usize,d:&burn::tensor::Device)->Tensor<2,Int>{
    Tensor::<2,Int>::from_data(TensorData::new((0..actors).flat_map(|_|[1i32,0,0,0]).collect::<Vec<_>>(),[actors,4]),d)
}

pub fn columns(creeps:usize)->Vec<i32> {
    let mut cols:Vec<i32>=(HERO_OFFSET..CREEP_OFFSET+creeps*2*UNIT_FIELDS).map(|i|i as i32).collect();
    cols.extend((TOWER_OFFSET..MASK_OFFSET+13).map(|i|i as i32));
    cols.extend((MASK_OFFSET+13..MASK_OFFSET+13+creeps).map(|i|i as i32));
    cols.extend((MASK_OFFSET+13+observe::CREEPS..MASK_OFFSET+13+observe::CREEPS+creeps).map(|i|i as i32));cols
}

pub fn decode(obs:&Observed,pc:usize,pt:usize,pp:usize)->(crate::model::Observation,chio_actions::Masks){
    let n=obs.packed.dims()[0];let packed=obs.packed.clone().cast(burn::tensor::DType::F32);
    let legacy=Observed{packed:packed.clone().slice([0..n,0..HERO_OFFSET]),refs:obs.refs.clone(),reset:obs.reset.clone()};
    let mut b=legacy.decode_bucket(pc,pt,pp);
    let cut=|at:usize,len:usize|packed.clone().slice([0..n,at..at+len]);
    b.hero_combat=Tensor::cat(vec![b.hero_combat,cut(HERO_OFFSET,4*HERO_FIELDS).reshape([n,2,2*HERO_FIELDS])],2);
    b.creep_combat=Tensor::cat(vec![b.creep_combat,cut(CREEP_OFFSET,pc*2*UNIT_FIELDS).reshape([n,pc,2*UNIT_FIELDS])],2);
    b.tower_combat=Tensor::cat(vec![b.tower_combat,cut(TOWER_OFFSET,8*UNIT_FIELDS).reshape([n,4,2*UNIT_FIELDS])],2);
    let ids=Tensor::<1,Int>::from_data(ITEM_IDS,&packed.device());b.items=b.items.select(1,ids.clone());b.enemy_items=b.enemy_items.select(1,ids);
    b.previous=cut(PREVIOUS_OFFSET,PREVIOUS);
    let types=cut(MASK_OFFSET,13);let enemy=cut(MASK_OFFSET+13,pc);let ally=cut(MASK_OFFSET+13+observe::CREEPS,pc);
    let masks=chio_actions::Masks{packed:primitive(Tensor::cat(vec![types.clone(),enemy.clone(),ally.clone()],1)),types:types.greater_elem(0.),enemy:enemy.greater_elem(0.),ally:ally.greater_elem(0.)};
    b.action_mask=masks.types.clone();(b,masks)
}
