//! GPU state layout. Every match owns its complete state; host code only launches batches.
use anyhow::{Result,ensure};
use burn::tensor::{Device,Tensor,TensorData};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use serde::Deserialize;
use std::path::Path;
use std::sync::{Mutex,OnceLock};
use crate::{contract::*,runtime::{primitive,tensor,argument}};

pub const UNITS:usize=256;
pub const STRIDE:usize=176;
pub const ITEM_RULE_FIELDS:usize=21;
pub const SHOTS:usize=512;
pub const SHOT_FIELDS:usize=12;
pub const META:usize=40;
pub const META_ATTACK_POINT_ERROR:usize=32;
pub const META_ATTACK_NOMINAL_END:usize=34;
pub const META_ATTACK_POINT_PENDING:usize=36;
pub const META_ATTACK_ANIMATION_START:usize=38;
// Self-play owns18..32; weapon state must survive outcome accumulation and
// must be reset independently of these persistent counters.
const _:()=assert!(META_ATTACK_POINT_ERROR>=32&&META_ATTACK_ANIMATION_START+2<=META);
pub const COMMAND_FIELDS:usize=9;pub const COMMAND_WORDS:usize=2*4*COMMAND_FIELDS;
pub const FACING:usize=48;pub const TEAM:usize=49;pub const KIND:usize=50;
pub const ORDER:usize=51;pub const TARGET:usize=52;pub const DEST_X:usize=53;pub const DEST_Y:usize=54;
pub const ATTACK_READY:usize=55;pub const WINDUP:usize=56;pub const WINDUP_TARGET:usize=57;
pub const CAST_END:usize=58;pub const CAST:usize=59;pub const COOLDOWNS:usize=60;
pub const FEAR_END:usize=64;pub const STACK_END:usize=65;pub const XP:usize=66;
pub const ATTACK_HAND:usize=67;
pub const DAMAGE:usize=68;pub const CREDIT:usize=69;pub const FLASK_END:usize=70;
pub const TANGO_END:usize=71;pub const CLARITY_END:usize=72;pub const TP_END:usize=73;
pub const TP_X:usize=74;pub const TP_Y:usize=75;pub const AGGRO_END:usize=76;
pub const GENERATION:usize=77;pub const BOUNTY:usize=78;pub const XP_BOUNTY:usize=79;
pub const ACTIVE:usize=80;pub const HOLD:usize=81;pub const DELIVERY:usize=82;
pub const SLOW_PERCENT:usize=83;
pub const ITEM_START:usize=84;
pub const ITEM_READY:usize=92;pub const STASH_START:usize=100;pub const COURIER_START:usize=108;
pub const STOCK_START:usize=116;pub const RESTOCK_START:usize=124;
pub const MRES_END:usize=132;pub const MRES_REDUCTION:usize=133;pub const FEAR_X:usize=134;pub const FEAR_Y:usize=135;
// >=0 means a natural lane-goal unit; -1 is a goal-less controlled NPC.
pub const LANE_POSITION:usize=136;
pub const WINDUP_GENERATION:usize=137;
pub const TANGO_TARGET:usize=138;pub const BACKPACK_START:usize=144;
pub const SLOT_START:usize=152;
pub const TARGET_GENERATION:usize=139;pub const REVEAL_END:usize=140;pub const CREDIT_TEAM:usize=141;
pub const STRUCTURE_PARENT:usize=142;pub const TOWER_TIER:usize=143;
pub const TIER1_AURA_END:usize=160;pub const HIGH_AURA_END:usize=161;
pub const FLAGBEARER:usize=162;pub const FLAG_AURA_END:usize=163;
// Attack readiness and the observed facing have distinct calibrated turn curves.
pub const ATTACK_AIM:usize=164;
pub const RESPAWN_AT:usize=165;pub const RESPAWN_RESET:usize=166;pub const DEATHS:usize=167;
pub const RELIABLE_GOLD:usize=168;pub const FREE_SCROLL:usize=169;
pub const AGGRO_READY:usize=170;pub const AGGRO_PENDING:usize=171;
pub const AGGRO_PREVIOUS:usize=172;pub const AGGRO_RECOVER:usize=173;pub const AGGRO_PREVIOUS_GENERATION:usize=174;
pub const INVULNERABLE:usize=175;
pub const BACKGROUND_START:usize=UNITS-20;
pub const FOG_START:usize=BACKGROUND_START-8;
pub const CREEP_END:usize=FOG_START;
pub const FOUNTAIN_START:usize=UNITS-2;
pub const COLLISION_CELLS:usize=32*32;
pub const COLLISION_WORDS:usize=COLLISION_CELLS+UNITS;
// Metadata: tick, episode, finished, accumulated game ticks, completed matches,
// reward potential, decision counter, error flag. Slots8..32 are metrics;
// slots32..40 are the two heroes' fractional weapon state.

#[derive(Deserialize)]
pub struct Position{pub x:f32,pub y:f32,pub z:f32}
#[derive(Deserialize)]
pub struct StaticVision{pub x:f32,pub y:f32,pub team:f32,pub day:f32,pub night:f32,#[serde(default)]pub unobstructed:bool}
#[derive(Deserialize)]
pub struct SfProjectileAnimation {
    pub cycles_per_base_second:f32,
    #[serde(default)]pub defer_initial_movement:bool,
    /// Each row is [animation cycle, forward offset, lateral offset].
    pub curves:[Vec<[f32;3]>;2],
}
#[derive(Deserialize)]
pub struct Engine {
    pub calibration:String,pub build:String,pub geometry:String,#[serde(default)]pub navigation:String,pub trees:Vec<Position>,
    pub prototypes:Vec<Vec<f32>>,pub spells:Vec<Vec<Vec<f32>>>,pub items:Vec<Vec<f32>>,pub hero:Vec<f32>,pub combat:Vec<f32>,pub experience:Vec<f32>,
    pub bases:Vec<Position>,pub spawns:Vec<Position>,pub lane:Vec<Position>,pub fountains:Vec<Position>,pub pre_game_seconds:f32,pub decision_ticks:usize,pub offset_ticks:usize,
    /// Directed movement goals for Radiant and Dire; legacy tables use a
    /// shared lane traversed in opposite directions.
    #[serde(default)]pub lane_routes:Vec<Vec<Position>>,
    #[serde(default)]pub sf_projectile_animation:Option<SfProjectileAnimation>,
    pub max_seconds:f32,pub behavior_validated:bool,pub approximations:Vec<String>,
    #[serde(default)]pub extra_vision:Vec<StaticVision>,
    #[serde(default)]pub fow_blockers:String,
    /// Measured immutable building sight in supported base regions. Empty uses rays.
    #[serde(default)]pub static_vision:String,
    #[serde(default)]pub movement_geometry:String,
}

pub struct World {
    pub state:CubeTensor,pub shots:CubeTensor,pub shot_hits:CubeTensor,pub meta:CubeTensor,pub tree_alive:CubeTensor,
    pub commands:CubeTensor,pub rewards:CubeTensor,pub terminals:CubeTensor,
    pub collision_grid:CubeTensor,pub extra_vision:CubeTensor,pub fow_blockers:CubeTensor,pub static_vision:CubeTensor,
    pub sf_projectile_animation:CubeTensor,
    pub prototypes:CubeTensor,pub spells:CubeTensor,pub items:CubeTensor,pub hero:CubeTensor,pub combat:CubeTensor,pub experience:CubeTensor,
    pub geometry:CubeTensor,pub movement_geometry:Option<CubeTensor>,pub navigation:CubeTensor,pub trees:CubeTensor,pub tree_map:CubeTensor,pub bases:CubeTensor,pub spawns:CubeTensor,pub lane:CubeTensor,pub fountains:CubeTensor,
    pub matches:usize,pub tree_count:usize,pub lane_lengths:[usize;2],pub engine:Engine,pub device:Device,pub fingerprint:String,
}

fn data(values:Vec<f32>,d:&Device)->CubeTensor {let n=values.len();primitive(Tensor::<1>::from_data(TensorData::new(values,[n]),d))}

// A process uses one GPU. Sharing the immutable atlas also prevents repeated
// validation worlds from retaining one128MB upload each in the device pool.
// Keep only the latest content/device key: a changed map cannot grow this cache.
static NAVIGATION_CACHE:OnceLock<Mutex<Option<(String,CubeTensor)>>>=OnceLock::new();
fn navigation_data(bytes:&[u8],hash:&str,d:&Device)->CubeTensor {
    let key=format!("{d:?}:{hash}");let mut cache=NAVIGATION_CACHE.get_or_init(||Mutex::new(None)).lock().unwrap();
    if let Some((cached,tensor))=&*cache {if *cached==key{return tensor.clone();}}
    let navigation=data(bytes.chunks_exact(4).map(|b|f32::from_bits(u32::from_le_bytes(b.try_into().unwrap()))).collect(),d);
    *cache=Some((key,navigation.clone()));navigation
}

fn tree_map(trees:&[Position])->Vec<f32>{
    let mut ids=vec![0f32;512*512];let mut distance=vec![f32::INFINITY;512*512];
    for (i,t) in trees.iter().enumerate(){
        let x0=((t.x-128.+8192.)/32.).floor().clamp(0.,511.) as usize;let x1=((t.x+128.+8192.)/32.).floor().clamp(0.,511.) as usize;
        let y0=((t.y-128.+8192.)/32.).floor().clamp(0.,511.) as usize;let y1=((t.y+128.+8192.)/32.).floor().clamp(0.,511.) as usize;
        for y in y0..=y1 {for x in x0..=x1 {let dx=-8192.+(x as f32+0.5)*32.-t.x;let dy=-8192.+(y as f32+0.5)*32.-t.y;let d=dx*dx+dy*dy;let at=y*512+x;
            if d<distance[at] {distance[at]=d;ids[at]=(i+1) as f32;}
        }}
    }ids
}

impl World {
    pub fn load(path:&Path,matches:usize,d:&Device)->Result<Self>{
        let engine:Engine=serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(matches>0 && matches<=10000,"invalid match count");
        ensure!(engine.decision_ticks==4 && engine.offset_ticks==6,"unsupported decision clock");
        ensure!(engine.prototypes.len()==29 && engine.prototypes.iter().all(|x|x.len()==UNIT_COUNT),"invalid prototypes; regenerate engine.json");
        ensure!(engine.extra_vision.len()<=64 && engine.extra_vision.iter().all(|v|[v.x,v.y,v.team,v.day,v.night].iter().all(|x|x.is_finite()) && (v.team==0. || v.team==1.) && v.day>=0. && v.night>=0.),"invalid static vision sources");
        ensure!(engine.items.len()==8 && engine.items.iter().all(|x|x.len()==ITEM_RULE_FIELDS),"invalid item tables; regenerate engine.json");
        ensure!(engine.hero.len()>=33 && engine.combat.len()==25 && engine.experience.len()==30 && engine.spawns.len()==8 && engine.fountains.len()==2 && engine.lane.len()>=2,"incomplete engine tables; regenerate engine.json");
        ensure!(engine.lane_routes.is_empty() || (engine.lane_routes.len()==2 && engine.lane_routes.iter().all(|r|r.len()>=2 && r.iter().all(|p|p.x.is_finite() && p.y.is_finite() && p.z.is_finite()))),"two finite directed lane routes required");
        let mut projectile_animation=vec![0.;8];
        if let Some(profile)=&engine.sf_projectile_animation {
            ensure!(profile.cycles_per_base_second.is_finite() && profile.cycles_per_base_second>0.,"invalid SF animation rate");
            for curve in &profile.curves {
                ensure!((2..=32).contains(&curve.len()) && curve.iter().all(|p|p.iter().all(|v|v.is_finite()) && p[0]>=0. && p[0]<=1.) && curve.windows(2).all(|p|p[0][0]<p[1][0]),"invalid SF attachment curve");
            }
            projectile_animation[0]=1.;projectile_animation[1]=if profile.defer_initial_movement{1.}else{0.};
            projectile_animation[2]=profile.cycles_per_base_second;projectile_animation[3]=profile.curves[0].len()as f32;projectile_animation[4]=profile.curves[1].len()as f32;
            projectile_animation.extend(profile.curves.iter().flatten().flatten().copied());
        }
        let sf_projectile_animation=data(projectile_animation,d);
        let calibration=if !cfg!(windows) && engine.calibration.as_bytes().get(1)==Some(&b':') {
            std::path::PathBuf::from(format!("/mnt/{}/{}",engine.calibration[..1].to_lowercase(),engine.calibration[3..].replace('\\',"/")))
        } else {std::path::PathBuf::from(&engine.calibration)};
        let bytes=std::fs::read(calibration.join(&engine.geometry))?;
        ensure!(bytes.len()==512*512*2*4,"geometry is incomplete");
        let geometry=data(bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect(),d);
        // Bot API passability is an observation feature, not the installed
        // movement grid: native heroes can walk across API-false locations.
        let (movement_geometry,movement_hash)=if engine.movement_geometry.is_empty(){(None,"legacy-api".to_string())}else{
            let file=calibration.join(&engine.movement_geometry);let physical=std::fs::read(&file)?;ensure!(physical.len()==512*512*2*4,"movement geometry incomplete");
            ensure!(physical.chunks_exact(8).zip(bytes.chunks_exact(8)).all(|(p,o)|p[..4]==o[..4] && matches!(f32::from_le_bytes(p[4..8].try_into().unwrap()),0.0|1.0)),"movement geometry must retain observation heights and binary walkability");
            (Some(data(physical.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect(),d)),crate::runtime::file_sha256(&file)?)
        };
        ensure!(!engine.navigation.is_empty(),"build navigation before loading the environment");
        let bytes=std::fs::read(calibration.join(&engine.navigation))?;
        ensure!(bytes.len()==crate::navigation::GOALS*crate::navigation::GOALS*crate::navigation::WORDS_PER_GOAL*4,"navigation is incomplete; rebuild the64-unit navigation table");
        let navigation_hash=crate::runtime::bytes_sha256(&bytes);
        let navigation=navigation_data(&bytes,&navigation_hash,d);
        let (fow_blockers,fow_hash)=if engine.fow_blockers.is_empty(){
            (data(vec![0.;512*512],d),"none".to_string())
        }else{
            let file=calibration.join(&engine.fow_blockers);let bytes=std::fs::read(&file)?;ensure!(bytes.len()==512*512*4,"incomplete authored FOW map");
            (data(bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect(),d),crate::runtime::file_sha256(&file)?)
        };
        let kernels=concat!(include_str!("physics.rs"),include_str!("observe.rs"),include_str!("world.rs"));
        let (static_vision,static_hash)=if engine.static_vision.is_empty(){
            (data(vec![0.;512*512],d),"rays".to_string())
        }else{
            let file=calibration.join(&engine.static_vision);let bytes=std::fs::read(&file)?;
            ensure!(bytes.len()==512*512*4,"incomplete static vision map");
            let words:Vec<u32>=bytes.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect();
            ensure!(words.iter().all(|v|*v<4096 && (v&63)&!((v>>6)&63)==0),"static sight must have phase/team validity bits");
            (data(words.into_iter().map(f32::from_bits).collect(),d),crate::runtime::file_sha256(&file)?)
        };
        let fingerprint=format!("{}:{}:{}:{}:{}:{}:{}",crate::runtime::file_sha256(path)?,crate::runtime::file_sha256(&calibration.join(&engine.geometry))?,navigation_hash,fow_hash,movement_hash,static_hash,crate::runtime::bytes_sha256(kernels.as_bytes()));
        let trees=data(engine.trees.iter().flat_map(|p|[p.x,p.y,p.z]).collect(),d);
        let tree_map=data(tree_map(&engine.trees),d);
        let bases=data(engine.bases.iter().flat_map(|p|[p.x,p.y,p.z]).collect(),d);
        let spawns=data(engine.spawns.iter().flat_map(|p|[p.x,p.y,p.z]).collect(),d);
        let (lane_values,lane_lengths)=if engine.lane_routes.is_empty(){
            (engine.lane.iter().chain(engine.lane.iter().rev()).flat_map(|p|[p.x,p.y,p.z]).collect(),[engine.lane.len();2])
        }else{
            (engine.lane_routes.iter().flatten().flat_map(|p|[p.x,p.y,p.z]).collect(),[engine.lane_routes[0].len(),engine.lane_routes[1].len()])
        };
        let lane=data(lane_values,d);
        let fountains=data(engine.fountains.iter().flat_map(|p|[p.x,p.y,p.z]).collect(),d);
        let extra_vision=data(if engine.extra_vision.is_empty(){vec![0.]}else{engine.extra_vision.iter().flat_map(|v|[v.x,v.y,v.team,v.day,v.night,if v.unobstructed{1.}else{0.}]).collect()},d);
        let zeros=|n|primitive(Tensor::<1>::zeros([n],d));let tree_count=engine.trees.len();
        let mut world=Self {
            state:zeros(matches*UNITS*STRIDE),shots:zeros(matches*SHOTS*SHOT_FIELDS),shot_hits:zeros(matches*SHOTS*8),meta:zeros(matches*META),
            tree_alive:primitive(Tensor::<1>::ones([matches*tree_count],d)),commands:zeros(matches*COMMAND_WORDS),
            rewards:zeros(matches*2),terminals:zeros(matches*2),
            collision_grid:zeros(matches*COLLISION_WORDS),extra_vision,fow_blockers,static_vision,sf_projectile_animation,
            prototypes:data(engine.prototypes.iter().flatten().copied().collect(),d),
            spells:data(engine.spells.iter().flatten().flatten().copied().collect(),d),
            items:data(engine.items.iter().flatten().copied().collect(),d),hero:data(engine.hero.clone(),d),combat:data(engine.combat.clone(),d),experience:data(engine.experience.clone(),d),
            geometry,movement_geometry,navigation,trees,tree_map,bases,spawns,lane,fountains,matches,tree_count,lane_lengths,engine,device:d.clone(),fingerprint,
        };
        world.reset();Ok(world)
    }
    pub fn reset(&mut self){
        reset::launch(&self.state.client,CubeCount::Static(self.matches.div_ceil(32) as u32,1,1),CubeDim::new_1d(32),
            argument(&self.state),argument(&self.meta),argument(&self.prototypes),self.matches,self.engine.pre_game_seconds);
    }
    pub fn snapshot(&self)->Tensor<2>{tensor::<1>(self.state.clone()).reshape([self.matches,UNITS*STRIDE])}
    pub fn lane_start(&self,side:usize)->usize{if side==0{0}else{self.lane_lengths[0]}}
}

#[cube]
pub fn cell(x:f32,y:f32)->usize {
    let gx=((x+8192.0f32)/32.0f32).floor().clamp(0.0f32,511.0f32) as usize;
    let gy=((y+8192.0f32)/32.0f32).floor().clamp(0.0f32,511.0f32) as usize;
    (gy*512+gx)*2
}

#[cube]
pub fn ground_z(height_level:f32)->f32 {
    // Train-plane modes 1/2/3/4 -> 384/256/128/0; independently checked
    // against held-out native positions in record-height-rules.json.
    (4.0f32-height_level)*128.0f32
}

#[cube]
pub fn higher_ground(height_level:f32,origin_level:f32)->bool {height_level<origin_level}

#[cube]
pub fn wrap_angle(angle:f32)->f32 {angle-((angle+3.1415927f32)/6.2831855f32).floor()*6.2831855f32}

#[cube]
pub fn tower_armor(state:&mut [f32],p:usize,time:f32)->f32 {
    if state[p+HIGH_AURA_END]>time {5.0f32}else if state[p+TIER1_AURA_END]>time {3.0f32}else{0.0f32}
}

#[cube]
pub fn tower_regen(state:&mut [f32],p:usize,time:f32)->f32 {
    if state[p+HIGH_AURA_END]>time {3.0f32}else if state[p+TIER1_AURA_END]>time {1.0f32}else{0.0f32}
}

#[cube]
pub fn effective_armor(state:&mut [f32],hero_rules:&[f32],base:usize,p:usize,time:f32)->f32 {
    let mut armor=state[p+UNIT_ARMOR];
    if state[p+KIND]==0.0f32 {armor+=tower_armor(state,p,time);}
    if state[p+KIND]!=1.0f32 {
        for h in 0..2 {let a=base+h*STRIDE;let level=state[a+UNIT_LEVEL];let dx=state[a+UNIT_X]-state[p+UNIT_X];let dy=state[a+UNIT_Y]-state[p+UNIT_Y];
            if state[a+ACTIVE]>0.0f32 && state[a+UNIT_HEALTH]>0.0f32 && state[a+TEAM]!=state[p+TEAM] && level>=2.0f32 && dx*dx+dy*dy<=1200.0f32*1200.0f32 {
                let rank=if level>=9.0f32 {3usize}else if level>=8.0f32 {2usize}else if level>=4.0f32 {1usize}else{0usize};armor+=hero_rules[7+rank];
            }
        }
    }
    armor
}

#[cube]
pub fn initialize(state:&mut [f32],prototypes:&[f32],m:usize){
    let base=m*UNITS*STRIDE;
    for i in 0..UNITS*STRIDE {state[base+i]=0.0f32;}
    for u in 0..6 {
        let p=base+u*STRIDE;
        for f in 0..UNIT_COUNT {state[p+f]=prototypes[u*UNIT_COUNT+f];}
        state[p+TEAM]=if u==0 || u==2 || u==3 {0.0f32} else {1.0f32};
        state[p+KIND]=if u<2 {0.0f32} else {1.0f32};
        state[p+ACTIVE]=1.0f32;state[p+FACING]=if u==1 {-2.3561945f32} else {0.7853982f32};
        state[p+ATTACK_AIM]=state[p+FACING];
        state[p+TARGET]=-1.0f32;state[p+WINDUP_TARGET]=-1.0f32;state[p+CREDIT]=-1.0f32;
        state[p+STRUCTURE_PARENT]=-1.0f32;
        if u>=2 {state[p+TOWER_TIER]=if u==2 || u==4 {1.0f32}else{2.0f32};state[p+INVULNERABLE]=1.0f32;}
        if u==3 || u==5 {state[p+STRUCTURE_PARENT]=(u-1) as f32;}
        if u<2 {
            for item in 0..8 {state[p+SLOT_START+item]=-1.0f32;}state[p+SLOT_START]=0.0f32;
            state[p+UNIT_GOLD]=95.0f32;state[p+ITEM_START]=1.0f32;
            // Native mode supplies a free scroll, ready at DotaTime +10s.
            state[p+ITEM_START+3]=1.0f32;state[p+SLOT_START+3]=15.0f32;state[p+ITEM_READY+3]=93.0f32;
            state[p+FREE_SCROLL]=1.0f32;
            state[p+UNIT_HEALTH_MAX]+=44.0f32;state[p+UNIT_HEALTH]=state[p+UNIT_HEALTH_MAX];
            state[p+UNIT_MANA_MAX]+=24.0f32;state[p+UNIT_MANA]=state[p+UNIT_MANA_MAX];
            state[p+UNIT_ARMOR]+=2.5833333f32;state[p+UNIT_ATTACK_DAMAGE]+=5.0f32;
            state[p+UNIT_ATTACK_INTERVAL]=1.6f32/1.36f32;
            state[p+UNIT_MAGIC_RESIST]+=0.002f32;state[p+UNIT_HEALTH_REGEN]+=0.2f32;state[p+UNIT_MANA_REGEN]+=0.1f32;
        }
    }
    for side in 0..2 {
        let p=base+(FOUNTAIN_START+side)*STRIDE;
        for f in 0..UNIT_COUNT {state[p+f]=prototypes[(9+side)*UNIT_COUNT+f];}
        state[p+TEAM]=side as f32;state[p+KIND]=1.0f32;state[p+ACTIVE]=1.0f32;state[p+TARGET]=-1.0f32;state[p+CREDIT]=-1.0f32;state[p+INVULNERABLE]=1.0f32;
    }
    for index in 0..18 {
        let unit=BACKGROUND_START+index;let p=base+unit*STRIDE;let side=index/9;let local=index%9;
        for f in 0..UNIT_COUNT {state[p+f]=prototypes[(11+index)*UNIT_COUNT+f];}
        state[p+TEAM]=side as f32;state[p+KIND]=1.0f32;state[p+ACTIVE]=1.0f32;state[p+TARGET]=-1.0f32;state[p+CREDIT]=-1.0f32;
        let mut tier=4.0f32.runtime();let mut parent=(-2.0f32).runtime();
        if local<6 {tier=(local%3+1) as f32;parent=if local%3==0 {-1.0f32}else{(unit-1) as f32};}
        if local==6 {tier=3.0f32;parent=if side==0 {3.0f32}else{5.0f32};}
        state[p+TOWER_TIER]=tier;state[p+STRUCTURE_PARENT]=parent;state[p+INVULNERABLE]=1.0f32;
    }
    // Public initial tower statistics, then each observer's last seen state.
    // Inactive cache slots participate in neither physics nor visibility.
    for observer in 0..2 {for tower in 0..4 {
        let source=base+(tower+2)*STRIDE;let target=base+(FOG_START+observer*4+tower)*STRIDE;
        for f in 0..STRIDE {state[target+f]=state[source+f];}state[target+ACTIVE]=0.0f32;
        // Unseen enemy-building flags default false in the native protobuf.
        if state[source+TEAM]!=observer as f32 {state[target+INVULNERABLE]=0.0f32;}
    }}
}

#[cube(launch)]
fn reset(state:&mut [f32],meta:&mut [f32],prototypes:&[f32],matches:usize,pregame:f32){
    let m=ABSOLUTE_POS;
    if m<matches {initialize(state,prototypes,m);reset_attack_phase(meta,m);meta[m*META]=-pregame*30.0f32;}
}

#[cube]
pub fn reset_attack_phase(meta:&mut [f32],m:usize){
    for f in META_ATTACK_POINT_ERROR..META_ATTACK_ANIMATION_START+2{meta[m*META+f]=0.0f32;}
}

#[cube]
pub fn daytime(time:f32)->bool {time>=0.016666667f32 && time<300.0166667f32}

#[cube]
pub fn vision_radius(unit:usize,kind:f32,time:f32)->f32 {
    let mut radius=750.0f32.runtime();
    // Direct GetCurrentVisionRange capture:800 in pregame,1800 after0,
    // back to800 at300.033s (dota-vision-003). Threshold ticks still use
    // the previous phase because the native fractional clock lags slightly.
    // Compare halfway between30Hz ticks; shader division can turn9000/30
    // into300.00003 and must not move sunset one tick early.
    let day=daytime(time);
    if unit<2 {radius=if day {1800.0f32}else{800.0f32};}
    if kind==1.0f32 {
        let mut tier_one=unit==2 || unit==4;
        if unit>=BACKGROUND_START && unit<FOUNTAIN_START {
            let local=(unit-BACKGROUND_START)%9;tier_one=local<6 && local%3==0;
        }
        radius=if day {1900.0f32}else if tier_one {800.0f32}else{1100.0f32};
    }
    if unit>=FOUNTAIN_START {radius=1800.0f32;}radius
}

#[cube]
pub fn distance(state:&mut [f32],p:usize,q:usize)->f32 {
    let dx=state[p+UNIT_X]-state[q+UNIT_X];let dy=state[p+UNIT_Y]-state[q+UNIT_Y];
    (dx*dx+dy*dy).sqrt()
}

#[cube]
pub fn potential(state:&mut [f32],base:usize)->f32 {
    let a=base;let b=base+STRIDE;let t1=base+2*STRIDE;let t2=base+4*STRIDE;
    ((state[a+UNIT_LAST_HITS]-state[b+UNIT_LAST_HITS])*0.01f32).clamp(-0.3f32,0.3f32)+
    ((state[a+UNIT_DENIES]-state[b+UNIT_DENIES])*0.003f32).clamp(-0.06f32,0.06f32)+
    0.1f32*(state[a+UNIT_HEALTH]/state[a+UNIT_HEALTH_MAX]-state[b+UNIT_HEALTH]/state[b+UNIT_HEALTH_MAX])+
    0.2f32*(state[t1+UNIT_HEALTH]/state[t1+UNIT_HEALTH_MAX]-state[t2+UNIT_HEALTH]/state[t2+UNIT_HEALTH_MAX])
}





