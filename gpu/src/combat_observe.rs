//! Optional GGD inputs. Relations are resolved through the production visible
//! unit list. No weapon timers, rolled shot damage, or future state are inputs.
use burn::tensor::{Tensor,TensorData};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use crate::{contract::*,world::*,observe::{self,Observed,WIDTH,GGD_WIDTH,TARGETS,CREEPS,COMBAT_UNIT_OFFSET,COMBAT_PROJECTILE_OFFSET},runtime::{primitive,tensor,argument}};

const CACHE_FIELDS:usize=4; // last observed damage, x, y, kind
const CACHE_WIDTH:usize=UNITS*CACHE_FIELDS+1;

#[cube]
fn put(out:&mut [f32],at:usize,fields:usize,f:usize,value:f32,scale:f32){out[at+f]=value/scale;out[at+fields+f]=1.0f32;}

#[cube]
fn damage(state:&mut [f32],hero:&[f32],combat:&[f32],base:usize,q:usize,kind:usize,raw:f32,time:f32)->f32{
    let armor=effective_armor(state,hero,base,q,time);
    (raw*combat[kind*5+state[q+KIND]as usize]*(1.0f32-0.06f32*armor/(1.0f32+0.06f32*armor.abs()))).ceil()
}

#[cube(launch)]
fn relations(state:&mut [f32],shots:&[f32],meta:&[f32],hero:&[f32],combat:&[f32],refs:&[f32],base_obs:&[f32],
    cache:&mut [f32],out:&mut [f32],unit_scale:&[f32],projectile_scale:&[f32],
    geometry:&[f32],fow:&[f32],static_vision:&[u32],tree_map:&[f32],trees:&[f32],tree_alive:&mut [f32],extra_vision:&[f32],
    actors:usize,tree_count:usize,vision_count:usize,pregame:f32){
    let a=ABSOLUTE_POS;
    if a<actors {
        let m=a/2;let side=a%2;let base=m*UNITS*STRIDE;let p=base+side*STRIDE;
        let row=a*WIDTH;let output=a*GGD_WIDTH;let memory=a*CACHE_WIDTH;
        let game_time=meta[m*META]/30.0f32;let time=game_time+pregame;
        if cache[memory+UNITS*CACHE_FIELDS]!=meta[m*META+1]+1.0f32 {
            for i in 0..UNITS*CACHE_FIELDS {cache[memory+i]=0.0f32;}
            cache[memory+UNITS*CACHE_FIELDS]=meta[m*META+1]+1.0f32;
        }
        let mut known=Array::<i32>::new(UNITS);let mut target=Array::<i32>::new(UNITS);
        for u in 0..UNITS {known[u]=-1;target[u]=-1;}
        known[side]=0;
        if base_obs[row+observe::HERO_VALID+1]>0.0f32 {known[1-side]=1;}
        for c in 0..CREEPS {let id=refs[a*TARGETS+2+c];if id>=0.0f32 && id<UNITS as f32 {known[id as usize]=(2+c)as i32;}}
        for t in 0..4 {let at=row+observe::TOWER_OFFSET+t*2*UNIT_COUNT;let q=base+(t+2)*STRIDE;
            if state[q+ACTIVE]>0.0f32 && (state[q+TEAM]==side as f32 || base_obs[at+UNIT_COUNT+UNIT_IS_ATTACKING]>0.0f32) {known[t+2]=(2+CREEPS+t)as i32;}
        }
        for u in 0..UNITS {if known[u]>=0 {let q=base+u*STRIDE;let c=memory+u*CACHE_FIELDS;
            cache[c]=state[q+UNIT_ATTACK_DAMAGE];cache[c+1]=state[q+UNIT_X];cache[c+2]=state[q+UNIT_Y];cache[c+3]=state[q+KIND];
            let id=state[q+TARGET]as usize;
            if state[q+TARGET]>=0.0f32 && id<UNITS && known[id]>=0 && state[base+id*STRIDE+GENERATION]==state[q+TARGET_GENERATION] {target[u]=id as i32;}
        }}
        for u in 0..UNITS {if known[u]>=0 {let q=base+u*STRIDE;let at=output+COMBAT_UNIT_OFFSET+known[u]as usize*2*COMBAT_UNIT_COUNT;
            if target[u]>=0 {let r=base+target[u]as usize*STRIDE;
                put(out,at,COMBAT_UNIT_COUNT,COMBAT_UNIT_ATTACK_TARGET_DX,state[r+UNIT_X]-state[p+UNIT_X],unit_scale[COMBAT_UNIT_ATTACK_TARGET_DX]);
                put(out,at,COMBAT_UNIT_COUNT,COMBAT_UNIT_ATTACK_TARGET_DY,state[r+UNIT_Y]-state[p+UNIT_Y],unit_scale[COMBAT_UNIT_ATTACK_TARGET_DY]);
                put(out,at,COMBAT_UNIT_COUNT,COMBAT_UNIT_ATTACK_TARGET_HEALTH,state[r+UNIT_HEALTH],unit_scale[COMBAT_UNIT_ATTACK_TARGET_HEALTH]);
            }
            let kind=state[q+KIND]as usize;let spread=if kind==0{hero[27]}else if kind==1{hero[31]}else{hero[26+kind]};
            put(out,at,COMBAT_UNIT_COUNT,COMBAT_UNIT_DAMAGE_SPREAD,spread,unit_scale[COMBAT_UNIT_DAMAGE_SPREAD]);
            let mut count=0.0f32.runtime();let mut total=0.0f32.runtime();let mut dps=0.0f32.runtime();
            for source in 0..UNITS {if known[source]>=0 && target[source]==u as i32 {
                let r=base+source*STRIDE;let raw=state[r+UNIT_ATTACK_DAMAGE];let kind=state[r+KIND]as usize;
                let dealt=damage(state,hero,combat,base,q,kind,raw,time);
                count+=1.0f32;total+=dealt;dps+=dealt/state[r+UNIT_ATTACK_INTERVAL].max(0.1f32);
            }}
            put(out,at,COMBAT_UNIT_COUNT,COMBAT_UNIT_INCOMING_ATTACKERS,count,unit_scale[COMBAT_UNIT_INCOMING_ATTACKERS]);
            put(out,at,COMBAT_UNIT_COUNT,COMBAT_UNIT_INCOMING_ATTACK_DAMAGE,total,unit_scale[COMBAT_UNIT_INCOMING_ATTACK_DAMAGE]);
            put(out,at,COMBAT_UNIT_COUNT,COMBAT_UNIT_INCOMING_ATTACK_DPS,dps,unit_scale[COMBAT_UNIT_INCOMING_ATTACK_DPS]);
            let mut projectiles=0.0f32.runtime();let mut amount=0.0f32.runtime();let mut all_known=true;
            let mut first=(1e20f32).runtime();let mut first_damage=0.0f32.runtime();let mut first_known=false;
            for s in 0..SHOTS {let k=(m*SHOTS+s)*SHOT_FIELDS;
                if shots[k]==1.0f32 && shots[k+3]==u as f32 && shots[k+4]==state[q+GENERATION] {
                    let source=shots[k+7]as usize;let c=memory+source*CACHE_FIELDS;let kind=shots[k+9]as usize;
                    let available=source<UNITS && cache[c]>0.0f32 && cache[c+3]==kind as f32;
                    let mut dealt=0.0f32.runtime();if available {let raw=cache[c];dealt=damage(state,hero,combat,base,q,kind,raw,time);}
                    let dx=state[q+UNIT_X]-shots[k+1];let dy=state[q+UNIT_Y]-shots[k+2];
                    let eta=((dx*dx+dy*dy).sqrt()-8.0f32).max(0.0f32)/shots[k+5].max(1.0f32);
                    if eta<first {first=eta;first_damage=dealt;first_known=available;}
                    projectiles+=1.0f32;amount+=dealt;all_known=all_known && available;
                }
            }
            put(out,at,COMBAT_UNIT_COUNT,COMBAT_UNIT_INCOMING_PROJECTILES,projectiles,unit_scale[COMBAT_UNIT_INCOMING_PROJECTILES]);
            if all_known {put(out,at,COMBAT_UNIT_COUNT,COMBAT_UNIT_INCOMING_PROJECTILE_DAMAGE,amount,unit_scale[COMBAT_UNIT_INCOMING_PROJECTILE_DAMAGE]);}
            if projectiles>0.0f32 {
                put(out,at,COMBAT_UNIT_COUNT,COMBAT_UNIT_NEXT_PROJECTILE_ETA,first,unit_scale[COMBAT_UNIT_NEXT_PROJECTILE_ETA]);
                if first_known {put(out,at,COMBAT_UNIT_COUNT,COMBAT_UNIT_NEXT_PROJECTILE_DAMAGE,first_damage,unit_scale[COMBAT_UNIT_NEXT_PROJECTILE_DAMAGE]);}
            }
        }}
        let mut index=0usize;
        for s in 0..SHOTS {let k=(m*SHOTS+s)*SHOT_FIELDS;
            if shots[k]==1.0f32 {let id=shots[k+3]as usize;let q=base+id*STRIDE;
                let mut seen=state[q+TEAM]==side as f32 || known[id]>=0;
                // Match the production projectile ordering even for visible
                // background structures not represented in the target list.
                if !seen && state[q+KIND]==1.0f32 {
                    let x=state[q+UNIT_X];let y=state[q+UNIT_Y];
                    seen=observe::sight(state,geometry,fow,static_vision,tree_map,trees,tree_alive,extra_vision,vision_count,m,tree_count,base,side as f32,x,y,game_time);
                }
                if seen && index<SHOTS {let at=output+COMBAT_PROJECTILE_OFFSET+index*2*COMBAT_PROJECTILE_COUNT;
                    if known[id]>=0 {
                        put(out,at,COMBAT_PROJECTILE_COUNT,COMBAT_PROJECTILE_TARGET_DX,state[q+UNIT_X]-state[p+UNIT_X],projectile_scale[COMBAT_PROJECTILE_TARGET_DX]);
                        put(out,at,COMBAT_PROJECTILE_COUNT,COMBAT_PROJECTILE_TARGET_DY,state[q+UNIT_Y]-state[p+UNIT_Y],projectile_scale[COMBAT_PROJECTILE_TARGET_DY]);
                        put(out,at,COMBAT_PROJECTILE_COUNT,COMBAT_PROJECTILE_TARGET_HEALTH,state[q+UNIT_HEALTH],projectile_scale[COMBAT_PROJECTILE_TARGET_HEALTH]);
                        put(out,at,COMBAT_PROJECTILE_COUNT,COMBAT_PROJECTILE_TARGET_TEAM_RELATIVE,if state[q+TEAM]==side as f32{1.0f32}else{-1.0f32},projectile_scale[COMBAT_PROJECTILE_TARGET_TEAM_RELATIVE]);
                    }
                    let source=shots[k+7]as usize;let c=memory+source*CACHE_FIELDS;let kind=shots[k+9];
                    if source<UNITS && cache[c]>0.0f32 && cache[c+3]==kind {
                        put(out,at,COMBAT_PROJECTILE_COUNT,COMBAT_PROJECTILE_CASTER_DAMAGE,cache[c],projectile_scale[COMBAT_PROJECTILE_CASTER_DAMAGE]);
                        put(out,at,COMBAT_PROJECTILE_COUNT,COMBAT_PROJECTILE_CASTER_KIND,kind,projectile_scale[COMBAT_PROJECTILE_CASTER_KIND]);
                        if known[source]>=0 {
                            put(out,at,COMBAT_PROJECTILE_COUNT,COMBAT_PROJECTILE_CASTER_DX,cache[c+1]-state[p+UNIT_X],projectile_scale[COMBAT_PROJECTILE_CASTER_DX]);
                            put(out,at,COMBAT_PROJECTILE_COUNT,COMBAT_PROJECTILE_CASTER_DY,cache[c+2]-state[p+UNIT_Y],projectile_scale[COMBAT_PROJECTILE_CASTER_DY]);
                        }
                    }
                    index+=1;
                }
            }
        }
    }
}

pub struct CombatObserver {cache:CubeTensor,unit_scale:CubeTensor,projectile_scale:CubeTensor}
impl CombatObserver {
    pub fn new(w:&World)->Self {
        let data=|v:Vec<f32>|{let n=v.len();primitive(Tensor::<1>::from_data(TensorData::new(v,[n]),&w.device))};
        Self{cache:primitive(Tensor::<1>::zeros([w.matches*2*CACHE_WIDTH],&w.device)),unit_scale:data(COMBAT_UNIT_SCALES.to_vec()),projectile_scale:data(COMBAT_PROJECTILE_SCALES.to_vec())}
    }
    pub fn enrich(&self,w:&World,observed:&Observed)->Observed {
        assert_eq!(observed.packed.dims()[1],WIDTH);
        let n=w.matches*2;let base=observed.packed.clone().cast(burn::tensor::DType::F32);
        let packed=primitive(Tensor::cat(vec![base.clone(),Tensor::zeros([n,GGD_WIDTH-WIDTH],&w.device)],1));let base=primitive(base);
        relations::launch(&w.state.client,CubeCount::Static(n.div_ceil(32)as u32,1,1),CubeDim::new_1d(32),
            argument(&w.state),argument(&w.shots),argument(&w.meta),argument(&w.hero),argument(&w.combat),argument(&observed.refs),argument(&base),
            argument(&self.cache),argument(&packed),argument(&self.unit_scale),argument(&self.projectile_scale),
            argument(&w.geometry),argument(&w.fow_blockers),argument(&w.static_vision),argument(&w.tree_map),argument(&w.trees),argument(&w.tree_alive),argument(&w.extra_vision),
            n,w.tree_count,w.engine.extra_vision.len(),w.engine.pre_game_seconds);
        Observed{packed:tensor(packed),refs:observed.refs.clone(),reset:observed.reset.clone()}
    }
}
