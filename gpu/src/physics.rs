use burn_cubecl::tensor::CubeTensor;
// Fixed 30 Hz, non-differentiable device simulation. No host match stepping.
use cubecl::prelude::*;
use crate::{world::*,contract::*,runtime::{argument,primitive,tensor}};
use burn::tensor::{Tensor,Int};

#[cube]
fn random(rng:&mut [u32],m:usize)->f32{
    let mut x=rng[m];x=x^(x<<13);x=x^(x>>17);x=x^(x<<5);rng[m]=x;
    (x>>8) as f32/16777216.0f32
}

#[cube]
fn tree_blocked(tree_map:&[f32],trees:&[f32],alive:&mut [f32],offset:usize,x:f32,y:f32,radius:f32)->bool {
    let id=tree_map[cell(x,y)/2] as usize;let mut blocked=false;
    if id>0 {let t=id-1;let dx=x-trees[t*3];let dy=y-trees[t*3+1];blocked=alive[offset+t]>0.0f32 && dx*dx+dy*dy<radius*radius;}blocked
}

#[cube]
fn visible(state:&mut [f32],geometry:&[f32],fow_blockers:&[f32],static_vision:&[u32],tree_map:&[f32],trees:&[f32],tree_alive:&mut [f32],extra_vision:&[f32],vision_count:usize,m:usize,tree_count:usize,base:usize,side:f32,q:usize,time:f32,timer:f32)->bool{
    let revealed=state[q+TEAM]!=side && state[q+REVEAL_END]>timer;
    let x=state[q+UNIT_X];let y=state[q+UNIT_Y];
    revealed || crate::observe::sight(state,geometry,fow_blockers,static_vision,tree_map,trees,tree_alive,extra_vision,vision_count,m,tree_count,base,side,x,y,time)
}

#[cube]
pub fn hull(kind:f32)->f32 {
    // Native bounding_radius in train match-0001-attempt-09: SF24, tower144,
    // melee16, ranged8, siege16 (first siege at DotaTime300).
    if kind==0.0f32 {24.0f32}else if kind==1.0f32 {144.0f32}else if kind==3.0f32 {8.0f32}else{16.0f32}
}

#[cube]
pub fn attack_reach(state:&mut [f32],p:usize,q:usize)->f32 {
    state[p+UNIT_ATTACK_RANGE]+hull(state[p+KIND])+hull(state[q+KIND])
}

#[cube]
fn creep_orders_ready(state:&mut [f32],p:usize,time:f32)->bool {
    // Controlled native siege016: target can change after2.4s of its3s
    // attack cycle, while weapon cooldown continues. Melee uses its full1s.
    let interval=state[p+UNIT_ATTACK_INTERVAL];
    let lock=if state[p+KIND]==4.0f32 {interval*0.8f32}else{interval};
    state[p+WINDUP]==0.0f32 && time>=state[p+ATTACK_READY]-interval+lock
}

#[cube]
pub fn observed_attack_target(state:&mut [f32],base:usize,p:usize,time:f32)->f32 {
    let mut result=(-1.0f32).runtime();let target=state[p+WINDUP_TARGET] as usize;
    let active=state[p+WINDUP]>0.0f32 || (state[p+ATTACK_READY]>time && state[p+TARGET]==state[p+WINDUP_TARGET]);
    if state[p+TARGET]>=0.0f32 && state[p+WINDUP_TARGET]>=0.0f32 && active && target<UNITS {
        let q=base+target*STRIDE;
        if state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>0.0f32 && state[q+GENERATION]==state[p+WINDUP_GENERATION] {result=target as f32;}
    }
    result
}

#[cube]
fn collision_cell(x:f32,y:f32)->usize {
    let x=((x+8192.0f32)/512.0f32).floor().clamp(0.0f32,31.0f32) as usize;
    let y=((y+8192.0f32)/512.0f32).floor().clamp(0.0f32,31.0f32) as usize;y*32+x
}

#[cube]
fn build_collision_grid(state:&mut [f32],grid:&mut [u32],m:usize,base:usize){
    let offset=m*COLLISION_WORDS;
    for cell in 0..COLLISION_CELLS {grid[offset+cell]=0u32;}
    for u in 0..UNITS {let p=base+u*STRIDE;
        if state[p+ACTIVE]>0.0f32 && state[p+UNIT_HEALTH]>0.0f32 {
            let cell=collision_cell(state[p+UNIT_X],state[p+UNIT_Y]);
            grid[offset+COLLISION_CELLS+u]=grid[offset+cell];grid[offset+cell]=(u+1) as u32;
        }
    }
}

#[cube]
fn body_blocked(state:&mut [f32],grid:&mut [u32],m:usize,p:usize,x:f32,y:f32)->bool {
    let base=m*UNITS*STRIDE;let offset=m*COLLISION_WORDS;let cell=collision_cell(x,y);let mut blocked=false;
    // Cells are 512 units wide, larger than any hull pair plus one tick's
    // movement. Rebuild once per physics tick; neighbouring cells cover movers.
    for neighbor in 0..9 {
        let cx=cell as i32%32+neighbor as i32%3-1;let cy=cell as i32/32+neighbor as i32/3-1;
        if cx>=0 && cx<32 && cy>=0 && cy<32 {
            let mut next=grid[offset+(cy*32+cx) as usize];
            while next>0u32 {
                let unit=next as usize-1;let q=base+unit*STRIDE;
                if p!=q && state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>0.0f32 {
                    let radius=hull(state[p+KIND])+hull(state[q+KIND]);let dx=x-state[q+UNIT_X];let dy=y-state[q+UNIT_Y];
                    let ox=state[p+UNIT_X]-state[q+UNIT_X];let oy=state[p+UNIT_Y]-state[q+UNIT_Y];
                    // Permit separation after overlapping spawns/teleports.
                    if dx*dx+dy*dy<radius*radius && dx*dx+dy*dy<ox*ox+oy*oy {blocked=true;}
                }
                next=grid[offset+COLLISION_CELLS+unit];
            }
        }
    }blocked
}

#[cube]
fn move_unit(state:&mut [f32],grid:&mut [u32],geometry:&[f32],navigation:&[u32],tree_map:&[f32],trees:&[f32],tree_alive:&mut [f32],m:usize,tree_count:usize,p:usize,x:f32,y:f32,dt:f32,turn:f32,pathfind:bool){
    // Native SF movement captures: about24.3deg/tick at configured turn rate0.9,
    // with translation beginning before facing fully aligns. This30Hz model
    // approximates the initial yaw acceleration. Controlled lane captures give
    //19.2deg/tick for creeps; their attack starts independently of visible yaw.
    let turn=if state[p+KIND]>=2.0f32 {0.3351032f32}else if state[p+KIND]==0.0f32 {turn*0.4712389f32}else{turn};
    let mut dest_x=x;let mut dest_y=y;
    let origin_cell=cell(state[p+UNIT_X],state[p+UNIT_Y]);let mut recovering=false;
    // Native creeps can spawn inside a cell whose sampled centre is blocked.
    // Walk to a nearby clear point at normal speed, permitting motion within
    // this initial cell. Entering another blocked cell remains forbidden.
    if geometry[origin_cell+1]<0.5f32 {
        let index=origin_cell/2;let cx=index%512;let cy=index/512;let mut best=(1e20f32).runtime();
        let radius=if state[p+KIND]==0.0f32 {48.0f32}else if state[p+KIND]==4.0f32 {64.0f32}else{40.0f32};
        for candidate in 0..25 {
            let gx=(cx as i32+candidate as i32%5-2).clamp(0,511) as usize;let gy=(cy as i32+candidate as i32/5-2).clamp(0,511) as usize;
            let px=-8192.0f32+(gx as f32+0.5f32)*32.0f32;let py=-8192.0f32+(gy as f32+0.5f32)*32.0f32;
            let dx=px-state[p+UNIT_X];let dy=py-state[p+UNIT_Y];let distance=dx*dx+dy*dy;
            if geometry[(gy*512+gx)*2+1]>0.5f32 && distance<best && !tree_blocked(tree_map,trees,tree_alive,m*tree_count,px,py,radius) {
                let mut clear=true;
                for i in 1..9 {let f=i as f32/8.0f32;let at=cell(state[p+UNIT_X]+dx*f,state[p+UNIT_Y]+dy*f);
                    if at!=origin_cell && geometry[at+1]<0.5f32 {clear=false;}
                }
                if clear {best=distance;dest_x=px;dest_y=py;recovering=true;}
            }
        }
    }
    if pathfind && !recovering {
        let mut direct=true;
        for step in 1..33 {
            let f=step as f32/32.0f32;
            let sx=state[p+UNIT_X]+(x-state[p+UNIT_X])*f;let sy=state[p+UNIT_Y]+(y-state[p+UNIT_Y])*f;
            if geometry[cell(sx,sy)+1]<0.5f32 || tree_blocked(tree_map,trees,tree_alive,m*tree_count,sx,sy,48.0f32) {direct=false;}
        }
        if !direct {
            let gx=((x+8192.0f32)/256.0f32).floor().clamp(0.0f32,63.0f32) as usize;
            let gy=((y+8192.0f32)/256.0f32).floor().clamp(0.0f32,63.0f32) as usize;
            let nx=((state[p+UNIT_X]+8192.0f32)/64.0f32).floor().clamp(0.0f32,255.0f32) as usize;
            let ny=((state[p+UNIT_Y]+8192.0f32)/64.0f32).floor().clamp(0.0f32,255.0f32) as usize;
            let node=ny*256+nx;let direction=(navigation[(gy*64+gx)*8192+node/8]>>((node%8)*4) as u32)&15u32;
            if direction<8u32 {
                let dx=if direction==0u32 || direction==1u32 || direction==7u32 {1.0f32} else if direction>=3u32 && direction<=5u32 {-1.0f32} else {0.0f32};
                let dy=if direction>=1u32 && direction<=3u32 {1.0f32} else if direction>=5u32 {-1.0f32} else {0.0f32};
                dest_x=-8192.0f32+(nx as f32+0.5f32+dx)*64.0f32;dest_y=-8192.0f32+(ny as f32+0.5f32+dy)*64.0f32;
                // Pull a short flow path straight where the fine movement grid
                // permits it. Avoid steering through every64-unit cell centre.
                let mut follow_x=(nx as i32+dx as i32).clamp(0,255) as usize;
                let mut follow_y=(ny as i32+dy as i32).clamp(0,255) as usize;
                let mut following=true;
                for _lookahead in 0..6 {
                    if following {
                        let next=follow_y*256+follow_x;
                        let flow=(navigation[(gy*64+gx)*8192+next/8]>>((next%8)*4) as u32)&15u32;
                        if flow<8u32 {
                            let fx=if flow==0u32 || flow==1u32 || flow==7u32 {1i32}else if flow>=3u32 && flow<=5u32{-1i32}else{0i32};
                            let fy=if flow>=1u32 && flow<=3u32 {1i32}else if flow>=5u32{-1i32}else{0i32};
                            follow_x=(follow_x as i32+fx).clamp(0,255)as usize;follow_y=(follow_y as i32+fy).clamp(0,255)as usize;
                            let px=-8192.0f32+(follow_x as f32+0.5f32)*64.0f32;let py=-8192.0f32+(follow_y as f32+0.5f32)*64.0f32;
                            let mut clear=true;
                            for sample in 1..33 {let f=sample as f32/32.0f32;let sx=state[p+UNIT_X]+(px-state[p+UNIT_X])*f;let sy=state[p+UNIT_Y]+(py-state[p+UNIT_Y])*f;
                                if geometry[cell(sx,sy)+1]<0.5f32 || tree_blocked(tree_map,trees,tree_alive,m*tree_count,sx,sy,48.0f32) {clear=false;}
                            }
                            if clear {dest_x=px;dest_y=py;}
                        }else{following=false;}
                    }
                }
            } else if direction==9u32 {
                // A valid fine-grid position can lie in a coarse cell whose
                // center is blocked. Recover to a nearby reachable flow node.
                let mut best=(512.0f32*512.0f32).runtime();
                for candidate in 0..25 {
                    let cx=(nx as i32+candidate as i32%5-2).clamp(0,255) as usize;
                    let cy=(ny as i32+candidate as i32/5-2).clamp(0,255) as usize;let node=cy*256+cx;
                    let flow=(navigation[(gy*64+gx)*8192+node/8]>>((node%8)*4) as u32)&15u32;
                    let px=-8192.0f32+(cx as f32+0.5f32)*64.0f32;let py=-8192.0f32+(cy as f32+0.5f32)*64.0f32;
                    let dx=px-state[p+UNIT_X];let dy=py-state[p+UNIT_Y];let distance=dx*dx+dy*dy;
                    if flow<9u32 && distance<best {
                        let mut clear=true;
                        for i in 1..17 {let f=i as f32/16.0f32;let sx=state[p+UNIT_X]+dx*f;let sy=state[p+UNIT_Y]+dy*f;
                            if geometry[cell(sx,sy)+1]<0.5f32 || tree_blocked(tree_map,trees,tree_alive,m*tree_count,sx,sy,48.0f32) {clear=false;}}
                        if clear {best=distance;dest_x=px;dest_y=py;}
                    }
                }
            }
        }
    }
    let dx=dest_x-state[p+UNIT_X];let dy=dest_y-state[p+UNIT_Y];let length=(dx*dx+dy*dy).sqrt();
    if length>1.0f32 {
        let desired=dy.atan2(dx);let current=state[p+FACING];
        let delta=wrap_angle(desired-current);
        state[p+FACING]=wrap_angle(current+delta.clamp(-turn,turn));
        if state[p+KIND]!=1.0f32 {state[p+ATTACK_AIM]=state[p+FACING];}
        // SF and controlled lane creeps start translating once the pre-turn
        // heading error is below135deg; their visible yaw continues to catch up.
        let can_move=if state[p+KIND]!=1.0f32 {delta.abs()<2.3561945f32}else{f32::max(0.0f32,delta.abs()-turn)<0.2f32};
        if can_move {
            let step=f32::min(length,state[p+UNIT_MOVE_SPEED]*dt);
            let mut moved=false;
            for candidate in 0..9 {
                let rank=(candidate+1)/2;
                let sign=if candidate%2==0 {1.0f32} else {-1.0f32};
                let angle=desired+rank as f32*0.3926991f32*sign;
                let nx=state[p+UNIT_X]+angle.cos()*step;let ny=state[p+UNIT_Y]+angle.sin()*step;
                let radius=if state[p+KIND]==0.0f32 {48.0f32}else if state[p+KIND]==4.0f32 {64.0f32}else{40.0f32};
                let destination_cell=cell(nx,ny);let clear=geometry[destination_cell+1]>0.5f32 || (recovering && destination_cell==origin_cell);
                if !moved && clear && !tree_blocked(tree_map,trees,tree_alive,m*tree_count,nx,ny,radius) && !body_blocked(state,grid,m,p,nx,ny) {
                    state[p+UNIT_X]=nx;state[p+UNIT_Y]=ny;state[p+UNIT_Z]=ground_z(geometry[cell(nx,ny)]);
                    moved=true;
                }
            }
        }
    }
}

#[cube]
fn hit(state:&mut [f32],hero:&[f32],combat:&[f32],base:usize,source:usize,target:usize,amount:f32,magical:bool,time:f32,source_kind:f32,source_team:f32){
    let q=base+target*STRIDE;
    if state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>state[q+DAMAGE] {
        let armor=effective_armor(state,hero,base,q,time);
        let resistance=state[q+UNIT_MAGIC_RESIST]-if state[q+MRES_END]>time {state[q+MRES_REDUCTION]} else {0.0f32};
        let reduction=if magical {resistance} else {0.06f32*armor/(1.0f32+0.06f32*armor.abs())};
        let multiplier=if magical {1.0f32}else{combat[source_kind as usize*5+state[q+KIND] as usize]};
        // Tier 2 cannot be damaged while its own Tier 1 stands.
        let mut invulnerable=target>=FOUNTAIN_START || state[q+INVULNERABLE]>0.0f32;
        if state[q+KIND]==1.0f32 {
            let parent=state[q+STRUCTURE_PARENT];
            if parent>=0.0f32 && state[base+parent as usize*STRIDE+ACTIVE]>0.0f32 {invulnerable=true;}
            if parent== -2.0f32 {
                for v in CREEP_END..FOUNTAIN_START {let r=base+v*STRIDE;
                    if state[r+TEAM]==state[q+TEAM] && state[r+TOWER_TIER]==3.0f32 && state[r+ACTIVE]>0.0f32 {invulnerable=true;}
                }
            }
        }
        // Native OnTakeDamage reports integral damage after armor/type rules:
        // melee21 -> SF12, ranged23 -> SF9, SF49 -> melee44 (both sides).
        if !invulnerable {state[q+DAMAGE]+=(amount*multiplier*(1.0f32-reduction)).ceil();}
        state[q+CREDIT]=source as f32;
        state[q+CREDIT_TEAM]=source_team;
        if source<2 || source_kind==1.0f32 {
            state[q+FLASK_END]=0.0f32;state[q+CLARITY_END]=0.0f32;
        }
    }
}

#[cube]
fn launch_requiem(state:&mut [f32],shots:&mut [f32],shot_hits:&mut [u32],meta:&mut [f32],hero:&[f32],base:usize,m:usize,source:usize,damage:f32){
    let p=base+source*STRIDE;let count=f32::min(hero[18],(state[p+UNIT_SOULS]/hero[17]).floor()) as usize;
    for line in 0..20 {
        if line<count {
            let angle=state[p+FACING]+line as f32*6.2831855f32/count as f32;let mut allocated=false;
            for s in 0..SHOTS {
                let k=(m*SHOTS+s)*SHOT_FIELDS;
                if !allocated && shots[k]==0.0f32 {
                    // Axial and grazing native fixtures:150-unit initial offset,
                    // projectile radius without adding the victim hull. The
                    // old125+24 approximation gave false hits at600/22.5deg.
                    shots[k]=2.0f32;shots[k+1]=state[p+UNIT_X]+angle.cos()*150.0f32;shots[k+2]=state[p+UNIT_Y]+angle.sin()*150.0f32;shots[k+3]=hero[16];
                    shots[k+4]=hero[14];shots[k+5]=hero[13];shots[k+6]=damage;shots[k+7]=source as f32;shots[k+8]=state[p+TEAM];
                    shots[k+9]=angle.cos();shots[k+10]=angle.sin();shots[k+11]=0.0f32;
                    for word in 0..8 {shot_hits[(m*SHOTS+s)*8+word]=0u32;}
                    allocated=true;
                }
            }
            if !allocated {meta[m*META+7]=1.0f32;}
        }
    }
}

#[cube]
fn launch_shot(state:&mut [f32],shots:&mut [f32],meta:&mut [f32],hero:&[f32],animation:&[f32],rng:&mut [u32],base:usize,m:usize,source:usize,target:usize,damage:f32,time:f32){
    let mut allocated=false;let p=base+source*STRIDE;
    for s in 0..SHOTS {
        let k=(m*SHOTS+s)*SHOT_FIELDS;
        if !allocated && shots[k]==0.0f32 {
            shots[k]=1.0f32;shots[k+1]=state[p+UNIT_X];shots[k+2]=state[p+UNIT_Y];shots[k+3]=target as f32;
            shots[k+10]=0.0f32;shots[k+11]=0.0f32;
            if source<2 {
                if state[p+ATTACK_HAND]==0.0f32 {state[p+ATTACK_HAND]=if random(rng,m)<0.5f32 {-1.0f32}else{1.0f32};}
                let facing=state[p+FACING];let mut side=state[p+ATTACK_HAND]*hero[25];let mut forward=hero[24];
                if animation[0]>0.0f32 {
                    // The native animation starts displaying one tick after
                    // windup begins. Retain the actual tick start, including
                    // the fractional weapon phase of repeated attacks.
                    let age=f32::max(0.0f32,((time-meta[m*META+META_ATTACK_ANIMATION_START+source])*30.0f32+0.003f32).floor()-1.0f32);
                    let cycle=age/30.0f32*1.6f32/state[p+UNIT_ATTACK_INTERVAL]*animation[2];
                    let first=if state[p+ATTACK_HAND]<0.0f32{8usize}else{8+animation[3]as usize*3};
                    let count=if state[p+ATTACK_HAND]<0.0f32{animation[3]as usize}else{animation[4]as usize};
                    forward=animation[first+1];side=animation[first+2];
                    for row in 1..count {
                        let at=first+row*3;let previous=at-3;
                        let fraction=((cycle-animation[previous])/(animation[at]-animation[previous])).clamp(0.0f32,1.0f32);
                        if cycle>=animation[previous] {
                            forward=animation[previous+1]+(animation[at+1]-animation[previous+1])*fraction;
                            side=animation[previous+2]+(animation[at+2]-animation[previous+2])*fraction;
                        }
                    }
                    shots[k+11]=animation[1];
                }
                shots[k+1]+=facing.cos()*forward-facing.sin()*side;shots[k+2]+=facing.sin()*forward+facing.cos()*side;
                state[p+ATTACK_HAND]=-state[p+ATTACK_HAND];
            }
            shots[k+4]=state[base+target*STRIDE+GENERATION];shots[k+5]=state[p+UNIT_PROJECTILE_SPEED];
            shots[k+6]=damage;shots[k+7]=source as f32;shots[k+8]=state[p+TEAM];allocated=true;
            shots[k+9]=state[p+KIND];
            if source<2 && target<2 {meta[m*META+8]+=1.0f32;meta[m*META+9]=meta[m*META];}
        }
    }
    if !allocated {meta[m*META+7]=1.0f32;}
}

#[cube]
fn equip(state:&mut [f32],p:usize,item:usize,charges:f32){
    if charges>0.0f32 {
        let mut used=0usize;
        for i in 0..8 {if i!=3 && state[p+ITEM_START+i]>0.0f32 {used+=1;}}
        if item==3 || state[p+ITEM_START+item]>0.0f32 || used<6 {
            if item==3 {state[p+SLOT_START+item]=15.0f32;}
            else if state[p+ITEM_START+item]==0.0f32 {
                let mut selected=false;
                for slot in 0..6 {let mut occupied=false;
                    for i in 0..8 {if i!=3 && state[p+ITEM_START+i]>0.0f32 && state[p+SLOT_START+i]==slot as f32 {occupied=true;}}
                    if !selected && !occupied {state[p+SLOT_START+item]=slot as f32;selected=true;}
                }
            }
            state[p+ITEM_START+item]+=charges;
        }
        else {state[p+BACKPACK_START+item]+=charges;}
    }
}

#[cube]
fn net_worth(state:&mut [f32],items:&[f32],p:usize)->f32 {
    let mut worth=state[p+UNIT_GOLD];
    for item in 0..8 {let mut count=state[p+ITEM_START+item]+state[p+BACKPACK_START+item]+state[p+STASH_START+item]+state[p+COURIER_START+item];
        if item==3 {count=f32::max(0.0f32,count-state[p+FREE_SCROLL]);}
        worth+=count*items[item*ITEM_RULE_FIELDS]/f32::max(1.0f32,items[item*ITEM_RULE_FIELDS+1]);
    }
    worth
}

#[cube]
fn add_xp(state:&mut [f32],proto:&[f32],hero:&[f32],items:&[f32],experience:&[f32],p:usize,u:usize,amount:f32){
    let old_level=state[p+UNIT_LEVEL];state[p+XP]+=amount;
    for _ in 0..29 {let level=state[p+UNIT_LEVEL] as usize;
        if level<30 && state[p+XP]>=experience[level-1] {state[p+XP]-=experience[level-1];state[p+UNIT_LEVEL]+=1.0f32;}
    }
    if state[p+UNIT_LEVEL]!=old_level {level_stats(state,proto,hero,items,p,u);}
    state[p+UNIT_XP_TO_NEXT]=if state[p+UNIT_LEVEL]>=30.0f32 {0.0f32}else{experience[state[p+UNIT_LEVEL] as usize-1]-state[p+XP]};
}

#[cube]
fn level_stats(state:&mut [f32],proto:&[f32],hero:&[f32],items:&[f32],p:usize,u:usize){
    let level=state[p+UNIT_LEVEL];let boots=state[p+ITEM_START+1];let bracer=state[p+ITEM_START+2];
    let strength=hero[3]*(level-1.0f32)+items[9]+bracer*items[2*ITEM_RULE_FIELDS+9];
    let agility=hero[4]*(level-1.0f32)+items[10]+bracer*items[2*ITEM_RULE_FIELDS+10];
    let intellect=hero[5]*(level-1.0f32)+items[11]+bracer*items[2*ITEM_RULE_FIELDS+11];
    let old_hp=state[p+UNIT_HEALTH_MAX];let old_mana=state[p+UNIT_MANA_MAX];
    // Native max resources use whole total attributes; attack speed, resistance
    // and regeneration below retain their continuous attribute contributions.
    let hp_strength=(hero[0]+strength+0.00001f32).floor()-hero[0];
    let mana_intellect=(hero[2]+intellect+0.00001f32).floor()-hero[2];
    state[p+UNIT_HEALTH_MAX]=proto[u*UNIT_COUNT+UNIT_HEALTH_MAX]+hp_strength*22.0f32+bracer*items[2*ITEM_RULE_FIELDS+19];
    state[p+UNIT_MANA_MAX]=proto[u*UNIT_COUNT+UNIT_MANA_MAX]+mana_intellect*12.0f32;
    if state[p+UNIT_HEALTH]>0.0f32 {
        state[p+UNIT_HEALTH]=f32::min(state[p+UNIT_HEALTH]+state[p+UNIT_HEALTH_MAX]-old_hp,state[p+UNIT_HEALTH_MAX]);
        state[p+UNIT_MANA]=f32::min(state[p+UNIT_MANA]+state[p+UNIT_MANA_MAX]-old_mana,state[p+UNIT_MANA_MAX]);
    }
    state[p+UNIT_ARMOR]=proto[u*UNIT_COUNT+UNIT_ARMOR]+agility/6.0f32+items[13];
    state[p+UNIT_ATTACK_INTERVAL]=1.6f32/(1.0f32+(hero[1]+agility+items[12])/100.0f32);
    state[p+UNIT_MOVE_SPEED]=proto[u*UNIT_COUNT+UNIT_MOVE_SPEED]+boots*items[ITEM_RULE_FIELDS+14];
    state[p+UNIT_ATTACK_DAMAGE]=proto[u*UNIT_COUNT+UNIT_ATTACK_DAMAGE]+agility+state[p+UNIT_SOULS]*(hero[11]+(level/6.0f32).floor()*0.8f32);
    state[p+UNIT_HEALTH_REGEN]=proto[u*UNIT_COUNT+UNIT_HEALTH_REGEN]+strength*0.1f32+state[p+ITEM_START+4]*items[4*ITEM_RULE_FIELDS+4]+bracer*items[2*ITEM_RULE_FIELDS+20];
    state[p+UNIT_MANA_REGEN]=proto[u*UNIT_COUNT+UNIT_MANA_REGEN]+intellect*0.05f32;
    state[p+UNIT_MAGIC_RESIST]=proto[u*UNIT_COUNT+UNIT_MAGIC_RESIST]+intellect*0.001f32;
}

#[cube]
pub fn spell_level(level:f32,spell:usize)->usize {
    let mut result=1usize;
    if spell<3 {
        if level>=3.0f32 {result=2;}if level>=5.0f32 {result=3;}if level>=7.0f32 {result=4;}
    } else {result=if level>=6.0f32 {1usize} else {0usize};}
    result
}

#[cube]
fn cast_spell(state:&mut [f32],spells:&[f32],p:usize,spell:usize,time:f32){
    let level=spell_level(state[p+UNIT_LEVEL],spell);
    if level>0 && state[p+COOLDOWNS+spell]<=time {
        let k=(spell*4+level-1)*8;let cost=spells[k+1];
        if state[p+UNIT_MANA]>=cost {
            state[p+CAST]=(spell+1) as f32;state[p+CAST_END]=time+spells[k+5];
            state[p+WINDUP]=0.0f32;
            // Native no-target abilities suspend a movement order, continue
            // its facing change during casting, then resume that movement.
            if state[p+ORDER]!=1.0f32 && state[p+ORDER]!=2.0f32 {state[p+ORDER]=0.0f32;}
        }
    }
}

#[cube(launch)]
fn enqueue(actions:&[f32],refs:&[f32],commands:&mut [f32],state:&[f32],meta:&[f32],bases:&[f32],actors:usize,targets:usize,tree_count:usize){
    let a=ABSOLUTE_POS;
    if a<actors {
        let m=a/2;let slot=(meta[m*META+6] as usize)%4;let start=(a*4+slot)*COMMAND_FIELDS;
        commands[start]=meta[m*META]+6.0f32;
        for f in 0..6 {commands[start+1+f]=actions[a*6+f];}
        commands[start+4]=refs[a*targets+actions[a*6+3] as usize];
        let p=(m*UNITS+a%2)*STRIDE;
        commands[start+5]=state[p+UNIT_X]+(actions[a*6+4]-64.0f32)*16.0f32;
        commands[start+6]=state[p+UNIT_Y]+(actions[a*6+5]-64.0f32)*16.0f32;
        // TP coordinates are relative to the selected building/home shop.
        if actions[a*6]==10.0f32 && actions[a*6+2]==3.0f32 {
            let ref_id=commands[start+4];let target=ref_id as usize;
            if ref_id>=0.0f32 && target<UNITS {
                let q=(m*UNITS+target)*STRIDE;
                commands[start+5]=state[q+UNIT_X]+(actions[a*6+4]-64.0f32)*16.0f32;
                commands[start+6]=state[q+UNIT_Y]+(actions[a*6+5]-64.0f32)*16.0f32;
            } else if target==UNITS+tree_count+a%2 {
                commands[start+5]=bases[a%2*3]+(actions[a*6+4]-64.0f32)*16.0f32;
                commands[start+6]=bases[a%2*3+1]+(actions[a*6+5]-64.0f32)*16.0f32;
            }
        }
        commands[start+7]=1.0f32;
        let target=commands[start+4] as usize;
        commands[start+8]=if commands[start+4]>=0.0f32 && target<UNITS {state[(m*UNITS+target)*STRIDE+GENERATION]}else{0.0f32};
    }
}

#[cube(launch)]
fn simulate(state:&mut [f32],shots:&mut [f32],shot_hits:&mut [u32],meta:&mut [f32],commands:&mut [f32],
    teacher_memory:&[f32],teacher_forecasts:&[f32],teacher_audit:&mut [f32],
    rewards:&mut [f32],terminals:&mut [f32],tree_alive:&mut [f32],rng:&mut [u32],grid:&mut [u32],
    proto:&[f32],spells:&[f32],items:&[f32],hero:&[f32],animation:&[f32],combat:&[f32],experience:&[f32],geometry:&[f32],fow_blockers:&[f32],static_vision:&[u32],navigation:&[u32],trees:&[f32],tree_map:&[f32],bases:&[f32],spawns:&[f32],lane:&[f32],fountains:&[f32],extra_vision:&[f32],
    matches:usize,tree_count:usize,vision_count:usize,pregame:f32,ticks_per_call:usize,radiant_lane_points:usize,dire_lane_points:usize,#[comptime] spawn_waves:bool,#[comptime] audit_enabled:bool){
    let m=ABSOLUTE_POS;
    if m<matches && (!audit_enabled || teacher_audit[m*crate::scripted_teacher::AUDIT_FIELDS]==0.0f32) {
        let base=m*UNITS*STRIDE;let mb=m*META;
        if meta[mb+2]>0.0f32 {
            initialize(state,proto,m);reset_attack_phase(meta,m);meta[mb]=-pregame*30.0f32;meta[mb+1]+=1.0f32;meta[mb+2]=0.0f32;
            for i in 0..SHOTS*SHOT_FIELDS {shots[m*SHOTS*SHOT_FIELDS+i]=0.0f32;}
            for i in 0..COMMAND_WORDS {commands[m*COMMAND_WORDS+i]=0.0f32;}
            for t in 0..tree_count {tree_alive[m*tree_count+t]=1.0f32;}
        }
        let before=potential(state,base);let mut result=(0.0f32).runtime();
        rewards[m*2]=0.0f32;rewards[m*2+1]=0.0f32;terminals[m*2]=0.0f32;terminals[m*2+1]=0.0f32;
        for _substep in 0..ticks_per_call {
            if meta[mb+2]==0.0f32 {
                let tick=meta[mb] as i32;let game_time=meta[mb]/30.0f32;
                // Timers use nonnegative episode time; DotaTime is negative in
                // pregame and must never activate a zero-valued inactive buff.
                let time=f32::max(0.0f32,(meta[mb]+pregame*30.0f32)/30.0f32);let dt=1.0f32/30.0f32;
                let mut fresh_hero_attack=0u32.runtime();
                for v in 2..UNITS {let p=base+v*STRIDE;
                    if state[p+ACTIVE]>0.0f32 && state[p+KIND]==1.0f32 {
                        let mut protected=game_time<0.0f32 || v>=FOUNTAIN_START;let parent=state[p+STRUCTURE_PARENT];
                        if parent>=0.0f32 && state[base+parent as usize*STRIDE+ACTIVE]>0.0f32 {protected=true;}
                        if parent== -2.0f32 {for tower in CREEP_END..FOUNTAIN_START {let q=base+tower*STRIDE;
                            if state[q+TEAM]==state[p+TEAM] && state[q+TOWER_TIER]==3.0f32 && state[q+ACTIVE]>0.0f32 {protected=true;}
                        }}
                        state[p+INVULNERABLE]=if protected {1.0f32}else{0.0f32};
                    }
                }
                for h in 0..2 {let p=base+h*STRIDE;
                    if state[p+ACTIVE]==0.0f32 && state[p+RESPAWN_AT]>0.0f32 && time>=state[p+RESPAWN_AT] {
                        meta[mb+META_ATTACK_POINT_ERROR+h]=0.0f32;meta[mb+META_ATTACK_NOMINAL_END+h]=0.0f32;
                        meta[mb+META_ATTACK_POINT_PENDING+h]=0.0f32;meta[mb+META_ATTACK_ANIMATION_START+h]=0.0f32;
                        state[p+ACTIVE]=1.0f32;state[p+RESPAWN_AT]=0.0f32;
                        state[p+RESPAWN_RESET]=meta[mb+6]+1.0f32;
                        for f in UNIT_X..UNIT_Z+1 {state[p+f]=if hero.len()>=39 {hero[33+h*3+f-UNIT_X]}else{proto[h*UNIT_COUNT+f]};}
                        state[p+FACING]=if h==0 {0.7853982f32}else{-2.3561945f32};state[p+ATTACK_AIM]=state[p+FACING];
                        state[p+UNIT_HEALTH]=state[p+UNIT_HEALTH_MAX];state[p+UNIT_MANA]=state[p+UNIT_MANA_MAX];
                    }
                }
                if spawn_waves && tick>=0 && tick%900==0 {
                    // Three melee and one ranged per side. Slot generation protects
                    // tracking projectiles against reused dead creep slots.
                    for side in 0..2 {
                        let count=if tick>0 && tick%9000==0 {5usize}else{4usize};
                        let flag_wave=tick>=3600 && tick%1800==0;
                        let flag_slot=if flag_wave {(random(rng,m)*3.0f32).floor()as usize}else{3usize};
                        for c in 0..5 {
                          if c<count {
                            let mut allocated=false;
                            for u in 6..CREEP_END {
                                let p=base+u*STRIDE;
                                if !allocated && state[p+ACTIVE]==0.0f32 {
                                    let generation=state[p+GENERATION]+1.0f32;
                                    for f in 0..STRIDE {state[p+f]=0.0f32;}
                                    let kind=if c<3 {6usize} else if c==3 {7usize} else {8usize};
                                    for f in 0..UNIT_COUNT {state[p+f]=proto[kind*UNIT_COUNT+f];}
                                    let spawn=(side*4+(c+1)%4)*3;
                                    state[p+UNIT_X]=spawns[spawn];state[p+UNIT_Y]=spawns[spawn+1];state[p+UNIT_Z]=spawns[spawn+2];
                                    let growth=(game_time/450.0f32).floor();
                                    if c<4 {state[p+UNIT_HEALTH_MAX]+=12.0f32*growth;state[p+UNIT_HEALTH]=state[p+UNIT_HEALTH_MAX];state[p+UNIT_ATTACK_DAMAGE]+=growth*if c<3 {1.0f32}else{2.0f32};}
                                    state[p+TEAM]=side as f32;state[p+KIND]=if c<3 {2.0f32} else if c==3 {3.0f32} else {4.0f32};
                                    state[p+ACTIVE]=1.0f32;state[p+GENERATION]=generation;state[p+TARGET]=-1.0f32;
                                    state[p+BOUNTY]=if c<3 {34.0f32+(random(rng,m)*6.0f32).floor()} else if c==3 {43.0f32+(random(rng,m)*10.0f32).floor()} else {59.0f32+(random(rng,m)*14.0f32).floor()};
                                    state[p+XP_BOUNTY]=if c<3 {57.0f32} else if c==3 {69.0f32+growth*8.0f32}else{88.0f32};
                                    if flag_wave && c==flag_slot {
                                        state[p+FLAGBEARER]=1.0f32;state[p+XP_BOUNTY]+=3.0f32;state[p+BOUNTY]=state[p+BOUNTY]*2.0f32+10.0f32;
                                        let mut remaining=0.6f32.runtime();
                                        for interval in 0..15 {if (interval as f32)<growth {remaining*=0.96f32;}}
                                        state[p+UNIT_MAGIC_RESIST]=1.0f32-remaining;
                                    }
                                    state[p+LANE_POSITION]=if side==0 {0.0f32}else{radiant_lane_points as f32};
                                    allocated=true;
                                }
                            }
                            if !allocated {meta[mb+7]=2.0f32;}
                          }
                        }
                    }
                }
                build_collision_grid(state,grid,m,base);
                for flag in 6..CREEP_END {let source=base+flag*STRIDE;
                    if state[source+ACTIVE]>0.0f32 && state[source+UNIT_HEALTH]>0.0f32 && state[source+FLAGBEARER]>0.0f32 {
                        for unit in 6..CREEP_END {let p=base+unit*STRIDE;
                            if state[p+ACTIVE]>0.0f32 && state[p+UNIT_HEALTH]>0.0f32 && state[p+TEAM]==state[source+TEAM] && distance(state,source,p)<=700.0f32+hull(state[p+KIND]) {state[p+FLAG_AURA_END]=time+0.5f32;}
                        }
                    }
                }
                // Native tower protection applies to allied heroes, including
                // their hull in the900-unit radius. Each tier retains its own
                // half-second linger; the stronger active tier wins.
                for h in 0..2 {
                    let p=base+h*STRIDE;
                    if state[p+UNIT_HEALTH]>0.0f32 {
                        for v in 2..UNITS {let q=base+v*STRIDE;
                            if state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>0.0f32 && state[q+TEAM]==state[p+TEAM] && state[q+KIND]==1.0f32 && v<FOUNTAIN_START && state[q+TOWER_TIER]>=1.0f32 && distance(state,p,q)<=924.0f32 {
                                let field=if state[q+TOWER_TIER]==1.0f32 {TIER1_AURA_END}else{HIGH_AURA_END};state[p+field]=time+0.5f32;
                            }
                        }
                    }
                }
                for u in 0..2 {
                    let p=base+u*STRIDE;let a=m*2+u;
                    // Passive gold continues through death; unlike kill/creep
                    // gold it is reliable and is protected from death loss.
                    if game_time>=0.0f32 {let gold=100.0f32/60.0f32*dt;state[p+UNIT_GOLD]+=gold;state[p+RELIABLE_GOLD]+=gold;}
                    for slot in 0..4 {
                        let k=(a*4+slot)*COMMAND_FIELDS;
                        if commands[k+7]>0.0f32 && commands[k]+1.0f32<=meta[mb] {
                            let action=commands[k+1] as usize;let buy=commands[k+2] as usize;let item=commands[k+3] as usize;
                            let target=commands[k+4] as usize;
                            let x=commands[k+5];
                            let y=commands[k+6];
                            let same_cast=action>=6 && action<=9 && state[p+CAST]==(action-5) as f32 && state[p+CAST_END]>time;
                            if state[p+ACTIVE]>0.0f32 && state[p+UNIT_HEALTH]>0.0f32 && state[p+FEAR_END]<=time && action>0 && !same_cast {
                                // Reissuing the same attack order preserves windup in
                                // the real executor; changing orders still cancels it.
                                let same_attack=action==3 && state[p+ORDER]==3.0f32 && state[p+TARGET]==target as f32 && state[p+TARGET_GENERATION]==commands[k+8];
                                if !same_attack {if audit_enabled && state[p+WINDUP]>time {crate::ggd_metrics::cancel(teacher_audit,m,u);}state[p+WINDUP]=0.0f32;}
                                state[p+CAST_END]=0.0f32;state[p+CAST]=0.0f32;state[p+TP_END]=0.0f32;
                                state[p+TANGO_TARGET]=0.0f32;
                                state[p+HOLD]=0.0f32;
                                if action==1 || action==2 {state[p+ORDER]=action as f32;state[p+TARGET]=-1.0f32;state[p+DEST_X]=x;state[p+DEST_Y]=y;}
                                if action==3 && commands[k+4]>=0.0f32 && target<UNITS && state[base+target*STRIDE+ACTIVE]>0.0f32 && state[base+target*STRIDE+GENERATION]==commands[k+8] {
                                    let q=base+target*STRIDE;
                                    if (state[q+KIND]==1.0f32 || visible(state,geometry,fow_blockers,static_vision,tree_map,trees,tree_alive,extra_vision,vision_count,m,tree_count,base,u as f32,q,game_time,time)) && (state[q+TEAM]!=u as f32 || state[q+UNIT_HEALTH]<=state[q+UNIT_HEALTH_MAX]*0.5f32) {
                                        if !same_attack {state[p+ATTACK_AIM]=state[p+FACING];}
                                        state[p+ORDER]=3.0f32;state[p+TARGET]=target as f32;
                                        if !same_attack{fresh_hero_attack=fresh_hero_attack|(1u32<<u as u32);}
                                        state[p+TARGET_GENERATION]=state[q+GENERATION];
                                        for v in 2..UNITS {
                                            let q=base+v*STRIDE;
                                            if target<2 && state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>0.0f32 && state[q+TEAM]!=u as f32 && distance(state,p,q)<500.0f32 {
                                                if state[q+KIND]>=2.0f32 && state[q+AGGRO_READY]<=time {
                                                    if state[q+AGGRO_END]==0.0f32 {state[q+AGGRO_PREVIOUS]=state[q+TARGET]+1.0f32;state[q+AGGRO_PREVIOUS_GENERATION]=state[q+TARGET_GENERATION];}
                                                    state[q+AGGRO_PENDING]=(u+1)as f32;state[q+AGGRO_END]=time+2.3f32;state[q+AGGRO_READY]=time+3.0f32;
                                                }
                                                if state[q+KIND]==1.0f32 && state[q+AGGRO_END]<=time {
                                                    state[q+TARGET]=u as f32;state[q+AGGRO_END]=time+2.3f32;state[q+TARGET_GENERATION]=state[p+GENERATION];
                                                }
                                            }
                                        }
                                    }
                                }
                                if action==4 || action==5 {state[p+ORDER]=0.0f32;state[p+TARGET]=-1.0f32;state[p+HOLD]=if action==5 {1.0f32} else {0.0f32};}
                                if action>=6 && action<=9 {cast_spell(state,spells,p,action-6,time);}
                                if action==10 && item>=3 && item<8 && state[p+ITEM_START+item]>0.0f32 && state[p+ITEM_READY+item]<=time && state[p+UNIT_MANA]>=items[item*ITEM_RULE_FIELDS+16] {
                                    let mut used=false;
                                    if item==4 {state[p+UNIT_MANA]=f32::min(state[p+UNIT_MANA_MAX],state[p+UNIT_MANA]+items[item*ITEM_RULE_FIELDS+6]);used=true;}
                                    if item==5 || item==7 {
                                        let buff=if item==5 {FLASK_END} else {CLARITY_END};
                                        state[p+buff]=time+items[item*ITEM_RULE_FIELDS+5];used=true;
                                    }
                                    if item==6 && target>=UNITS && target<UNITS+tree_count {
                                        let t=target-UNITS;let dx=trees[t*3]-state[p+UNIT_X];let dy=trees[t*3+1]-state[p+UNIT_Y];
                                        if dx*dx+dy*dy<=165.0f32*165.0f32 && tree_alive[m*tree_count+t]>0.0f32 {
                                            tree_alive[m*tree_count+t]=0.0f32;state[p+TANGO_END]=time+items[item*ITEM_RULE_FIELDS+5];used=true;
                                        } else if tree_alive[m*tree_count+t]>0.0f32 {
                                            state[p+ORDER]=6.0f32;state[p+TANGO_TARGET]=(t+1) as f32;state[p+TARGET]=-1.0f32;
                                        }
                                    }
                                    if item==3 {
                                        let valid_tower=commands[k+4]>=0.0f32 && target>=2 && target<6 && state[base+target*STRIDE+TEAM]==u as f32 && state[base+target*STRIDE+ACTIVE]>0.0f32;
                                        if valid_tower || target==UNITS+tree_count+u {
                                            let mut origin_x=bases[u*3];let mut origin_y=bases[u*3+1];
                                            let mut best=(x-origin_x)*(x-origin_x)+(y-origin_y)*(y-origin_y);
                                            for tower in 2..6 {let q=base+tower*STRIDE;
                                                if state[q+TEAM]==u as f32 && state[q+ACTIVE]>0.0f32 {
                                                    let dx=x-state[q+UNIT_X];let dy=y-state[q+UNIT_Y];let distance=dx*dx+dy*dy;
                                                    if distance<best {best=distance;origin_x=state[q+UNIT_X];origin_y=state[q+UNIT_Y];}
                                                }
                                            }
                                            let dx=x-origin_x;let dy=y-origin_y;let length=(dx*dx+dy*dy).sqrt();
                                            let angle=dy.atan2(dx);let radius=length.clamp(hero[22],hero[23]);
                                            let mut tx=origin_x+angle.cos()*radius;let mut ty=origin_y+angle.sin()*radius;let mut valid=false;
                                            for candidate in 0..33 {
                                                let a=angle+candidate as f32*0.7853982f32;let offset=((candidate+7)/8) as f32*64.0f32;
                                                let sx=tx+a.cos()*offset;let sy=ty+a.sin()*offset;let rx=sx-origin_x;let ry=sy-origin_y;
                                                let r=(rx*rx+ry*ry).sqrt().clamp(hero[22],hero[23]);let bearing=ry.atan2(rx);
                                                let sx=origin_x+bearing.cos()*r;let sy=origin_y+bearing.sin()*r;
                                                if !valid && geometry[cell(sx,sy)+1]>0.5f32 {tx=sx;ty=sy;valid=true;}
                                            }
                                            if valid {state[p+TP_END]=time+items[item*ITEM_RULE_FIELDS+18];state[p+TP_X]=tx;state[p+TP_Y]=ty;state[p+ORDER]=0.0f32;used=true;}
                                        }
                                    }
                                    if used {state[p+ITEM_START+item]-=1.0f32;state[p+ITEM_READY+item]=time+items[item*ITEM_RULE_FIELDS+2];state[p+UNIT_MANA]-=items[item*ITEM_RULE_FIELDS+16];if item==3 {state[p+FREE_SCROLL]=0.0f32;}}
                                }
                            }
                            if buy>0 {
                                let mut which=buy+1;
                                if buy==1 {which=if state[p+ITEM_START+1]+state[p+BACKPACK_START+1]+state[p+STASH_START+1]+state[p+COURIER_START+1]==0.0f32 {1usize} else {2usize};}
                                let stock=items[which*ITEM_RULE_FIELDS+7];
                                let pending_permanent=which<3 && state[p+STASH_START+which]+state[p+COURIER_START+which]+state[p+ITEM_START+which]+state[p+BACKPACK_START+which]>0.0f32;
                                if which<8 && !pending_permanent && state[p+UNIT_GOLD]>=items[which*ITEM_RULE_FIELDS] && (stock<0.0f32 || state[p+STOCK_START+which]<stock) {
                                    let cost=items[which*ITEM_RULE_FIELDS];let unreliable=state[p+UNIT_GOLD]-state[p+RELIABLE_GOLD];
                                    state[p+RELIABLE_GOLD]=f32::max(0.0f32,state[p+RELIABLE_GOLD]-f32::max(0.0f32,cost-unreliable));
                                    state[p+UNIT_GOLD]-=cost;state[p+STASH_START+which]+=items[which*ITEM_RULE_FIELDS+1];
                                    if stock>=0.0f32 {
                                        state[p+STOCK_START+which]+=1.0f32;
                                        if state[p+RESTOCK_START+which]==0.0f32 {state[p+RESTOCK_START+which]=time+items[which*ITEM_RULE_FIELDS+8];}
                                    }
                                }
                            }
                            commands[k+7]=0.0f32;
                        }
                    }
                    let home_dx=state[p+UNIT_X]-bases[u*3];let home_dy=state[p+UNIT_Y]-bases[u*3+1];
                    let home_distance=(home_dx*home_dx+home_dy*home_dy).sqrt();
                    for item in 0..8 {
                        if state[p+RESTOCK_START+item]>0.0f32 && state[p+RESTOCK_START+item]<=time {
                            state[p+STOCK_START+item]=f32::max(0.0f32,state[p+STOCK_START+item]-1.0f32);
                            state[p+RESTOCK_START+item]=if state[p+STOCK_START+item]>0.0f32 {time+items[item*ITEM_RULE_FIELDS+8]} else {0.0f32};
                        }
                        if home_distance<1100.0f32 {let count=state[p+STASH_START+item];equip(state,p,item,count);state[p+STASH_START+item]=0.0f32;}
                        if state[p+DELIVERY]>0.0f32 && state[p+DELIVERY]<=time {let count=state[p+COURIER_START+item];equip(state,p,item,count);state[p+COURIER_START+item]=0.0f32;}
                    }
                    if state[p+DELIVERY]>0.0f32 && state[p+DELIVERY]<=time {state[p+DELIVERY]=0.0f32;}
                    // Bounded courier approximation: transport starts on the real
                    // executor's four-second cadence; payload is frozen in flight.
                    if game_time>=0.0f32 && tick%120==0 && state[p+DELIVERY]==0.0f32 {
                        let mut loaded=false;
                        for item in 0..8 {state[p+COURIER_START+item]=state[p+STASH_START+item];state[p+STASH_START+item]=0.0f32;if state[p+COURIER_START+item]>0.0f32 {loaded=true;}}
                        if loaded {state[p+DELIVERY]=time+home_distance/800.0f32+4.0f32;}
                    }
                    level_stats(state,proto,hero,items,p,u);
                    state[p+UNIT_XP_TO_NEXT]=experience[state[p+UNIT_LEVEL] as usize-1]-state[p+XP];
                }
                for u in 0..UNITS {
                    let p=base+u*STRIDE;
                        if state[p+ACTIVE]>0.0f32 && state[p+UNIT_HEALTH]>state[p+DAMAGE] {
                        let side=state[p+TEAM];
                        if u>=2 {let regen=state[p+UNIT_HEALTH_REGEN]+if state[p+KIND]>=2.0f32 && state[p+FLAG_AURA_END]>time {3.0f32}else{0.0f32};state[p+UNIT_HEALTH]=f32::min(state[p+UNIT_HEALTH_MAX],state[p+UNIT_HEALTH]+regen*dt);}
                        if state[p+KIND]>=2.0f32 && state[p+AGGRO_PENDING]>0.0f32 && creep_orders_ready(state,p,time) {
                            let target=state[p+AGGRO_PENDING]as usize-1;let q=base+target*STRIDE;
                            if state[p+AGGRO_END]>time && state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>0.0f32 && visible(state,geometry,fow_blockers,static_vision,tree_map,trees,tree_alive,extra_vision,vision_count,m,tree_count,base,side,q,game_time,time) {
                                state[p+TARGET]=target as f32;state[p+TARGET_GENERATION]=state[q+GENERATION];state[p+AGGRO_RECOVER]=0.0f32;
                                if state[p+KIND]==4.0f32 {state[p+WINDUP_TARGET]=target as f32;state[p+WINDUP_GENERATION]=state[q+GENERATION];}
                            }else{state[p+AGGRO_END]=0.0f32;}
                            state[p+AGGRO_PENDING]=0.0f32;
                        }
                        let old=base+state[p+TARGET].clamp(0.0f32,(UNITS-1)as f32)as usize*STRIDE;
                        let in_range=state[p+TARGET]>=0.0f32 && state[p+TARGET]<(UNITS as f32) && state[old+ACTIVE]>0.0f32 && state[old+UNIT_HEALTH]>0.0f32 && distance(state,p,old)<=attack_reach(state,p,old);
                        if state[p+KIND]>=2.0f32 && state[p+AGGRO_END]>0.0f32 && state[p+AGGRO_END]<=time && (!in_range || state[p+KIND]==4.0f32) && creep_orders_ready(state,p,time) && state[p+AGGRO_PENDING]==0.0f32 {
                            state[p+DEST_X]=state[old+UNIT_X];state[p+DEST_Y]=state[old+UNIT_Y];
                            let previous=state[p+AGGRO_PREVIOUS]as usize;state[p+TARGET]=-1.0f32;
                            if previous>0 && previous<=UNITS {let q=base+(previous-1)*STRIDE;
                                if state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>0.0f32 && state[q+GENERATION]==state[p+AGGRO_PREVIOUS_GENERATION] {state[p+TARGET]=(previous-1)as f32;state[p+TARGET_GENERATION]=state[q+GENERATION];}
                            }
                            state[p+AGGRO_END]=0.0f32;state[p+AGGRO_PREVIOUS]=0.0f32;state[p+AGGRO_RECOVER]=if !in_range{time+0.5f32}else{0.0f32};
                            if state[p+KIND]==4.0f32 {state[p+WINDUP_TARGET]=state[p+TARGET];state[p+WINDUP_GENERATION]=state[p+TARGET_GENERATION];}
                        }
                        let current=base+state[p+TARGET].clamp(0.0f32,(UNITS-1)as f32)as usize*STRIDE;
                        let aggro_expired=state[p+AGGRO_END]>0.0f32 && state[p+AGGRO_END]<=time;
                        let holding=u<2 && state[p+HOLD]>0.0f32;
                        let acquisition=if holding {state[p+UNIT_ATTACK_RANGE]+24.0f32}else{900.0f32};
                        let expired=aggro_expired && !(state[p+KIND]>=2.0f32 && in_range);
                        // Native natural wave: a chased hero is reconsidered
                        // when a closer enemy appears. Keep acquired creep
                        // targets, including the target restored after a pull.
                        let reconsider_hero=state[p+KIND]>=2.0f32 && state[p+TARGET]>=0.0f32 && state[current+KIND]==0.0f32 && !in_range && state[p+AGGRO_END]<=time && creep_orders_ready(state,p,time);
                        if (u>=2 || holding) && state[p+AGGRO_RECOVER]<=time && (reconsider_hero || state[p+TARGET]<0.0f32 || state[current+ACTIVE]==0.0f32 || state[current+GENERATION]!=state[p+TARGET_GENERATION] || distance(state,p,current)>acquisition || expired || !visible(state,geometry,fow_blockers,static_vision,tree_map,trees,tree_alive,extra_vision,vision_count,m,tree_count,base,side,current,game_time,time)) {
                            state[p+AGGRO_RECOVER]=0.0f32;
                            if aggro_expired {state[p+AGGRO_END]=0.0f32;}
                            // Installed/native acquisition: melee500, ranged600,
                            // siege800. Towers use their own attack range.
                            let radius=if holding {acquisition}else if state[p+KIND]==1.0f32 {state[p+UNIT_ATTACK_RANGE]}else if state[p+KIND]==2.0f32 {500.0f32}else if state[p+KIND]==3.0f32 {600.0f32}else{800.0f32};let mut nearest=radius;let mut priority=3usize.runtime();let mut selected=(-1.0f32).runtime();
                            for v in 0..UNITS {
                                let q=base+v*STRIDE;
                                if state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>0.0f32 && state[q+TEAM]!=side && state[q+INVULNERABLE]==0.0f32 {
                                    let dist=distance(state,p,q);
                                    // Natural lane-goal creeps prioritize enemy
                                    // creeps: native natural-priority002 chooses
                                    // creep393 over idle hero108. CreateUnitByName
                                    // without a lane goal instead chooses nearest.
                                    let rank=if state[p+KIND]>=2.0f32 && state[p+LANE_POSITION]>=0.0f32 && state[q+KIND]<2.0f32 {1usize}else{0usize};
                                    if dist<radius && (rank<priority || (rank==priority && dist<nearest)) && visible(state,geometry,fow_blockers,static_vision,tree_map,trees,tree_alive,extra_vision,vision_count,m,tree_count,base,side,q,game_time,time) {nearest=dist;priority=rank;selected=v as f32;state[p+TARGET_GENERATION]=state[q+GENERATION];}
                                }
                            }
                            state[p+TARGET]=selected;
                        }
                        let prior_target=state[p+TARGET]as usize;
                        if u<2 && state[p+TARGET]>=0.0f32 && prior_target<UNITS && (state[base+prior_target*STRIDE+ACTIVE]==0.0f32 || state[base+prior_target*STRIDE+UNIT_HEALTH]<=0.0f32 || state[base+prior_target*STRIDE+GENERATION]!=state[p+TARGET_GENERATION]) {
                            state[p+TARGET]=-1.0f32;if state[p+ORDER]==3.0f32 {state[p+ORDER]=0.0f32;}state[p+WINDUP]=0.0f32;
                        }
                        if u<2 && state[p+TARGET]>=0.0f32 && prior_target<UNITS && state[base+prior_target*STRIDE+KIND]!=1.0f32 && state[base+prior_target*STRIDE+TEAM]!=side && !visible(state,geometry,fow_blockers,static_vision,tree_map,trees,tree_alive,extra_vision,vision_count,m,tree_count,base,side,base+prior_target*STRIDE,game_time,time) {
                            state[p+TARGET]=-1.0f32;if state[p+ORDER]==3.0f32 {state[p+ORDER]=0.0f32;}state[p+WINDUP]=0.0f32;
                        }
                        let moving_cast=u<2 && state[p+CAST_END]>0.0f32 && (state[p+ORDER]==1.0f32 || state[p+ORDER]==2.0f32);
                        let aimed_cast=u<2 && state[p+CAST_END]>0.0f32 && !moving_cast && state[p+TARGET]>=0.0f32 && prior_target<UNITS && state[base+prior_target*STRIDE+ACTIVE]>0.0f32 && state[base+prior_target*STRIDE+UNIT_HEALTH]>0.0f32 && state[base+prior_target*STRIDE+GENERATION]==state[p+TARGET_GENERATION];
                        if moving_cast {
                            let x=state[p+DEST_X];let y=state[p+DEST_Y];let pathfind=state[p+ORDER]==1.0f32;
                            move_unit(state,grid,geometry,navigation,tree_map,trees,tree_alive,m,tree_count,p,x,y,0.0f32,hero[6],pathfind);
                        }
                        if aimed_cast {
                            let q=base+prior_target*STRIDE;let desired=(state[q+UNIT_Y]-state[p+UNIT_Y]).atan2(state[q+UNIT_X]-state[p+UNIT_X]);
                            let delta=wrap_angle(desired-state[p+FACING]);
                            state[p+FACING]=wrap_angle(state[p+FACING]+delta.clamp(-hero[6]*0.4712389f32,hero[6]*0.4712389f32));
                            state[p+ATTACK_AIM]=state[p+FACING];
                        }
                        if state[p+CAST_END]>0.0f32 && state[p+CAST_END]<=time {
                            let spell=state[p+CAST] as usize-1;let level=spell_level(state[p+UNIT_LEVEL],spell);let k=(spell*4+level-1)*8;
                            let x=state[p+UNIT_X]+state[p+FACING].cos()*spells[k+3];let y=state[p+UNIT_Y]+state[p+FACING].sin()*spells[k+3];
                            let mut telemetry_heroes=0.0f32.runtime();let mut telemetry_creeps=0.0f32.runtime();let mut telemetry_stacks=0.0f32.runtime();
                            for v in 0..UNITS {
                                let q=base+v*STRIDE;
                                if state[q+TEAM]!=side && state[q+KIND]!=1.0f32 && state[q+ACTIVE]>0.0f32 {
                                    let dx=state[q+UNIT_X]-x;let dy=state[q+UNIT_Y]-y;
                                    // Native Shadowraze includes the target's hull:
                                    // SF24, melee/siege16 and ranged8 boundary probes.
                                    let radius=spells[k+4]+hull(state[q+KIND]);
                                    if spell<3 && dx*dx+dy*dy<=radius*radius {
                                        let stacks=if state[q+STACK_END]>time {state[q+UNIT_RAZE_STACKS]} else {0.0f32};
                                        if audit_enabled {if v<2 {telemetry_heroes+=1.0f32;if stacks>0.0f32 {telemetry_stacks+=1.0f32;}}else if v<CREEP_END {telemetry_creeps+=1.0f32;}}
                                        let talent=if state[p+UNIT_LEVEL]>=10.0f32 {hero[21]}else{0.0f32};
                                        let amount=spells[k]+stacks*(spells[k+6]+talent)+state[p+UNIT_SOULS]*2.0f32;hit(state,hero,combat,base,u,v,amount,true,time,0.0f32,side);
                                        if v<2 {meta[mb+11]=meta[mb];meta[mb+12]+=1.0f32;}
                                        state[q+UNIT_RAZE_STACKS]=stacks+1.0f32;state[q+STACK_END]=time+spells[k+7];
                                    }
                                }
                            }
                            if spell==3 {launch_requiem(state,shots,shot_hits,meta,hero,base,m,u,spells[k]);}
                            if audit_enabled {crate::ggd_metrics::cast(teacher_audit,m,u,spell,telemetry_heroes,telemetry_creeps,telemetry_stacks);}
                            state[p+UNIT_MANA]-=spells[k+1];state[p+COOLDOWNS+spell]=time+spells[k+2];state[p+CAST_END]=0.0f32;state[p+CAST]=0.0f32;
                        }
                        let target=state[p+TARGET] as usize;
                        if state[p+FEAR_END]>time {
                            let dx=state[p+UNIT_X]-state[p+FEAR_X];let dy=state[p+UNIT_Y]-state[p+FEAR_Y];
                            let length=(dx*dx+dy*dy).sqrt().max(1.0f32);
                            // Optional calibrated flee lookahead; legacy tables
                            // retain their old distance until native validation.
                            let mut distance=length*2.0f32;
                            if hero.len()>39 {if hero[39]>0.0f32 {distance=hero[39];}}
                            let mut origin_x=state[p+UNIT_X];let mut origin_y=state[p+UNIT_Y];
                            if hero.len()>40 {if hero[40]>0.0f32 {origin_x=state[p+FEAR_X];origin_y=state[p+FEAR_Y];}}
                            let away_x=origin_x+dx/length*distance;
                            let away_y=origin_y+dy/length*distance;
                            let mut goal_x=away_x;let mut goal_y=away_y;
                            if hero.len()>41 {if hero[41]>0.0f32 {goal_x=state[p+FEAR_X];goal_y=state[p+FEAR_Y];}}
                            let slowed_dt=dt*(1.0f32+state[p+SLOW_PERCENT]);
                            move_unit(state,grid,geometry,navigation,tree_map,trees,tree_alive,m,tree_count,p,goal_x,goal_y,slowed_dt,hero[6],true);
                        } else if state[p+CAST_END]==0.0f32 && state[p+TP_END]==0.0f32 && !moving_cast && !aimed_cast {
                            if u<2 && state[p+ORDER]==6.0f32 && state[p+TANGO_TARGET]>0.0f32 {
                                let t=state[p+TANGO_TARGET] as usize-1;let x=trees[t*3];let y=trees[t*3+1];
                                let dx=x-state[p+UNIT_X];let dy=y-state[p+UNIT_Y];
                                if tree_alive[m*tree_count+t]==0.0f32 || state[p+ITEM_START+6]==0.0f32 {state[p+ORDER]=0.0f32;state[p+TANGO_TARGET]=0.0f32;}
                                else if dx*dx+dy*dy<=165.0f32*165.0f32 {tree_alive[m*tree_count+t]=0.0f32;state[p+TANGO_END]=time+items[6*ITEM_RULE_FIELDS+5];state[p+ITEM_START+6]-=1.0f32;state[p+ORDER]=0.0f32;state[p+TANGO_TARGET]=0.0f32;}
                                else {move_unit(state,grid,geometry,navigation,tree_map,trees,tree_alive,m,tree_count,p,x,y,dt,hero[6],true);}
                            }
                            else if state[p+KIND]>=2.0f32 && state[p+AGGRO_RECOVER]>time {let x=state[p+DEST_X];let y=state[p+DEST_Y];move_unit(state,grid,geometry,navigation,tree_map,trees,tree_alive,m,tree_count,p,x,y,dt,hero[6],true);}
                            else if u<2 && (state[p+ORDER]==1.0f32 || state[p+ORDER]==2.0f32) {let x=state[p+DEST_X];let y=state[p+DEST_Y];let pathfind=state[p+ORDER]==1.0f32;move_unit(state,grid,geometry,navigation,tree_map,trees,tree_alive,m,tree_count,p,x,y,dt,hero[6],pathfind);}
                            else if state[p+TARGET]>=0.0f32 && target<UNITS && state[base+target*STRIDE+ACTIVE]>0.0f32 && state[base+target*STRIDE+GENERATION]==state[p+TARGET_GENERATION] {
                                let q=base+target*STRIDE;let dist=distance(state,p,q);
                                let unseen_building=state[q+KIND]==1.0f32 && state[q+TEAM]!=side && !visible(state,geometry,fow_blockers,static_vision,tree_map,trees,tree_alive,extra_vision,vision_count,m,tree_count,base,side,q,game_time,time);
                                if dist>attack_reach(state,p,q) || state[q+INVULNERABLE]>0.0f32 || unseen_building {
                                    let can_chase=u<2 || creep_orders_ready(state,p,time);
                                    if state[p+KIND]!=1.0f32 && !holding && can_chase {let x=state[q+UNIT_X];let y=state[q+UNIT_Y];move_unit(state,grid,geometry,navigation,tree_map,trees,tree_alive,m,tree_count,p,x,y,dt,hero[6],true);}
                                } else if state[p+KIND]!=1.0f32 || (time>=state[p+ATTACK_READY] && state[p+WINDUP]==0.0f32) {
                                    let desired=(state[q+UNIT_Y]-state[p+UNIT_Y]).atan2(state[q+UNIT_X]-state[p+UNIT_X]);
                                    let heading=if state[p+KIND]!=1.0f32 {state[p+ATTACK_AIM]}else{state[p+FACING]};
                                    let delta=wrap_angle(desired-heading);
                                    let turn=if state[p+KIND]==1.0f32 {6.2831855f32}else if u>=6 {0.5f32}else{hero[6]};
                                    let mut readiness_turn=turn;
                                    if u<2 {
                                        if fresh_hero_attack&(1u32<<u as u32)!=0u32{readiness_turn=turn*0.4712389f32;}
                                        state[p+ATTACK_AIM]=wrap_angle(heading+delta.clamp(-readiness_turn,readiness_turn));
                                        let observed=wrap_angle(desired-state[p+FACING]);
                                        state[p+FACING]=wrap_angle(state[p+FACING]+observed.clamp(-turn*0.4712389f32,turn*0.4712389f32));
                                    }else if state[p+KIND]>=2.0f32 {
                                        let observed=wrap_angle(desired-state[p+FACING]);state[p+FACING]=wrap_angle(state[p+FACING]+observed.clamp(-0.3351032f32,0.3351032f32));
                                        state[p+ATTACK_AIM]=desired;
                                    }else{state[p+FACING]=wrap_angle(state[p+FACING]+delta.clamp(-turn,turn));}
                                    let aligned=state[p+KIND]>=2.0f32 || f32::max(0.0f32,delta.abs()-readiness_turn)<0.2f32;
                                    if time>=state[p+ATTACK_READY] && state[p+WINDUP]==0.0f32 && aligned {
                                        let base_interval=if u<2 {1.6f32} else if state[p+KIND]==4.0f32 {proto[8*UNIT_COUNT+UNIT_ATTACK_INTERVAL]} else if state[p+KIND]==1.0f32 {0.9f32} else {1.0f32};
                                        // Keep the fractional weapon phase across uninterrupted
                                        // cycles. Rounding every new start to this tick changes
                                        // native SF1.17647s into1.2s and accumulates timing drift.
                                        let ready=state[p+ATTACK_READY];
                                        let start=if u<2 && ready>0.0f32 && time-ready<dt+0.0001f32 {ready}else{time};
                                        let point=state[p+UNIT_ATTACK_POINT]*state[p+UNIT_ATTACK_INTERVAL]/base_interval;
                                        state[p+WINDUP]=start+point;
                                        if u<2 {
                                            meta[mb+META_ATTACK_NOMINAL_END+u]=state[p+WINDUP];
                                            meta[mb+META_ATTACK_ANIMATION_START+u]=time;
                                            // Native callback durations preserve the windup
                                            // fraction independently of the weapon cooldown.
                                            let error=meta[mb+META_ATTACK_POINT_ERROR+u];
                                            let ticks=f32::max(0.0f32,(point*30.0f32-error-0.0001f32).ceil());
                                            meta[mb+META_ATTACK_POINT_PENDING+u]=(error+ticks-point*30.0f32).clamp(0.0f32,0.99999f32);
                                            state[p+WINDUP]=(meta[mb]+pregame*30.0f32+ticks)/30.0f32;
                                        }
                                        state[p+WINDUP_TARGET]=target as f32;
                                        state[p+WINDUP_GENERATION]=state[q+GENERATION];
                                    }
                                }
                            } else if u>=6 && u<CREEP_END && state[p+LANE_POSITION]>=0.0f32 {
                                let mut waypoint=state[p+LANE_POSITION] as usize;
                                let dx=state[p+UNIT_X]-lane[waypoint*3];let dy=state[p+UNIT_Y]-lane[waypoint*3+1];
                                if dx*dx+dy*dy<250.0f32*250.0f32 {
                                    let end=if side==0.0f32{radiant_lane_points}else{radiant_lane_points+dire_lane_points};
                                    if waypoint+1<end {waypoint+=1;}
                                    state[p+LANE_POSITION]=waypoint as f32;
                                }
                                move_unit(state,grid,geometry,navigation,tree_map,trees,tree_alive,m,tree_count,p,lane[waypoint*3],lane[waypoint*3+1],dt,0.5f32,true);
                            }
                        }
                        if state[p+WINDUP]>0.0f32 && state[p+WINDUP]<=time {
                            let target=state[p+WINDUP_TARGET] as usize;let q=base+target*STRIDE;
                            if target<UNITS && state[q+ACTIVE]>0.0f32 && state[q+GENERATION]==state[p+WINDUP_GENERATION] && distance(state,p,q)<state[p+UNIT_ATTACK_RANGE]+250.0f32 {
                                let variance=if u<2 {hero[27]}else if u>=FOUNTAIN_START {hero[32]}else if state[p+KIND]==1.0f32 {hero[31]}else{hero[26+state[p+KIND] as usize]};
                                let damage=state[p+UNIT_ATTACK_DAMAGE]+(random(rng,m)*2.0f32-1.0f32)*variance;
                                if state[p+UNIT_PROJECTILE_SPEED]>0.0f32 {launch_shot(state,shots,meta,hero,animation,rng,base,m,u,target,damage,time);}
                                else {let kind=state[p+KIND];hit(state,hero,combat,base,u,target,damage,false,time,kind,side);}
                                if state[q+TEAM]!=side {state[p+REVEAL_END]=time+2.0f32;}
                            }
                            let base_interval=if u<2 {1.6f32} else if state[p+KIND]==4.0f32 {proto[8*UNIT_COUNT+UNIT_ATTACK_INTERVAL]} else if state[p+KIND]==1.0f32 {0.9f32} else {1.0f32};
                            let point=state[p+UNIT_ATTACK_POINT]*state[p+UNIT_ATTACK_INTERVAL]/base_interval;
                            let nominal=if u<2{meta[mb+META_ATTACK_NOMINAL_END+u]}else{state[p+WINDUP]};
                            let continuous=u<2 && state[p+ATTACK_READY]>0.0f32 && (nominal-state[p+ATTACK_READY]-point).abs()<0.0001f32;
                            // Anchor a fresh attack at its actual release tick;
                            // subsequent cycles retain the continuous phase.
                            let release=if u>=2 || continuous{nominal}else{time};
                            state[p+ATTACK_READY]=release+f32::max(0.0f32,state[p+UNIT_ATTACK_INTERVAL]-point);
                            if u<2{meta[mb+META_ATTACK_POINT_ERROR+u]=meta[mb+META_ATTACK_POINT_PENDING+u];}
                            state[p+WINDUP]=0.0f32;
                        }
                        if u<2 {
                            let mut hp=state[p+UNIT_HEALTH_REGEN]+tower_regen(state,p,time);let mut mana=state[p+UNIT_MANA_REGEN];
                            if state[p+FLASK_END]>time {hp+=items[5*ITEM_RULE_FIELDS+4];}if state[p+TANGO_END]>time {hp+=items[6*ITEM_RULE_FIELDS+4];}
                            if state[p+CLARITY_END]>time {mana+=items[7*ITEM_RULE_FIELDS+6];}
                            let dx=state[p+UNIT_X]-fountains[u*3];let dy=state[p+UNIT_Y]-fountains[u*3+1];
                            if dx*dx+dy*dy<1100.0f32*1100.0f32 {hp+=state[p+UNIT_HEALTH_MAX]*0.05f32;mana+=state[p+UNIT_MANA_MAX]*0.06f32;}
                            state[p+UNIT_HEALTH]=f32::min(state[p+UNIT_HEALTH_MAX],state[p+UNIT_HEALTH]+hp*dt);
                            state[p+UNIT_MANA]=f32::min(state[p+UNIT_MANA_MAX],state[p+UNIT_MANA]+mana*dt);
                            if state[p+TP_END]>0.0f32 && state[p+TP_END]<=time {
                                state[p+UNIT_X]=state[p+TP_X];state[p+UNIT_Y]=state[p+TP_Y];state[p+TP_END]=0.0f32;
                                state[p+UNIT_Z]=ground_z(geometry[cell(state[p+UNIT_X],state[p+UNIT_Y])]);
                            }
                        }
                    }
                }
                for s in 0..SHOTS {
                    let k=(m*SHOTS+s)*SHOT_FIELDS;
                    if shots[k]==1.0f32 {
                        let target=shots[k+3] as usize;let q=base+target*STRIDE;
                        if state[q+ACTIVE]==0.0f32 || state[q+GENERATION]!=shots[k+4] {shots[k]=0.0f32;}
                        else if animation[0]>0.0f32 && shots[k+7]<2.0f32 && shots[k+11]>0.0f32 {shots[k+11]=0.0f32;}
                        else {
                            let dx=state[q+UNIT_X]-shots[k+1];let dy=state[q+UNIT_Y]-shots[k+2];let length=(dx*dx+dy*dy).sqrt();
                            if length<=shots[k+5]*dt+8.0f32 {
                                hit(state,hero,combat,base,shots[k+7] as usize,target,shots[k+6],false,time,shots[k+9],shots[k+8]);shots[k]=0.0f32;
                                if target<2 {meta[mb+10]=meta[mb];meta[mb+13]+=1.0f32;}
                            }
                            else {shots[k+1]+=dx/length*shots[k+5]*dt;shots[k+2]+=dy/length*shots[k+5]*dt;}
                        }
                    } else if shots[k]==2.0f32 {
                        let step=f32::min(shots[k+5]*dt,shots[k+3]);
                        let width=hero[14]+(hero[15]-hero[14])*(shots[k+11]+step)/hero[16];
                        for v in 0..UNITS {
                            let q=base+v*STRIDE;let word=(m*SHOTS+s)*8+v/32;let bit=1u32<<(v%32) as u32;
                            if state[q+ACTIVE]>0.0f32 && state[q+TEAM]!=shots[k+8] && state[q+KIND]!=1.0f32 && shot_hits[word]&bit==0u32 {
                                let dx=state[q+UNIT_X]-shots[k+1];let dy=state[q+UNIT_Y]-shots[k+2];
                                let along=(dx*shots[k+9]+dy*shots[k+10]).clamp(0.0f32,step);
                                let ex=dx-shots[k+9]*along;let ey=dy-shots[k+10]*along;
                                if ex*ex+ey*ey<=width*width {
                                    hit(state,hero,combat,base,shots[k+7] as usize,v,shots[k+6],true,time,0.0f32,shots[k+8]);shot_hits[word]=shot_hits[word]|bit;
                                    let was_feared=state[q+FEAR_END]>time;
                                    state[q+FEAR_END]=time+f32::min(hero[20],f32::max(0.0f32,state[q+FEAR_END]-time)+hero[19]);
                                    let mut fixed_goal=false;if hero.len()>41 {fixed_goal=hero[41]>0.0f32;}
                                    if !fixed_goal || !was_feared {
                                    state[q+FEAR_X]=state[base+shots[k+7] as usize*STRIDE+UNIT_X];state[q+FEAR_Y]=state[base+shots[k+7] as usize*STRIDE+UNIT_Y];
                                    if fixed_goal {
                                        let dx=state[q+UNIT_X]-state[q+FEAR_X];let dy=state[q+UNIT_Y]-state[q+FEAR_Y];let length=(dx*dx+dy*dy).sqrt().max(1.0f32);
                                        let mut radius=hero[39];if hero[40]==0.0f32 {radius+=length;}
                                        state[q+FEAR_X]+=dx/length*radius;state[q+FEAR_Y]+=dy/length*radius;
                                    }}
                                    state[q+WINDUP]=0.0f32;state[q+CAST_END]=0.0f32;state[q+CAST]=0.0f32;state[q+TP_END]=0.0f32;
                                    // Installed build6969: BotAPI MR unchanged and
                                    // paired100-magic probes deal73 before/during
                                    // fear in all5 native scenarios (requiem003).
                                    // The -5/-10/-15 resource entry is not applied
                                    // by this runtime; do not invent amplification.
                                    state[q+MRES_END]=0.0f32;state[q+MRES_REDUCTION]=0.0f32;
                                    state[q+SLOW_PERCENT]=hero[26];
                                }
                            }
                        }
                        shots[k+1]+=shots[k+9]*step;shots[k+2]+=shots[k+10]*step;shots[k+11]+=step;shots[k+3]-=step;
                        if shots[k+3]<=0.0f32 {shots[k]=0.0f32;}
                    }
                }
                for u in 0..UNITS {
                    let p=base+u*STRIDE;
                    if state[p+ACTIVE]>0.0f32 {
                        let previous_hp=state[p+UNIT_HEALTH];
                        state[p+UNIT_HEALTH]=f32::max(0.0f32,state[p+UNIT_HEALTH]-state[p+DAMAGE]);state[p+DAMAGE]=0.0f32;
                        if state[p+UNIT_HEALTH]==0.0f32 {
                            state[p+ACTIVE]=0.0f32;let source=state[p+CREDIT] as usize;
                            if u<2 {
                                if audit_enabled {crate::ggd_metrics::hero_death(teacher_audit,m,u,time);}
                                let first=state[base+DEATHS]+state[base+STRIDE+DEATHS]==0.0f32;
                                let enemy=1-u;let q=base+enemy*STRIDE;
                                let hero_kill=state[p+CREDIT]>=0.0f32 && source==enemy;
                                // Empirical level1/pre-game economy, not a general
                                // Dota bounty formula: native hero first kill326,
                                // tower first kill202, later tower deaths133.
                                // Later hero bounties remain an explicit approximation;
                                // the mode remembers the first death and ends at0.
                                state[q+UNIT_GOLD]+=if first {if hero_kill {326.0f32}else{202.0f32}}else{133.0f32};
                                if state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>state[q+DAMAGE] {
                                    let mut total_xp=state[p+XP];for l in 1..30 {if (l as f32)<state[p+UNIT_LEVEL] {total_xp+=experience[l-1];}}
                                    if distance(state,p,q)<1500.0f32 {add_xp(state,proto,hero,items,experience,q,enemy,(100.0f32+total_xp*0.13f32).floor());}
                                    if hero_kill {state[q+UNIT_SOULS]=f32::min(hero[12],state[q+UNIT_SOULS]+4.0f32);level_stats(state,proto,hero,items,q,enemy);}
                                }
                                state[p+DEATHS]+=1.0f32;state[p+GENERATION]+=1.0f32;
                                // Level1 delay and economy/soul loss checked in both
                                // native directions. Higher levels are terminal in
                                // this mode; their respawn delay is not simulated.
                                state[p+RESPAWN_AT]=time+12.0f32;
                                let worth=net_worth(state,items,p);let loss=f32::min(f32::max(0.0f32,state[p+UNIT_GOLD]-state[p+RELIABLE_GOLD]),(worth/40.0f32).floor());
                                state[p+UNIT_GOLD]-=loss;
                                state[p+UNIT_SOULS]=(state[p+UNIT_SOULS]*0.7f32).ceil();
                                state[p+ORDER]=0.0f32;state[p+TARGET]=-1.0f32;state[p+WINDUP_TARGET]=-1.0f32;
                                state[p+WINDUP]=0.0f32;state[p+ATTACK_READY]=0.0f32;state[p+CAST_END]=0.0f32;state[p+CAST]=0.0f32;state[p+TP_END]=0.0f32;
                                state[p+FEAR_END]=0.0f32;state[p+STACK_END]=0.0f32;state[p+UNIT_RAZE_STACKS]=0.0f32;state[p+MRES_END]=0.0f32;
                                state[p+FLASK_END]=0.0f32;state[p+CLARITY_END]=0.0f32;state[p+TANGO_END]=0.0f32;state[p+TANGO_TARGET]=0.0f32;
                                state[p+HOLD]=0.0f32;state[p+REVEAL_END]=0.0f32;state[p+TIER1_AURA_END]=0.0f32;state[p+HIGH_AURA_END]=0.0f32;
                                // No command observed in the old life may execute
                                // after respawn. New dead-life observations are CONT.
                                for slot in 0..4 {commands[(m*2+u)*4*COMMAND_FIELDS+slot*COMMAND_FIELDS+7]=0.0f32;}
                            }
                            if u>=6 && u<CREEP_END {
                                if audit_enabled {crate::scripted_teacher::death(state,shots,meta,teacher_memory,teacher_forecasts,teacher_audit,m,u,time,previous_hp);}
                                if state[p+CREDIT]>=0.0f32 && source<2 {
                                    let q=base+source*STRIDE;
                                    if state[q+TEAM]==state[p+TEAM] {state[q+UNIT_DENIES]+=1.0f32;}
                                    else {state[q+UNIT_LAST_HITS]+=1.0f32;state[q+UNIT_GOLD]+=state[p+BOUNTY];}
                                    state[q+UNIT_SOULS]=f32::min(hero[12],state[q+UNIT_SOULS]+1.0f32);
                                    level_stats(state,proto,hero,items,q,source);
                                }
                                for h in 0..2 {
                                    let q=base+h*STRIDE;
                                    if state[q+TEAM]!=state[p+TEAM] && state[q+ACTIVE]>0.0f32 && state[q+UNIT_HEALTH]>0.0f32 && distance(state,p,q)<1500.0f32 {
                                        let denied=state[p+CREDIT]>=0.0f32 && state[p+CREDIT_TEAM]==state[p+TEAM];
                                        let amount=state[p+XP_BOUNTY]*if denied {0.5f32} else {1.0f32};
                                        add_xp(state,proto,hero,items,experience,q,h,amount);
                                    }
                                }
                            }
                        }
                    }
                }
                meta[mb]+=1.0f32;meta[mb+3]+=1.0f32;
                // native_executor checks GetHeroDeaths even after a pregame
                // respawn. A recovered hero must still lose when time reaches0.
                let loss_a=state[base+DEATHS]>0.0f32 || state[base+UNIT_HEALTH]<=0.0f32 || state[base+2*STRIDE+UNIT_HEALTH]<=0.0f32;
                let loss_b=state[base+STRIDE+DEATHS]>0.0f32 || state[base+STRIDE+UNIT_HEALTH]<=0.0f32 || state[base+4*STRIDE+UNIT_HEALTH]<=0.0f32;
                if game_time>=0.0f32 && (loss_a || loss_b || game_time>=600.0f32) {
                    if loss_a && !loss_b {result=-1.0f32;}if loss_b && !loss_a {result=1.0f32;}
                    if !loss_a && !loss_b {
                        let delta=state[base+UNIT_LAST_HITS]-state[base+STRIDE+UNIT_LAST_HITS];
                        if delta>0.0f32 {result=1.0f32;}if delta<0.0f32 {result=-1.0f32;}
                    }
                    meta[mb+2]=1.0f32;meta[mb+4]+=1.0f32;terminals[m*2]=1.0f32;terminals[m*2+1]=1.0f32;
                    if result>0.0f32 {meta[mb+14]+=1.0f32;}else if result<0.0f32 {meta[mb+15]+=1.0f32;}else{meta[mb+16]+=1.0f32;}
                    meta[mb+17]=result;
                }
            }
        }
        let after=if meta[mb+2]>0.0f32 {0.0f32} else {potential(state,base)};
        rewards[m*2]=result+after-before;rewards[m*2+1]=-rewards[m*2];meta[mb+6]+=1.0f32;
    }
}

impl World {
    pub fn advance(&self,actions:Tensor<2,Int>,refs:&CubeTensor,rng:&crate::sampling::Sampler)->(Tensor<1>,Tensor<1>){
        let targets=refs.meta.shape()[1];
        let actions=primitive(actions.float().without_autodiff());
        enqueue::launch(&self.state.client,CubeCount::Static((self.matches*2).div_ceil(64) as u32,1,1),CubeDim::new_1d(64),
            argument(&actions),argument(refs),argument(&self.commands),argument(&self.state),argument(&self.meta),argument(&self.bases),self.matches*2,targets,self.tree_count);
        self.step_ticks(rng,4)
    }
    /// A one-tick launch is used only by differential validation. Training always batches four ticks.
    pub fn step_ticks(&self,rng:&crate::sampling::Sampler,ticks:usize)->(Tensor<1>,Tensor<1>){
        self.step_ticks_with_waves(rng,ticks,true)
    }
    /// Match native controlled fixtures with dota_creeps_no_spawning1.
    /// The production entrypoint above always enables natural waves.
    pub(crate) fn step_ticks_with_waves(&self,rng:&crate::sampling::Sampler,ticks:usize,spawn_waves:bool)->(Tensor<1>,Tensor<1>){
        self.step_ticks_audit(rng,ticks,spawn_waves,None)
    }
    pub(crate) fn advance_teacher(&self,actions:Tensor<2,Int>,refs:&CubeTensor,rng:&crate::sampling::Sampler,teacher:&crate::scripted_teacher::Teacher,spawn_waves:bool)->(Tensor<1>,Tensor<1>){
        let targets=refs.meta.shape()[1];let actions=primitive(actions.float().without_autodiff());
        enqueue::launch(&self.state.client,CubeCount::Static((self.matches*2).div_ceil(64)as u32,1,1),CubeDim::new_1d(64),
            argument(&actions),argument(refs),argument(&self.commands),argument(&self.state),argument(&self.meta),argument(&self.bases),self.matches*2,targets,self.tree_count);
        self.step_ticks_audit(rng,4,spawn_waves,Some(teacher))
    }
    fn step_ticks_audit(&self,rng:&crate::sampling::Sampler,ticks:usize,spawn_waves:bool,teacher:Option<&crate::scripted_teacher::Teacher>)->(Tensor<1>,Tensor<1>){
        simulate::launch(&self.state.client,CubeCount::Static(self.matches.div_ceil(32) as u32,1,1),CubeDim::new_1d(32),
            argument(&self.state),argument(&self.shots),argument(&self.shot_hits),argument(&self.meta),argument(&self.commands),
            argument(teacher.map_or(&self.meta,|t|&t.memory)),argument(teacher.map_or(&self.meta,|t|&t.forecasts)),argument(teacher.map_or(&self.rewards,|t|&t.audit)),argument(&self.rewards),argument(&self.terminals),
            argument(&self.tree_alive),unsafe{BufferArg::from_raw_parts(rng.state.clone(),1)},argument(&self.collision_grid),argument(&self.prototypes),argument(&self.spells),
            argument(&self.items),argument(&self.hero),argument(&self.sf_projectile_animation),argument(&self.combat),argument(&self.experience),argument(self.movement_geometry.as_ref().unwrap_or(&self.geometry)),argument(&self.fow_blockers),argument(&self.static_vision),argument(&self.navigation),argument(&self.trees),argument(&self.tree_map),argument(&self.bases),argument(&self.spawns),argument(&self.lane),argument(&self.fountains),argument(&self.extra_vision),
            self.matches,self.tree_count,self.engine.extra_vision.len(),self.engine.pre_game_seconds,ticks,self.lane_lengths[0],self.lane_lengths[1],spawn_waves,teacher.is_some());
        // Addition allocates independent storage: the next step mutates the
        // simulator's buffers, whereas rollout data must remain immutable.
        (tensor::<1>(self.rewards.clone())+0.0f32,tensor::<1>(self.terminals.clone())+0.0f32)
    }
}







