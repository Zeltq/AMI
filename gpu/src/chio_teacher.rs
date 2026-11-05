//! Deterministic v10-to-CHiO labels; execution consumes this same tensor.
use burn::tensor::{Tensor,Int};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use crate::{world::*,contract::*,observe::{self,Observed},chio_actions::{self,Masks},runtime::{argument,primitive,tensor}};

pub struct Translator {pub counters:CubeTensor, goals:CubeTensor}
impl Translator {
    pub fn new(w:&World)->Self {Self{counters:primitive(Tensor::<2>::zeros([w.matches*2,4],&w.device)),goals:primitive(Tensor::<2>::zeros([w.matches*2,4],&w.device))}}
    pub fn labels(&self,w:&World,obs:&Observed,masks:&Masks,legacy:Tensor<2,Int>)->Tensor<2,Int>{
        let n=w.matches*2;let legacy=primitive(legacy.float());let public=primitive(obs.packed.clone());let out=primitive(Tensor::<2>::zeros([n,4],&w.device));
        translate::launch(&w.state.client,CubeCount::Static(n.div_ceil(32)as u32,1,1),CubeDim::new_1d(32),
            argument(&w.state),argument(&w.commands),argument(&legacy),argument(&public),argument(&obs.refs),argument(&masks.packed),argument(&out),argument(&self.counters),argument(&self.goals),argument(&w.meta),argument(w.movement_geometry.as_ref().unwrap_or(&w.geometry)),argument(&w.navigation),argument(&w.tree_map),argument(&w.trees),argument(&w.tree_alive),n,w.tree_count);
        tensor::<2>(out).int()
    }
}

#[cube(launch)]
fn translate(state:&[f32],commands:&[f32],legacy:&[f32],obs:&[f32],refs:&[f32],mask:&[f32],out:&mut [f32],counters:&mut [f32],goals:&mut [f32],meta:&[f32],geometry:&[f32],navigation:&[u32],tree_map:&[f32],trees:&[f32],alive:&[f32],actors:usize,tree_count:usize){
    let a=ABSOLUTE_POS;if a<actors {
        let m=a/2;let side=a%2;let base=m*UNITS*STRIDE;let p=base+side*STRIDE;let r=a*4;
        if goals[r+3]!=meta[m*META+1] {goals[r+2]=0.0f32;goals[r+3]=meta[m*META+1];}
        let mut action=legacy[a*6]as usize;let mut item=legacy[a*6+2]as usize;
        let mut target=refs[a*observe::TARGETS+legacy[a*6+3]as usize];
        let mut x=state[p+UNIT_X]+(legacy[a*6+4]-64.0f32)*16.0f32;let mut y=state[p+UNIT_Y]+(legacy[a*6+5]-64.0f32)*16.0f32;
        for i in 0..4 {out[r+i]=0.0f32;}let mut kind=1usize.runtime();let mut unsupported=false;
        if action==1 || action==2 {goals[r]=x;goals[r+1]=y;goals[r+2]=1.0f32;}else if action!=0 {goals[r+2]=0.0f32;}
        if action==0 {
            counters[r+1]+=1.0f32;action=state[p+ORDER]as usize;target=state[p+TARGET];x=state[p+DEST_X];y=state[p+DEST_Y];
            if action==3 && target>=0.0f32 && target<UNITS as f32 && state[base+target as usize*STRIDE+GENERATION]!=state[p+TARGET_GENERATION] {target=-1.0f32;}
            if action==6 {action=10;item=6;}
            // Preserve the most recent accepted intent during the execution
            // delay; replacing a pending attack by STOP would cancel its windup.
            let mut latest=(-1e20f32).runtime();
            for slot in 0..4 {let k=(a*4+slot)*COMMAND_FIELDS;let command=commands[k+1]as usize;
                let stale_skill=command>=6 && command<=9 && obs[a*observe::WIDTH+observe::ACTION_MASK+command]<=0.0f32;
                if commands[k+7]>0.0f32 && command>0 && commands[k]>latest && !stale_skill {
                    latest=commands[k];action=command;item=commands[k+3]as usize;target=commands[k+4];x=commands[k+5];y=commands[k+6];
                    if action==3 && target>=0.0f32 && target<UNITS as f32 && state[base+target as usize*STRIDE+GENERATION]!=commands[k+8] {target=-1.0f32;}
                }
            }
            if mask[a*chio_actions::MASK_WIDTH]==0.0f32 {
                for skill in 6usize..10usize {if mask[a*chio_actions::MASK_WIDTH+skill]>0.0f32 {action=skill;}}
            }
        }
        if action==1 || action==2 {
            if goals[r+2]>0.0f32 {x=goals[r];y=goals[r+1];}
            let original_dx=x-state[p+UNIT_X];let original_dy=y-state[p+UNIT_Y];
            let point=path_lookahead(state,refs,a,geometry,navigation,tree_map,trees,alive,m*tree_count,state[p+UNIT_X],state[p+UNIT_Y],x,y);
            let dx=point.0-state[p+UNIT_X];let dy=point.1-state[p+UNIT_Y];
            if original_dx*original_dx+original_dy*original_dy>75.0f32*75.0f32 && dx*dx+dy*dy>1.0f32 {
                kind=0;let raw=(dy.atan2(dx)/0.1745329252f32+0.5f32).floor()as i32;
                out[r+1]=((raw+36)%36)as f32;
            }
        }else if action==3 {
            let mut found=false;
            if target>=0.0f32 && target<UNITS as f32 {
                let u=target as usize;let q=base+u*STRIDE;
                if u==1-side {kind=4;found=true;}
                else if u==if side==0{4usize}else{2usize} {kind=5;found=true;}
                else if u>=6 && u<CREEP_END && state[q+KIND]>=2.0f32 && state[q+KIND]<=4.0f32 {
                    for i in 0..observe::CREEPS {if refs[a*observe::TARGETS+2+i]==target {
                        kind=if state[q+TEAM]==side as f32 {3usize}else{2usize};out[r+if kind==3{3usize}else{2usize}]=i as f32;found=true;
                    }}
                }
            }if !found {unsupported=true;}
        }else if action>=6 && action<=9 {kind=action;}
        else if action==10 {if item==5 {kind=10;}else if item==4 {kind=11;}else if item==6 {kind=12;}else{unsupported=true;}}
        else if action!=0 && action!=4 {unsupported=true;}
        let mut valid=mask[a*chio_actions::MASK_WIDTH+kind]>0.0f32;
        if kind==2 || kind==3 {let index=out[r+if kind==3{3usize}else{2usize}]as usize;let offset=if kind==2{13usize}else{13+observe::CREEPS};valid=valid && mask[a*chio_actions::MASK_WIDTH+offset+index]>0.0f32;}
        if unsupported || !valid {kind=1;for i in 1..4{out[r+i]=0.0f32;}counters[r+2]+=if unsupported{1.0f32}else{0.0f32};counters[r+3]+=if !valid{1.0f32}else{0.0f32};}
        out[r]=kind as f32;counters[r]+=1.0f32;
    }
}

// Teacher-only use of the existing static flow atlas. Neither the model nor
// MOVE_DIRECT executes this navigation. Sample the SF hull and live trees.
#[cube]
fn clear_ray(geometry:&[f32],tree_map:&[f32],trees:&[f32],alive:&[f32],offset:usize,ox:f32,oy:f32,x:f32,y:f32)->bool {
    let dx=x-ox;let dy=y-oy;let distance=(dx*dx+dy*dy).sqrt();let count=(distance/8.0f32).ceil().max(1.0f32)as usize;let mut clear=true;
    for sample in 1..count+1 {if clear {let f=sample as f32/count as f32;let px=ox+dx*f;let py=oy+dy*f;
        if geometry[crate::world::cell(px,py)+1]<0.5f32 {clear=false;}
        for corner in 0..8 {let angle=corner as f32*0.7853982f32;
            if geometry[crate::world::cell(px+24.0f32*angle.cos(),py+24.0f32*angle.sin())+1]<0.5f32 {clear=false;}}
        let id=tree_map[crate::world::cell(px,py)/2]as usize;if id>0 {let t=id-1;let tx=px-trees[t*3];let ty=py-trees[t*3+1];
            if alive[offset+t]>0.0f32 && tx*tx+ty*ty<88.0f32*88.0f32 {clear=false;}}
    }}clear
}
#[cube]
fn lookahead(geometry:&[f32],nav:&[u32],tree_map:&[f32],trees:&[f32],alive:&[f32],offset:usize,ox:f32,oy:f32,gx:f32,gy:f32)->(f32,f32) {
    let dx=gx-ox;let dy=gy-oy;let length=(dx*dx+dy*dy).sqrt();let scale=500.0f32/length.max(500.0f32);
    let direct_x=ox+dx*scale;let direct_y=oy+dy*scale;let mut bx=ox;let mut by=oy;
    if clear_ray(geometry,tree_map,trees,alive,offset,ox,oy,direct_x,direct_y) {bx=direct_x;by=direct_y;}
    else {
        let goal_x=((gx+8192.0f32)/256.0f32).floor().clamp(0.0f32,63.0f32)as usize;
        let goal_y=((gy+8192.0f32)/256.0f32).floor().clamp(0.0f32,63.0f32)as usize;let start=(goal_y*64+goal_x)*8192;
        let mut nx=((ox+8192.0f32)/64.0f32).floor().clamp(0.0f32,255.0f32)as usize;
        let mut ny=((oy+8192.0f32)/64.0f32).floor().clamp(0.0f32,255.0f32)as usize;
        let node=ny*256+nx;let first=(nav[start+node/8]>>((node%8)*4)as u32)&15u32;
        if first==9u32 {let mut best=(1e20f32).runtime();let cx=nx;let cy=ny;
            for candidate in 0..49 {let tx=(cx as i32+candidate as i32%7-3).clamp(0,255)as usize;let ty=(cy as i32+candidate as i32/7-3).clamp(0,255)as usize;
                let node=ty*256+tx;let flow=(nav[start+node/8]>>((node%8)*4)as u32)&15u32;
                let px=-8192.0f32+(tx as f32+0.5f32)*64.0f32;let py=-8192.0f32+(ty as f32+0.5f32)*64.0f32;let distance=(px-ox)*(px-ox)+(py-oy)*(py-oy);
                if flow<9u32 && distance<best && clear_ray(geometry,tree_map,trees,alive,offset,ox,oy,px,py) {best=distance;nx=tx;ny=ty;bx=px;by=py;}
            }
        }
        let mut following=true;let mut farthest=(bx-ox)*(bx-ox)+(by-oy)*(by-oy);
        for _step in 0..64 {if following {
            let node=ny*256+nx;let flow=(nav[start+node/8]>>((node%8)*4)as u32)&15u32;
            if flow<8u32 {
                let fx=if flow==0u32 || flow==1u32 || flow==7u32 {1i32}else if flow>=3u32 && flow<=5u32 {-1i32}else{0i32};
                let fy=if flow>=1u32 && flow<=3u32 {1i32}else if flow>=5u32 {-1i32}else{0i32};
                nx=(nx as i32+fx).clamp(0,255)as usize;ny=(ny as i32+fy).clamp(0,255)as usize;
                let px=-8192.0f32+(nx as f32+0.5f32)*64.0f32;let py=-8192.0f32+(ny as f32+0.5f32)*64.0f32;let distance=(px-ox)*(px-ox)+(py-oy)*(py-oy);
                if distance>500.0f32*500.0f32 {following=false;}
                else if distance>farthest && clear_ray(geometry,tree_map,trees,alive,offset,ox,oy,px,py) {farthest=distance;bx=px;by=py;}
            }else{following=false;}
        }}
    }(bx,by)
}

// Same nine local steering choices as ordinary walking, used to construct a
// teacher path around unit hulls absent from the static flow atlas.
#[cube]
fn body_ray(state:&[f32],refs:&[f32],a:usize,ox:f32,oy:f32,x:f32,y:f32)->bool {
    let base=a/2*UNITS*STRIDE;let side=a%2;let mut clear=true;let dx=x-ox;let dy=y-oy;let length2=(dx*dx+dy*dy).max(0.001f32);
    for u in 0..UNITS {let p=base+u*STRIDE;
        if u!=side && state[p+ACTIVE]>0.0f32 && state[p+UNIT_HEALTH]>0.0f32 {
            let mut known=state[p+KIND]==1.0f32;
            if !known {for i in 0..2+observe::CREEPS {if refs[a*observe::TARGETS+i]==u as f32{known=true;}}}
            if known {let radius=24.0f32+crate::physics::hull(state[p+KIND]);let sx=state[p+UNIT_X]-ox;let sy=state[p+UNIT_Y]-oy;
                let f=((sx*dx+sy*dy)/length2).clamp(0.0f32,1.0f32);let ex=sx-dx*f;let ey=sy-dy*f;
                if ex*ex+ey*ey<radius*radius && ex*ex+ey*ey<sx*sx+sy*sy {clear=false;}
            }
        }
    }clear
}
#[cube]
fn path_lookahead(state:&[f32],refs:&[f32],a:usize,geometry:&[f32],nav:&[u32],tree_map:&[f32],trees:&[f32],alive:&[f32],offset:usize,ox:f32,oy:f32,gx:f32,gy:f32)->(f32,f32) {
    let target=lookahead(geometry,nav,tree_map,trees,alive,offset,ox,oy,gx,gy);
    let mut bx=target.0;let mut by=target.1;
    if (bx-ox)*(bx-ox)+(by-oy)*(by-oy)<1.0f32 || !body_ray(state,refs,a,ox,oy,bx,by) {
        bx=ox;by=oy;let mut px=ox;let mut py=oy;let mut farthest=0.0f32.runtime();let mut following=true;
        for _step in 0..64 {if following {
            let local=lookahead(geometry,nav,tree_map,trees,alive,offset,px,py,gx,gy);let angle=if (local.0-px)*(local.0-px)+(local.1-py)*(local.1-py)>1.0f32 {(local.1-py).atan2(local.0-px)}else{(gy-py).atan2(gx-px)};let mut moved=false;
            for candidate in 0..9 {if !moved {let rank=(candidate+1)/2;let sign=if candidate%2==0{1.0f32}else{-1.0f32};let direction=angle+rank as f32*0.3926991f32*sign;
                let nx=px+direction.cos()*16.0f32;let ny=py+direction.sin()*16.0f32;
                if clear_ray(geometry,tree_map,trees,alive,offset,px,py,nx,ny) && body_ray(state,refs,a,px,py,nx,ny) {px=nx;py=ny;moved=true;}
            }}
            let distance=(px-ox)*(px-ox)+(py-oy)*(py-oy);
            if distance>500.0f32*500.0f32 || !moved {following=false;}
            else if distance>farthest && clear_ray(geometry,tree_map,trees,alive,offset,ox,oy,px,py) && body_ray(state,refs,a,ox,oy,px,py) {bx=px;by=py;farthest=distance;}
        }}
    }(bx,by)
}

#[cfg(test)]
pub fn test_waypoint(w:&World,x:f32,y:f32)->Vec<f32> {
    let obs=w.observe(observe::initial_previous(w.matches*2,&w.device));let out=primitive(Tensor::<1>::zeros([2],&w.device));
    waypoint_probe::launch(&w.state.client,CubeCount::Static(1,1,1),CubeDim::new_1d(1),argument(&w.state),argument(&obs.refs),argument(w.movement_geometry.as_ref().unwrap_or(&w.geometry)),argument(&w.navigation),argument(&w.tree_map),argument(&w.trees),argument(&w.tree_alive),argument(&out),x,y);
    tensor::<1>(out).into_data().try_to_vec::<f32>().unwrap()
}
#[cfg(test)]
#[cube(launch)]
fn waypoint_probe(state:&[f32],refs:&[f32],geometry:&[f32],nav:&[u32],tree_map:&[f32],trees:&[f32],alive:&[f32],out:&mut [f32],x:f32,y:f32){
    let point=path_lookahead(state,refs,0,geometry,nav,tree_map,trees,alive,0,state[UNIT_X],state[UNIT_Y],x,y);out[0]=point.0;out[1]=point.1;
}
