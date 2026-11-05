//! Deterministic v10-to-CHiO labels; execution consumes this same tensor.
use burn::tensor::{Tensor,Int};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use crate::{world::*,contract::*,observe::{self,Observed},chio_actions::{self,Masks},runtime::{argument,primitive,tensor}};

pub struct Translator {pub counters:CubeTensor}
impl Translator {
    pub fn new(w:&World)->Self {Self{counters:primitive(Tensor::<2>::zeros([w.matches*2,4],&w.device))}}
    pub fn labels(&self,w:&World,obs:&Observed,masks:&Masks,legacy:Tensor<2,Int>)->Tensor<2,Int>{
        let n=w.matches*2;let legacy=primitive(legacy.float());let public=primitive(obs.packed.clone());let out=primitive(Tensor::<2>::zeros([n,4],&w.device));
        translate::launch(&w.state.client,CubeCount::Static(n.div_ceil(32)as u32,1,1),CubeDim::new_1d(32),
            argument(&w.state),argument(&w.commands),argument(&legacy),argument(&public),argument(&obs.refs),argument(&masks.packed),argument(&out),argument(&self.counters),n);
        tensor::<2>(out).int()
    }
}

#[cube(launch)]
fn translate(state:&[f32],commands:&[f32],legacy:&[f32],obs:&[f32],refs:&[f32],mask:&[f32],out:&mut [f32],counters:&mut [f32],actors:usize){
    let a=ABSOLUTE_POS;if a<actors {
        let m=a/2;let side=a%2;let base=m*UNITS*STRIDE;let p=base+side*STRIDE;let r=a*4;
        let mut action=legacy[a*6]as usize;let mut item=legacy[a*6+2]as usize;
        let mut target=refs[a*observe::TARGETS+legacy[a*6+3]as usize];
        let mut x=state[p+UNIT_X]+(legacy[a*6+4]-64.0f32)*16.0f32;let mut y=state[p+UNIT_Y]+(legacy[a*6+5]-64.0f32)*16.0f32;
        for i in 0..4 {out[r+i]=0.0f32;}let mut kind=1usize.runtime();let mut unsupported=false;
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
            let dx=x-state[p+UNIT_X];let dy=y-state[p+UNIT_Y];
            if dx*dx+dy*dy>75.0f32*75.0f32 {
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
