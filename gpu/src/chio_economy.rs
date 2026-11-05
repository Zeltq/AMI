//! Common CHiO purchasing, independent of teacher/model policy decisions.
use burn::tensor::Tensor;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use crate::{world::*,contract::*,observe::{self,Observed},runtime::{argument,primitive,tensor}};

pub struct Economy {memory:CubeTensor}
impl Economy {
    pub fn new(w:&World)->Self {Self{memory:primitive(Tensor::<2>::zeros([w.matches*2,2],&w.device))}}
    pub fn choose(&self,w:&World,obs:&Observed)->Tensor<1> {
        assert!(w.engine.chio_direct && w.engine.decision_ticks==6);
        let n=w.matches*2;let out=primitive(Tensor::<1>::zeros([n],&w.device));let packed=primitive(obs.packed.clone());
        choose::launch(&w.state.client,CubeCount::Static(n.div_ceil(32)as u32,1,1),CubeDim::new_1d(32),
            argument(&w.state),argument(&w.meta),argument(&w.commands),argument(&w.items),argument(&packed),argument(&self.memory),argument(&out),n,w.engine.pre_game_seconds);
        tensor::<1>(out)
    }
}

#[cube]
fn owned(s:&[f32],p:usize,item:usize)->f32 {
    s[p+ITEM_START+item]+s[p+BACKPACK_START+item]+s[p+STASH_START+item]+s[p+COURIER_START+item]
}

#[cube(launch)]
fn choose(s:&[f32],meta:&[f32],commands:&[f32],items:&[f32],obs:&[f32],memory:&mut [f32],out:&mut [f32],actors:usize,pregame:f32){
    let a=ABSOLUTE_POS;if a<actors {
        let m=a/2;let p=(m*UNITS+a%2)*STRIDE;let r=a*observe::WIDTH;let epoch=meta[m*META+1]+1.0f32;
        if memory[a*2]!=epoch {memory[a*2]=epoch;memory[a*2+1]=0.0f32;}
        let tango=owned(s,p,6);let flask=owned(s,p,5);let mango=owned(s,p,4);
        if tango>0.0f32 {memory[a*2+1]=1.0f32;}
        let mut pending=false;
        for slot in 0..4 {let k=(a*4+slot)*COMMAND_FIELDS;if commands[k+7]>0.0f32 && commands[k+2]>0.0f32 {pending=true;}}
        out[a]=0.0f32;
        if !pending && s[p+ACTIVE]>0.0f32 && s[p+UNIT_HEALTH]>0.0f32 {
            if meta[m*META]<0.0f32 && memory[a*2+1]==0.0f32 {
                if obs[r+observe::BUY_MASK+5]>0.0f32 {out[a]=5.0f32;}
            } else {
                let time=(meta[m*META]+pregame*30.0f32)/30.0f32;
                let healing=(s[p+FLASK_END]-time).max(0.0f32)*items[5*ITEM_RULE_FIELDS+4]
                    +(s[p+TANGO_END]-time).max(0.0f32)*items[6*ITEM_RULE_FIELDS+4];
                let missing=s[p+UNIT_HEALTH_MAX]-s[p+UNIT_HEALTH];let low=s[p+UNIT_LEVEL]<5.0f32;
                let max_flask=if low {2.0f32}else{3.0f32};let max_mango=if low {3.0f32}else{4.0f32};
                let min_mana=s[p+UNIT_MANA_MAX].min(if low {360.0f32}else{450.0f32});
                let need_flask=(flask<1.0f32 && tango==0.0f32) || (flask<max_flask && missing>400.0f32*flask+112.0f32*tango+healing-100.0f32);
                let need_mango=(missing<100.0f32 || !need_flask) && min_mana>110.0f32*mango+s[p+UNIT_MANA] && mango<max_mango;
                if need_flask && obs[r+observe::BUY_MASK+4]>0.0f32 {out[a]=4.0f32;}
                else if need_mango && obs[r+observe::BUY_MASK+3]>0.0f32 {out[a]=3.0f32;}
                else if !need_flask && !need_mango && obs[r+observe::BUY_MASK+1]>0.0f32 {out[a]=1.0f32;}
            }
        }
    }
}
