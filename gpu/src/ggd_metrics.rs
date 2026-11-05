//! Read-only gameplay telemetry shared by demonstrations and GGD evaluations.
//! Compact counters stay on GPU. Reserved audit words record actual cast/death
//! events inside physics; they never participate in observations or decisions.
use burn::tensor::{Tensor,Int};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use crate::{world::*,contract::*,runtime::{primitive,tensor,argument},scripted_teacher::AUDIT_FIELDS};

pub const CAST_OFFSET:usize=41;
pub const CAST_FIELDS:usize=11;
pub const FIELDS:usize=224;
pub const ACTIONS_AT:usize=40;
pub const BUYS_AT:usize=51;
pub const ITEMS_AT:usize=58;
pub const MINUTES_AT:usize=72;
pub const MINUTE_FIELDS:usize=12;
const _:()=assert!(CAST_OFFSET+2*CAST_FIELDS<=crate::scripted_teacher::AUDIT_HEADER);
const _:()=assert!(MINUTES_AT+12*MINUTE_FIELDS<=FIELDS);

#[cube]
pub fn cast(audit:&mut [f32],m:usize,side:usize,spell:usize,heroes:f32,creeps:f32,stacks:f32){
    let r=m*AUDIT_FIELDS+CAST_OFFSET+side*CAST_FIELDS;
    audit[r+spell]+=1.0f32;audit[r+4]+=heroes;audit[r+5]+=creeps;
    if heroes>0.0f32 {audit[r+6]+=1.0f32;}
    if heroes+creeps>0.0f32 {audit[r+7]+=1.0f32;}
    audit[r+8]+=stacks;
}

#[cube]
pub fn cancel(audit:&mut [f32],m:usize,side:usize){audit[m*AUDIT_FIELDS+CAST_OFFSET+side*CAST_FIELDS+9]+=1.0f32;}

#[cube]
pub fn hero_death(audit:&mut [f32],m:usize,side:usize,time:f32){
    let at=m*AUDIT_FIELDS+CAST_OFFSET+side*CAST_FIELDS+10;
    if audit[at]==0.0f32 {audit[at]=time+1.0f32;}
}

#[cube(launch)]
fn commands(state:&[f32],meta:&[f32],audit:&[f32],actions:&[f32],refs:&[f32],out:&mut [f32],actors:usize,pregame:f32){
    let a=ABSOLUTE_POS;if a<actors && audit[a/2*AUDIT_FIELDS]==0.0f32 {
        let r=a*FIELDS;let base=a/2*UNITS*STRIDE;let p=base+a%2*STRIDE;let cmd=actions[a*6]as usize;
        out[r+ACTIONS_AT+cmd]+=1.0f32;out[r+BUYS_AT+actions[a*6+1]as usize]+=1.0f32;
        if cmd==10 {out[r+ITEMS_AT+actions[a*6+2]as usize]+=1.0f32;}
        let time=(meta[a/2*META]/30.0f32+pregame).max(0.0f32);let id=refs[a*crate::observe::TARGETS+actions[a*6+3]as usize];
        if cmd==3 && id>=0.0f32 && id<UNITS as f32 {let q=base+id as usize*STRIDE;
            let k=if state[q+KIND]==0.0f32 {12usize}else if state[q+KIND]==1.0f32 {35usize}else if state[q+TEAM]==state[p+TEAM] {11usize}else{10usize};out[r+k]+=1.0f32;
        }
        if state[p+WINDUP]>time && cmd!=0 && (cmd!=3 || id!=state[p+WINDUP_TARGET]) {out[r+9]+=1.0f32;}
    }
}

#[cube(launch)]
fn sample(state:&[f32],meta:&[f32],audit:&[f32],items:&[f32],out:&mut [f32],actors:usize,pregame:f32){
    let a=ABSOLUTE_POS;if a<actors {let r=a*FIELDS;let ar=a/2*AUDIT_FIELDS;let p=(a/2*UNITS+a%2)*STRIDE;
        if out[r+1]==0.0f32 {
            let time=meta[a/2*META]/30.0f32;let elapsed=time+pregame;let dt=if out[r]>0.0f32{(elapsed-out[r+4]).max(0.0f32)}else{0.0f32};
            let alive=state[p+ACTIVE]>0.0f32 && state[p+UNIT_HEALTH]>0.0f32;let x=state[p+UNIT_X];let y=state[p+UNIT_Y];
            let dx=x-out[r+2];let dy=y-out[r+3];let moving=state[p+ORDER]==1.0f32 || state[p+ORDER]==2.0f32;
            if out[r]>0.0f32 {out[r+20]+=(dx*dx+dy*dy).sqrt();}
            if alive {out[r+21]+=dt;}else{out[r+22]+=dt;}
            if alive && x*x+y*y<4000000.0f32 && (x-y).abs()<1200.0f32 && out[r+14]==0.0f32 {out[r+14]=elapsed+1.0f32;}
            if moving && out[r+13]==0.0f32 {out[r+13]=elapsed+1.0f32;}
            let gx=state[p+DEST_X]-x;let gy=state[p+DEST_Y]-y;
            let stalled=alive && moving && gx*gx+gy*gy>1024.0f32 && dx*dx+dy*dy<1.0f32 && state[p+CAST_END]<=elapsed && state[p+TP_END]<=elapsed && state[p+FEAR_END]<=elapsed;
            if stalled {out[r+16]+=dt;}else{out[r+16]=0.0f32;}out[r+15]=out[r+15].max(out[r+16]);
            if alive && !moving && dx*dx+dy*dy<1.0f32 && state[p+WINDUP]<=elapsed && state[p+CAST_END]<=elapsed {out[r+18]+=dt;}else{out[r+18]=0.0f32;}out[r+19]=out[r+19].max(out[r+18]);
            if moving && alive {out[r+17]+=dt;}
            let hp=state[p+UNIT_HEALTH]/state[p+UNIT_HEALTH_MAX].max(1.0f32);
            if out[r]==0.0f32 {out[r+23]=hp;}else{out[r+23]=out[r+23].min(hp);}
            if state[p+ATTACK_READY]>out[r+5]+0.2f32 && out[r]>0.0f32 {out[r+8]+=1.0f32;}
            out[r+5]=state[p+ATTACK_READY];out[r+6]=state[p+DEATHS];out[r+7]=time;
            out[r+24]=state[p+UNIT_LAST_HITS];out[r+25]=state[p+UNIT_DENIES];out[r+26]=state[p+UNIT_LEVEL];out[r+27]=state[p+UNIT_HEALTH];out[r+28]=state[p+UNIT_MANA];out[r+29]=state[p+UNIT_GOLD];out[r+30]=state[p+UNIT_SOULS];
            let mut worth=state[p+UNIT_GOLD];for i in 0..8 {worth+=(state[p+ITEM_START+i]+state[p+STASH_START+i]+state[p+COURIER_START+i]+state[p+BACKPACK_START+i])*items[i*ITEM_RULE_FIELDS];}out[r+31]=worth;
            out[r+32]=state[p+XP];out[r+33]=state[p+UNIT_HEALTH_MAX];out[r+34]=state[p+UNIT_MANA_MAX];
            let slot=if time<0.0f32 {0usize}else{(time/60.0f32).floor().min(10.0f32)as usize+1};let b=r+MINUTES_AT+slot*MINUTE_FIELDS;
            out[b]=elapsed+1.0f32;out[b+1]=state[p+UNIT_LAST_HITS];out[b+2]=state[p+UNIT_DENIES];out[b+3]=state[p+DEATHS];out[b+4]=state[p+UNIT_LEVEL];out[b+5]=state[p+UNIT_HEALTH];out[b+6]=state[p+UNIT_MANA];out[b+7]=state[p+UNIT_GOLD];out[b+8]=state[p+UNIT_SOULS];out[b+9]=worth;out[b+10]=state[p+XP];
            let ca=ar+CAST_OFFSET+a%2*CAST_FIELDS;out[b+11]=audit[ca]+audit[ca+1]+audit[ca+2];
            out[r+2]=x;out[r+3]=y;out[r+4]=elapsed;out[r]+=1.0f32;
            if audit[ar]>0.0f32 {out[r+1]=1.0f32;}
        }
    }
}

pub struct Metrics {pub buffer:CubeTensor}
impl Metrics {
    pub fn new(w:&World)->Self{Self{buffer:primitive(Tensor::<1>::zeros([w.matches*2*FIELDS],&w.device))}}
    pub fn commands(&self,w:&World,t:&crate::scripted_teacher::Teacher,actions:Tensor<2,Int>,refs:&CubeTensor){
        let actions=primitive(actions.float());commands::launch(&w.state.client,CubeCount::Static((w.matches*2).div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&w.state),argument(&w.meta),argument(&t.audit),argument(&actions),argument(refs),argument(&self.buffer),w.matches*2,w.engine.pre_game_seconds);
    }
    pub fn sample(&self,w:&World,t:&crate::scripted_teacher::Teacher){sample::launch(&w.state.client,CubeCount::Static((w.matches*2).div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&w.state),argument(&w.meta),argument(&t.audit),argument(&w.items),argument(&self.buffer),w.matches*2,w.engine.pre_game_seconds);}
    pub fn games(&self,audit:&[f32],pregame:f32,offset:usize,learners:Option<&str>)->Vec<serde_json::Value>{
        let data=tensor::<1>(self.buffer.clone()).into_data().try_to_vec::<f32>().unwrap();
        audit.chunks_exact(AUDIT_FIELDS).enumerate().map(|(m,v)|{
            let sides:Vec<_>=(0..2).map(|side|{let a=&data[(m*2+side)*FIELDS..(m*2+side+1)*FIELDS];let ca=&v[CAST_OFFSET+side*CAST_FIELDS..CAST_OFFSET+(side+1)*CAST_FIELDS];
                let stamp=|x:f32|if x>0.{Some(x-1.-pregame)}else{None};
                let minutes:Vec<_>=(0..12).filter_map(|i|{let b=&a[MINUTES_AT+i*MINUTE_FIELDS..MINUTES_AT+(i+1)*MINUTE_FIELDS];(b[0]>0.).then(||serde_json::json!({"game_seconds":stamp(b[0]),"lh":b[1],"deny":b[2],"deaths":b[3],"level":b[4],"health":b[5],"mana":b[6],"gold":b[7],"souls":b[8],"net_worth":b[9],"xp_in_level":b[10],"razes_completed":b[11]}))}).collect();
                serde_json::json!({"side":if side==0{"radiant"}else{"dire"},"controller":match learners{Some("mirror")=>"GGD",Some(_)=>if side==m%2{"GGD"}else{"teacher"},None=>"teacher"},"lh":a[24],"deny":a[25],"deaths":a[6],"first_death_game_seconds":stamp(ca[10]),"first_move_game_seconds":stamp(a[13]),"mid_arrival_game_seconds":stamp(a[14]),"mid_arrival_elapsed_seconds":if a[14]>0.{Some(a[14]-1.)}else{None},"longest_movement_stall_seconds":a[15],"longest_idle_seconds":a[19],"distance_travelled":a[20],"alive_seconds":a[21],"dead_seconds":a[22],"minimum_health_fraction":a[23],"level":a[26],"health":a[27],"mana":a[28],"gold":a[29],"souls":a[30],"net_worth":a[31],"xp_in_level":a[32],"attack_cycles_completed":a[8],"commands_during_windup":a[9],"attack_enemy_creep_commands":a[10],"deny_commands":a[11],"attack_hero_commands":a[12],"attack_structure_commands":a[35],"actions":&a[ACTIONS_AT..ACTIONS_AT+11],"buys":&a[BUYS_AT..BUYS_AT+7],"items":&a[ITEMS_AT..ITEMS_AT+8],"spell_completions":&ca[..4],"raze_hero_hits":ca[4],"raze_creep_hits":ca[5],"razes_hitting_hero":ca[6],"razes_hitting_any_target":ca[7],"stacked_raze_hero_hits":ca[8],"attack_cancellations":ca[9],"minute_snapshots":minutes})
            }).collect();
            serde_json::json!({"game":offset+m,"game_seconds":v[1],"native_terminal":v[2]>0.,"lane_deaths":v[3],"credited":v[4],"accessible_deaths":v[5],"accessible_credited":v[6],"simulation_error":v[39],"sides":sides})
        }).collect()
    }
}

pub fn summary(games:&[serde_json::Value])->serde_json::Value{
    let sum=|key:&str|games.iter().map(|g|g[key].as_f64().unwrap_or(0.)).sum::<f64>();
    let mut sides=vec![];
    for controller in ["teacher","GGD"] {for side in ["radiant","dire"] {
        let rows:Vec<_>=games.iter().flat_map(|g|g["sides"].as_array().unwrap()).filter(|s|s["controller"]==controller&&s["side"]==side).collect();if rows.is_empty(){continue;}
        let mut row=serde_json::json!({"controller":controller,"side":side,"games":rows.len()});
        for key in ["lh","deny","deaths","attack_cycles_completed","commands_during_windup","raze_hero_hits","raze_creep_hits","razes_hitting_hero","razes_hitting_any_target","stacked_raze_hero_hits","attack_cancellations","alive_seconds","dead_seconds"] {row[key]=serde_json::json!(rows.iter().map(|s|s[key].as_f64().unwrap_or(0.)).sum::<f64>());}
        for key in ["mid_arrival_game_seconds","mid_arrival_elapsed_seconds","first_move_game_seconds","first_death_game_seconds","longest_movement_stall_seconds","longest_idle_seconds","level","net_worth","health"] {
            let mut values:Vec<_>=rows.iter().filter_map(|s|s[key].as_f64()).collect();values.sort_by(f64::total_cmp);let n=values.len();row[key]=serde_json::json!({"count":n,"denominator":rows.len(),"mean":if n>0{Some(values.iter().sum::<f64>()/n as f64)}else{None},"median":values.get(n/2),"minimum":values.first(),"maximum":values.last()});
        }
        for (key,len) in [("actions",11),("buys",7),("items",8),("spell_completions",4)] {row[key]=serde_json::json!((0..len).map(|i|rows.iter().map(|s|s[key][i].as_f64().unwrap_or(0.)).sum::<f64>()).collect::<Vec<_>>());}
        sides.push(row);
    }}
    serde_json::json!({"games":games.len(),"native_completed":games.iter().filter(|g|g["native_terminal"]==true).count(),"all_lane_creeps":{"numerator":sum("credited"),"denominator":sum("lane_deaths")},"within_attack_distance":{"numerator":sum("accessible_credited"),"denominator":sum("accessible_deaths")},"sides":sides,"action_names":["continue","move","move_direct","attack","stop","hold","raze_near","raze_mid","raze_far","requiem","item"],"spell_names":["raze_near","raze_mid","raze_far","requiem"],"time_origin":"Dota game seconds; negative values are pregame","position_sampling_seconds":4.0/30.0,"first_death_precision_seconds":1.0/30.0,"minute_snapshots":"latest observed state in each minute bin, plus pregame; early terminal games retain only reached bins"})
}

pub fn summary_with_clock(games:&[serde_json::Value],decision_ticks:usize)->serde_json::Value{
    let mut result=summary(games);
    result["position_sampling_seconds"]=serde_json::json!(decision_ticks as f64/30.0);
    result
}
