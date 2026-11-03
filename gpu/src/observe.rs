//! Visibility filtering and the production observation ABI, built entirely on device.
use cubecl::prelude::*;
use burn::tensor::{Tensor,Int,Device};
use burn_cubecl::tensor::CubeTensor;
use crate::{world::*,contract::*,model::Observation,runtime::{primitive,tensor,argument}};

pub const CREEPS:usize=256;pub const TREES:usize=256;pub const PROJECTILES:usize=512;
pub const TARGETS:usize=2+CREEPS+4+TREES+1;
pub const HERO_OFFSET:usize=0;
pub const CREEP_OFFSET:usize=2*2*UNIT_COUNT;
pub const TOWER_OFFSET:usize=CREEP_OFFSET+CREEPS*2*UNIT_COUNT;
pub const TREE_OFFSET:usize=TOWER_OFFSET+4*2*UNIT_COUNT;
pub const PROJECTILE_OFFSET:usize=TREE_OFFSET+TREES*2*TREE_COUNT;
pub const SKILL_OFFSET:usize=PROJECTILE_OFFSET+PROJECTILES*2*PROJECTILE_COUNT;
pub const ITEM_OFFSET:usize=SKILL_OFFSET+4*2*SKILL_COUNT;
pub const ENEMY_ITEM_OFFSET:usize=ITEM_OFFSET+8*2*ITEM_COUNT;
pub const TERRAIN_OFFSET:usize=ENEMY_ITEM_OFFSET+8*2*ITEM_COUNT;
pub const GLOBAL_OFFSET:usize=TERRAIN_OFFSET+256;
pub const PREVIOUS_OFFSET:usize=GLOBAL_OFFSET+2*GLOBAL_COUNT;
pub const HERO_VALID:usize=PREVIOUS_OFFSET+PREVIOUS;
pub const CREEP_VALID:usize=HERO_VALID+2;
pub const CREEP_ALLY:usize=CREEP_VALID+CREEPS;
pub const TREE_VALID:usize=CREEP_ALLY+CREEPS;
pub const PROJECTILE_VALID:usize=TREE_VALID+TREES;
pub const ACTION_MASK:usize=PROJECTILE_VALID+PROJECTILES;
pub const BUY_MASK:usize=ACTION_MASK+11;
pub const ITEM_MASK:usize=BUY_MASK+7;
pub const ATTACK_MASK:usize=ITEM_MASK+8;
pub const TREE_MASK:usize=ATTACK_MASK+TARGETS;
pub const TP_MASK:usize=TREE_MASK+TARGETS;
pub const SELF_MASK:usize=TP_MASK+TARGETS;
pub const WIDTH:usize=SELF_MASK+TARGETS;
pub const COMBAT_UNIT_OFFSET:usize=WIDTH;
pub const COMBAT_PROJECTILE_OFFSET:usize=COMBAT_UNIT_OFFSET+(2+CREEPS+4)*2*COMBAT_UNIT_COUNT;
pub const GGD_WIDTH:usize=COMBAT_PROJECTILE_OFFSET+PROJECTILES*2*COMBAT_PROJECTILE_COUNT;

#[cube]
fn scales(field:usize)->f32 {
    // A device table is supplied by the host once; this helper is deliberately
    // unused for scaling to keep the Python-generated constants authoritative.
    field as f32
}

#[cube]
pub fn sight(state:&mut [f32],geometry:&[f32],fow_blockers:&[f32],static_vision:&[u32],tree_map:&[f32],trees:&[f32],tree_alive:&mut [f32],extra_vision:&[f32],vision_count:usize,m:usize,tree_count:usize,base:usize,side:f32,x:f32,y:f32,time:f32)->bool {
    // Six measured phase/team bits and six validity bits. Only supported map
    // cells replace building rays; moving heroes/creeps still add their sight.
    let phase=if time<0.0f32{0usize}else if daytime(time){1usize}else{2usize};
    let bit=side as usize*3+phase;let entry=static_vision[cell(x,y)/2];
    let supported=x>=-8192.0f32 && x<8192.0f32 && y>=-8192.0f32 && y<8192.0f32 && (entry&(1u32<<(bit+6)as u32))!=0u32;
    let static_seen=supported && (entry&(1u32<<bit as u32))!=0u32;
    let x=((x+8192.0f32)/64.0f32).floor()*64.0f32-8192.0f32+32.0f32;
    let y=((y+8192.0f32)/64.0f32).floor()*64.0f32-8192.0f32+32.0f32;
    let mut seen=static_seen;
    for u in 0..UNITS+vision_count {
        let mut valid=false;let mut origin_x=0.0f32.runtime();let mut origin_y=0.0f32.runtime();let mut radius=0.0f32.runtime();let mut unobstructed=false;
        if u<UNITS {
            if (u<FOG_START || u>=BACKGROUND_START) && (!supported || u<2 || (u>=6 && u<FOG_START)) {
                let p=base+u*STRIDE;valid=state[p+ACTIVE]>0.0f32 && state[p+UNIT_HEALTH]>0.0f32 && state[p+TEAM]==side;
                origin_x=state[p+UNIT_X];origin_y=state[p+UNIT_Y];radius=vision_radius(u,state[p+KIND],time);
            }
        }else{
            let k=(u-UNITS)*6;valid=!supported && extra_vision[k+2]==side;unobstructed=extra_vision[k+5]>0.0f32;
            origin_x=extra_vision[k];origin_y=extra_vision[k+1];radius=if daytime(time){extra_vision[k+3]}else{extra_vision[k+4]};
        }
        if !seen && valid {
            let dx=x-origin_x;let dy=y-origin_y;
            if dx*dx+dy*dy<=radius*radius {
              if unobstructed {seen=true;}else{
                let height=geometry[cell(origin_x,origin_y)];let mut blocked=false;
                for i in 1..64 {
                    let f=i as f32/64.0f32;let sx=origin_x+dx*f;let sy=origin_y+dy*f;
                    if higher_ground(geometry[cell(sx,sy)],height) || fow_blockers[cell(sx,sy)/2]>0.0f32 {blocked=true;}
                    let id=tree_map[cell(sx,sy)/2] as usize;
                    if id>0 {
                        let t=id-1;let tx=trees[t*3];let ty=trees[t*3+1];let ex=sx-tx;let ey=sy-ty;
                        let along=(tx-origin_x)*dx+(ty-origin_y)*dy;
                        if tree_alive[m*tree_count+t]>0.0f32 && ex*ex+ey*ey<64.0f32*64.0f32 && along>0.0f32 && along<dx*dx+dy*dy {blocked=true;}
                    }
                }
                if !blocked {seen=true;}
              }
            }
        }
    }
    seen
}

#[cube]
fn encode_unit(state:&mut [f32],output:&mut [f32],scale:&[f32],items:&[f32],hero_rules:&[f32],fountains:&[f32],p:usize,origin:usize,dest:usize,own:bool,hero:bool,time:f32,extras:bool){
    let attack_target=crate::physics::observed_attack_target(state,(p/STRIDE/UNITS)*UNITS*STRIDE,p,time);
    for f in 0..UNIT_COUNT {
        let private=f==UNIT_GOLD || f==UNIT_XP_TO_NEXT || f==UNIT_LAST_HITS || f==UNIT_DENIES || f==UNIT_SOULS ||
            f==UNIT_FEAR_REMAINING || f==UNIT_RAZE_REMAINING || f==UNIT_ACTIVE_SPELL;
        let mut known=true;
        if private && (!own || !hero) {known=false;}
        if state[p+KIND]<2.0f32 && f>=UNIT_TEAM_RELATIVE {known=false;}
        if f==UNIT_MANA_FRACTION && state[p+UNIT_MANA_MAX]<=0.0f32 {known=false;}
        if !extras && (f==UNIT_ATTACK_INTERVAL || f==UNIT_IS_ATTACKING || f==UNIT_IS_USING_ABILITY || f==UNIT_IS_CASTING_ABILITY || f==UNIT_IS_CHANNELING) {known=false;}
        let mut value=state[p+f];
        if f==UNIT_HEALTH_REGEN && !hero && state[p+KIND]>=2.0f32 && state[p+FLAG_AURA_END]>time {value+=3.0f32;}
        if f==UNIT_ARMOR {value=effective_armor(state,hero_rules,(p/STRIDE/UNITS)*UNITS*STRIDE,p,time);}
        if hero && (f==UNIT_HEALTH_REGEN || f==UNIT_MANA_REGEN) {
            let side=state[p+TEAM] as usize;let dx=state[p+UNIT_X]-fountains[side*3];let dy=state[p+UNIT_Y]-fountains[side*3+1];
            if dx*dx+dy*dy<1100.0f32*1100.0f32 {
                value+=if f==UNIT_HEALTH_REGEN {state[p+UNIT_HEALTH_MAX]*0.05f32}else{state[p+UNIT_MANA_MAX]*0.06f32};
            }
            if f==UNIT_HEALTH_REGEN {
                value+=tower_regen(state,p,time);
                if state[p+FLASK_END]>time {value+=items[5*ITEM_RULE_FIELDS+4];}
                if state[p+TANGO_END]>time {value+=items[6*ITEM_RULE_FIELDS+4];}
            }else if state[p+CLARITY_END]>time {value+=items[7*ITEM_RULE_FIELDS+6];}
        }
        if f==UNIT_DX {value=state[p+UNIT_X]-state[origin+UNIT_X];}
        if f==UNIT_DY {value=state[p+UNIT_Y]-state[origin+UNIT_Y];}
        if f==UNIT_FACING_COS {value=state[p+FACING].cos();}if f==UNIT_FACING_SIN {value=state[p+FACING].sin();}
        if f==UNIT_HEALTH_FRACTION {value=state[p+UNIT_HEALTH]/f32::max(1.0f32,state[p+UNIT_HEALTH_MAX]);}
        if f==UNIT_MANA_FRACTION {value=state[p+UNIT_MANA]/f32::max(1.0f32,state[p+UNIT_MANA_MAX]);}
        if f==UNIT_IS_ATTACKING {
            // Native own-hero input uses action_type==ATTACK (including
            // chasing); other units use their attack animation activity.
            let attacking=if hero && own {state[p+WINDUP]>0.0f32 || state[p+ORDER]==3.0f32}else{attack_target>=0.0f32 && state[p+CAST_END]<=time && state[p+TP_END]<=time};
            value=if attacking {1.0f32}else{0.0f32};
        }
        if f==UNIT_IS_CASTING_ABILITY || f==UNIT_IS_USING_ABILITY {value=if state[p+CAST_END]>time {1.0f32} else {0.0f32};}
        if f==UNIT_IS_CHANNELING {value=if state[p+TP_END]>time {1.0f32} else {0.0f32};}
        if f==UNIT_FEAR {value=if state[p+FEAR_END]>time {1.0f32} else {0.0f32};}
        if f==UNIT_FEAR_REMAINING {value=f32::max(0.0f32,state[p+FEAR_END]-time);}
        if f==UNIT_MOVE_SPEED && state[p+FEAR_END]>time {value*=1.0f32+state[p+SLOW_PERCENT];}
        if f==UNIT_RAZE_REMAINING {value=f32::max(0.0f32,state[p+STACK_END]-time);}
        if f==UNIT_RAZE_STACKS && state[p+STACK_END]<=time {value=0.0f32;}
        if f==UNIT_FLASK {value=if state[p+FLASK_END]>time {1.0f32} else {0.0f32};}
        if f==UNIT_TANGO {value=if state[p+TANGO_END]>time {1.0f32} else {0.0f32};}
        if f==UNIT_CLARITY {value=if state[p+CLARITY_END]>time {1.0f32} else {0.0f32};}
        if f==UNIT_ACTIVE_SPELL {value=state[p+CAST];}
        if f==UNIT_ATTACK_TARGET_SELF {value=if attack_target==((origin/STRIDE)%UNITS) as f32 {1.0f32} else {0.0f32};}
        if f==UNIT_ATTACK_TARGET_ENEMY {let enemy=((origin/STRIDE)%UNITS+1)%2;value=if attack_target==enemy as f32 {1.0f32}else{0.0f32};}
        if f==UNIT_TEAM_RELATIVE {value=if state[p+TEAM]==state[origin+TEAM] {1.0f32} else {-1.0f32};}
        // Native adapter classifies by name: flagbearer is absent from all
        // three type flags, even though its physical behavior is melee.
        if f==UNIT_CREEP_MELEE {value=if state[p+KIND]==2.0f32 && state[p+FLAGBEARER]==0.0f32 {1.0f32} else {0.0f32};}
        if f==UNIT_CREEP_RANGED {value=if state[p+KIND]==3.0f32 {1.0f32} else {0.0f32};}
        if f==UNIT_CREEP_SIEGE {value=if state[p+KIND]==4.0f32 {1.0f32} else {0.0f32};}
        if known {output[dest+f]=value/scale[f];output[dest+UNIT_COUNT+f]=1.0f32;}
    }
}

#[cube(launch)]
fn observe(state:&mut [f32],shots:&[f32],meta:&mut [f32],trees:&[f32],tree_alive:&mut [f32],geometry:&[f32],fow_blockers:&[f32],static_vision:&[u32],tree_map:&[f32],
    spells:&[f32],items:&[f32],hero_rules:&[f32],fountains:&[f32],extra_vision:&[f32],scale:&[f32],previous:&[f32],output:&mut [f32],refs:&mut [f32],actors:usize,tree_count:usize,vision_count:usize,pregame:f32){
    let actor=ABSOLUTE_POS;
    if actor<actors {
        let m=actor/2;let side=actor%2;let base=m*UNITS*STRIDE;let p=base+side*STRIDE;
        let row=actor*WIDTH;let game_time=meta[m*META]/30.0f32;let time=f32::max(0.0f32,(meta[m*META]+pregame*30.0f32)/30.0f32);
        for f in 0..WIDTH {output[row+f]=0.0f32;}
        for f in 0..TARGETS {refs[actor*TARGETS+f]=-1.0f32;}
        refs[actor*TARGETS]=side as f32;refs[actor*TARGETS+1]=(1-side) as f32;
        encode_unit(state,output,scale,items,hero_rules,fountains,p,p,row+HERO_OFFSET,true,true,time,true);output[row+HERO_VALID]=1.0f32;
        let enemy=base+(1-side)*STRIDE;
        let revealed=state[enemy+REVEAL_END]>time;let enemy_x=state[enemy+UNIT_X];let enemy_y=state[enemy+UNIT_Y];
        let enemy_seen=state[enemy+ACTIVE]>0.0f32 && state[enemy+UNIT_HEALTH]>0.0f32 && (revealed || sight(state,geometry,fow_blockers,static_vision,tree_map,trees,tree_alive,extra_vision,vision_count,m,tree_count,base,side as f32,enemy_x,enemy_y,game_time));
        if enemy_seen && state[enemy+ACTIVE]>0.0f32 {
            encode_unit(state,output,scale,items,hero_rules,fountains,enemy,p,row+HERO_OFFSET+2*UNIT_COUNT,false,true,time,true);
            output[row+HERO_VALID+1]=1.0f32;output[row+ATTACK_MASK+1]=1.0f32;
        }
        let mut creeps=0usize;
        for u in 6..CREEP_END {
            let q=base+u*STRIDE;
            let own=state[q+TEAM]==side as f32;let active=state[q+ACTIVE]>0.0f32;let revealed=state[q+REVEAL_END]>time;let x=state[q+UNIT_X];let y=state[q+UNIT_Y];
            if active && (own || revealed || sight(state,geometry,fow_blockers,static_vision,tree_map,trees,tree_alive,extra_vision,vision_count,m,tree_count,base,side as f32,x,y,game_time)) {
                if creeps<CREEPS {
                    encode_unit(state,output,scale,items,hero_rules,fountains,q,p,row+CREEP_OFFSET+creeps*2*UNIT_COUNT,own,false,time,true);
                    output[row+CREEP_VALID+creeps]=1.0f32;
                    output[row+CREEP_ALLY+creeps]=if state[q+TEAM]==side as f32 {1.0f32} else {0.0f32};
                    refs[actor*TARGETS+2+creeps]=u as f32;
                    if state[q+TEAM]!=side as f32 || state[q+UNIT_HEALTH]<=state[q+UNIT_HEALTH_MAX]*0.5f32 {output[row+ATTACK_MASK+2+creeps]=1.0f32;}
                    let x=((state[q+UNIT_X]-state[p+UNIT_X])/256.0f32+4.0f32).floor() as i32;
                    let y=((state[q+UNIT_Y]-state[p+UNIT_Y])/256.0f32+4.0f32).floor() as i32;
                    if x>=0 && x<8 && y>=0 && y<8 {
                        let delta=if state[q+TEAM]==side as f32 {0.25f32} else {-0.25f32};
                        output[row+TERRAIN_OFFSET+192+y as usize*8+x as usize]+=delta;
                    }
                } else {meta[m*META+7]=3.0f32;}
                creeps+=1;
            }
        }
        for tower in 0..4 {
            let q=base+(tower+2)*STRIDE;let at=row+TOWER_OFFSET+tower*2*UNIT_COUNT;
            let index=2+CREEPS+tower;refs[actor*TARGETS+index]=(tower+2) as f32;
            let own=state[q+TEAM]==side as f32;
            let x=state[q+UNIT_X];let y=state[q+UNIT_Y];
            let seen=own || sight(state,geometry,fow_blockers,static_vision,tree_map,trees,tree_alive,extra_vision,vision_count,m,tree_count,base,side as f32,x,y,game_time);
            let memory=base+(FOG_START+side*4+tower)*STRIDE;
            if seen {for f in 0..STRIDE {if f!=ACTIVE {state[memory+f]=state[q+f];}}}
            // Building destruction is a public match event.
            if state[q+ACTIVE]==0.0f32 {state[memory+UNIT_HEALTH]=0.0f32;}
            encode_unit(state,output,scale,items,hero_rules,fountains,memory,p,at,own,false,time,seen);
            if own && state[q+ACTIVE]>0.0f32 {output[row+TP_MASK+index]=1.0f32;}
            if !own && state[memory+UNIT_HEALTH]>0.0f32 && state[memory+INVULNERABLE]==0.0f32 {output[row+ATTACK_MASK+index]=1.0f32;}
        }
        let mut tree_count_seen=0usize;
        for t in 0..tree_count {
            if tree_alive[m*tree_count+t]>0.0f32 {
                let dx=trees[t*3]-state[p+UNIT_X];let dy=trees[t*3+1]-state[p+UNIT_Y];let distance=(dx*dx+dy*dy).sqrt();
                if distance<1600.0f32 && sight(state,geometry,fow_blockers,static_vision,tree_map,trees,tree_alive,extra_vision,vision_count,m,tree_count,base,side as f32,trees[t*3],trees[t*3+1],game_time) {
                    if tree_count_seen<TREES {
                        let at=row+TREE_OFFSET+tree_count_seen*8;
                        output[at]=dx/2048.0f32;output[at+1]=dy/2048.0f32;output[at+2]=trees[t*3+2]/512.0f32;output[at+3]=distance/2048.0f32;
                        for f in 0..4 {output[at+4+f]=1.0f32;}
                        output[row+TREE_VALID+tree_count_seen]=1.0f32;
                        let index=2+CREEPS+4+tree_count_seen;output[row+TREE_MASK+index]=1.0f32;
                        refs[actor*TARGETS+index]=(UNITS+t) as f32;
                    } else {meta[m*META+7]=4.0f32;}
                    tree_count_seen+=1;
                }
            }
        }
        let mut count=0usize;
        for s in 0..SHOTS {
            let k=(m*SHOTS+s)*SHOT_FIELDS;
            // The existing Dota ABI exposes incoming tracking projectiles only.
            // Requiem's linear effects stay out of this observation channel.
            if shots[k]==1.0f32 {
                let target=shots[k+3] as usize;let q=base+target*STRIDE;
                let own=state[q+TEAM]==side as f32;let x=state[q+UNIT_X];let y=state[q+UNIT_Y];
                if own || sight(state,geometry,fow_blockers,static_vision,tree_map,trees,tree_alive,extra_vision,vision_count,m,tree_count,base,side as f32,x,y,game_time) {
                    if count<PROJECTILES {
                        let at=row+PROJECTILE_OFFSET+count*12;
                        output[at]=(shots[k+1]-state[p+UNIT_X])/2048.0f32;output[at+1]=(shots[k+2]-state[p+UNIT_Y])/2048.0f32;
                        output[at+2]=shots[k+5]/1500.0f32;output[at+3]=1.0f32;
                        output[at+4]=if shots[k+8]!=side as f32 {1.0f32} else {0.0f32};
                        output[at+5]=if target==side {1.0f32} else {0.0f32};
                        for f in 0..6 {output[at+6+f]=1.0f32;}
                        output[row+PROJECTILE_VALID+count]=1.0f32;
                    } else {meta[m*META+7]=5.0f32;}
                    count+=1;
                }
            }
        }
        output[row+ACTION_MASK]=1.0f32;output[row+ACTION_MASK+1]=1.0f32;output[row+ACTION_MASK+2]=1.0f32;
        // The installed Dota Bot API has no Action_HoldPosition. Native
        // model.can_hold is false and the executor rejects this action.
        output[row+ACTION_MASK+4]=1.0f32;output[row+BUY_MASK]=1.0f32;
        output[row+SELF_MASK]=1.0f32;output[row+TP_MASK+TARGETS-1]=1.0f32;refs[actor*TARGETS+TARGETS-1]=(UNITS+tree_count+side) as f32;
        let mut attack=false;
        for t in 0..TARGETS {if output[row+ATTACK_MASK+t]>0.0f32 {attack=true;}}
        if attack {output[row+ACTION_MASK+3]=1.0f32;}
        for spell in 0..4 {
            let level=crate::physics::spell_level(state[p+UNIT_LEVEL],spell);let at=row+SKILL_OFFSET+spell*12;
            output[at]=level as f32/4.0f32;
            if level>0 {
                let k=(spell*4+level-1)*8;
                output[at+1]=f32::max(0.0f32,state[p+COOLDOWNS+spell]-time)/120.0f32;output[at+2]=spells[k+1]/200.0f32;output[at+3]=spells[k+5]/2.0f32;
                let valid=state[p+UNIT_MANA]>=spells[k+1] && state[p+COOLDOWNS+spell]<=time;
                if valid {output[at+4]=1.0f32;output[row+ACTION_MASK+6+spell]=1.0f32;}
            }
            for f in 0..6 {output[at+6+f]=1.0f32;}
        }
        for item in 0..8 {
            let at=row+ITEM_OFFSET+item*16;
            output[at]=state[p+ITEM_START+item]/10.0f32;
            output[at+1]=state[p+BACKPACK_START+item]/10.0f32;
            output[at+2]=state[p+STASH_START+item]/10.0f32;output[at+3]=state[p+COURIER_START+item]/10.0f32;
            output[at+4]=f32::max(0.0f32,state[p+ITEM_READY+item]-time)/120.0f32;
            output[at+6]=items[item*ITEM_RULE_FIELDS+16]/200.0f32;
            if state[p+ITEM_START+item]>0.0f32 {output[at+7]=state[p+SLOT_START+item]/20.0f32;}
            else {output[at+7]=-1.0f32/20.0f32;}
            let valid=item>=3 && state[p+ITEM_START+item]>0.0f32 && state[p+ITEM_READY+item]<=time && state[p+UNIT_MANA]>=items[item*ITEM_RULE_FIELDS+16] && (item!=6 || tree_count_seen>0);
            if valid {output[at+5]=1.0f32;output[row+ITEM_MASK+item]=1.0f32;output[row+ACTION_MASK+10]=1.0f32;}
            for f in 0..8 {output[at+8+f]=1.0f32;}
            if state[p+ITEM_START+item]==0.0f32 {output[at+4]=0.0f32;output[at+6]=0.0f32;output[at+12]=0.0f32;output[at+14]=0.0f32;}
            if enemy_seen {
                let e=row+ENEMY_ITEM_OFFSET+item*16;
                if item!=3 {output[e]=state[enemy+ITEM_START+item]/10.0f32;output[e+1]=state[enemy+BACKPACK_START+item]/10.0f32;output[e+8]=1.0f32;output[e+9]=1.0f32;}
            }
        }
        let boots_owned=state[p+ITEM_START+1]+state[p+BACKPACK_START+1]+state[p+STASH_START+1]+state[p+COURIER_START+1];
        let bracer_owned=state[p+ITEM_START+2]+state[p+BACKPACK_START+2]+state[p+STASH_START+2]+state[p+COURIER_START+2];
        let permanent=if bracer_owned>0.0f32 {2.0f32} else if boots_owned>0.0f32 {1.0f32} else {0.0f32};
        let pending=if permanent<2.0f32 {state[p+STASH_START+permanent as usize+1]+state[p+COURIER_START+permanent as usize+1]} else {0.0f32};
        if permanent<2.0f32 && pending==0.0f32 && state[p+UNIT_GOLD]>=items[(permanent as usize+1)*ITEM_RULE_FIELDS] {output[row+BUY_MASK+1]=1.0f32;}
        for buy in 2..7 {
            let item=buy+1;let stock=items[item*ITEM_RULE_FIELDS+7];
            if state[p+UNIT_GOLD]>=items[item*ITEM_RULE_FIELDS] && (stock<0.0f32 || state[p+STOCK_START+item]<stock) {output[row+BUY_MASK+buy]=1.0f32;}
        }
        if state[p+FEAR_END]>time {for f in 1..11 {output[row+ACTION_MASK+f]=0.0f32;}}
        if state[p+UNIT_HEALTH]<=0.0f32 {
            // Native collector requests no policy decisions while dead. Keep
            // GPU batch shapes fixed with a deterministic no-op instead.
            for f in 1..11 {output[row+ACTION_MASK+f]=0.0f32;}
            for f in 1..7 {output[row+BUY_MASK+f]=0.0f32;}
            for f in 0..8 {output[row+ITEM_MASK+f]=0.0f32;}
        }
        output[row+GLOBAL_OFFSET]=game_time/600.0f32;output[row+GLOBAL_OFFSET+1]=permanent/2.0f32;
        output[row+GLOBAL_OFFSET+2]=if side==0 {1.0f32} else {-1.0f32};
        for f in 0..3 {output[row+GLOBAL_OFFSET+3+f]=1.0f32;}
        for f in 0..PREVIOUS {output[row+PREVIOUS_OFFSET+f]=previous[actor*PREVIOUS+f];}
        if state[p+RESPAWN_RESET]>0.0f32 && state[p+RESPAWN_RESET]==meta[m*META+6] {
            for f in 0..PREVIOUS {output[row+PREVIOUS_OFFSET+f]=0.0f32;}
            output[row+PREVIOUS_OFFSET]=1.0f32;
            output[row+PREVIOUS_OFFSET+15]=1.0f32;output[row+PREVIOUS_OFFSET+23]=1.0f32;
        }
        for y in 0..8 {for x in 0..8 {
            let wx=state[p+UNIT_X]+(x as f32-3.5f32)*256.0f32;let wy=state[p+UNIT_Y]+(y as f32-3.5f32)*256.0f32;
            let at=row+TERRAIN_OFFSET+y*8+x;output[at]=geometry[cell(wx,wy)]/5.0f32;
            if sight(state,geometry,fow_blockers,static_vision,tree_map,trees,tree_alive,extra_vision,vision_count,m,tree_count,base,side as f32,wx,wy,game_time) {output[at+64]=geometry[cell(wx,wy)+1];output[at+128]=1.0f32;}
        }}
    }
}

pub struct Observed {pub packed:Tensor<2>,pub refs:CubeTensor,pub reset:Tensor<1>}
impl Observed {
    pub fn previous(&self,actions:Tensor<2,Int>,bases:&CubeTensor,state:&CubeTensor)->Tensor<2>{
        let d=self.packed.device();let n=actions.dims()[0];
        let actions=primitive(actions.float());let input=primitive(self.packed.clone().cast(burn::tensor::DType::F32));
        let out=primitive(Tensor::<2>::zeros([n,PREVIOUS],&d));
        previous::launch(&out.client,CubeCount::Static(n.div_ceil(64) as u32,1,1),CubeDim::new_1d(64),
            argument(&actions),argument(&input),argument(bases),argument(state),argument(&out),n);tensor(out)
    }
    pub fn decode(&self)->Observation {
        self.decode_bucket(CREEPS,TREES,PROJECTILES)
    }
    pub fn decode_bucket(&self,pc:usize,pt:usize,pp:usize)->Observation {
        let packed=self.packed.clone().cast(burn::tensor::DType::F32);let n=packed.dims()[0];
        let cut=|offset:usize,len:usize|packed.clone().slice([0..n,offset..offset+len]);
        let mask=|offset,len|cut(offset,len).greater_elem(0.5f32);
        let target_mask=|offset|Tensor::cat(vec![mask(offset,2+pc),
            mask(offset+2+CREEPS,4+pt),mask(offset+TARGETS-1,1)],1);
        let extra=|offset,count,fields|if packed.dims()[1]>=GGD_WIDTH {cut(offset,count*2*fields).reshape([n,count,2*fields])}else{Tensor::zeros([n,count,2*fields],&packed.device())};
        let unit_extra=|slot,count| {
            let combat=extra(COMBAT_UNIT_OFFSET+slot*2*COMBAT_UNIT_COUNT,count,COMBAT_UNIT_COUNT);
            let combat=if packed.dims()[1]>=crate::timing_observe::WIDTH {Tensor::cat(vec![combat,extra(crate::timing_observe::OFFSET+slot*2*crate::timing_observe::FIELDS,count,crate::timing_observe::FIELDS)],2)}else{combat};
            if slot==0 && packed.dims()[1]==crate::enemy_observe::WIDTH {Tensor::cat(vec![combat,extra(crate::enemy_observe::OFFSET,2,crate::enemy_observe::FIELDS)],2)}else{combat}
        };
        Observation {
            combat_profile:packed.dims()[1]>=GGD_WIDTH,
            hero_combat:unit_extra(0,2),
            creep_combat:unit_extra(2,pc),
            tower_combat:unit_extra(2+CREEPS,4),
            projectile_combat:extra(COMBAT_PROJECTILE_OFFSET,pp,COMBAT_PROJECTILE_COUNT),
            heroes:cut(HERO_OFFSET,2*2*UNIT_COUNT).reshape([n,2,2*UNIT_COUNT]),
            creeps:cut(CREEP_OFFSET,pc*2*UNIT_COUNT).reshape([n,pc,2*UNIT_COUNT]),
            towers:cut(TOWER_OFFSET,4*2*UNIT_COUNT).reshape([n,4,2*UNIT_COUNT]),
            trees:cut(TREE_OFFSET,pt*8).reshape([n,pt,8]),projectiles:cut(PROJECTILE_OFFSET,pp*12).reshape([n,pp,12]),
            skills:cut(SKILL_OFFSET,48).reshape([n,4,12]),items:cut(ITEM_OFFSET,128).reshape([n,8,16]),enemy_items:cut(ENEMY_ITEM_OFFSET,128).reshape([n,8,16]),
            terrain:cut(TERRAIN_OFFSET,256).reshape([n,4,8,8]),global_features:cut(GLOBAL_OFFSET,6),previous:cut(PREVIOUS_OFFSET,PREVIOUS),
            hero_valid:mask(HERO_VALID,2),creep_valid:mask(CREEP_VALID,pc),creep_ally:mask(CREEP_ALLY,pc),tree_valid:mask(TREE_VALID,pt),
            projectile_valid:mask(PROJECTILE_VALID,pp),action_mask:mask(ACTION_MASK,11),buy_mask:mask(BUY_MASK,7),item_mask:mask(ITEM_MASK,8),
            attack_mask:target_mask(ATTACK_MASK),tree_mask:target_mask(TREE_MASK),tp_mask:target_mask(TP_MASK),self_mask:target_mask(SELF_MASK),
        }
    }
}

pub fn initial_previous(n:usize,d:&Device)->Tensor<2>{
    Tensor::cat(vec![Tensor::<1,Int>::zeros([n],d).one_hot::<2>(11).float(),Tensor::zeros([n,4],d),
        Tensor::<1,Int>::zeros([n],d).one_hot::<2>(8).float(),Tensor::<1,Int>::zeros([n],d).one_hot::<2>(7).float()],1)
}

#[cube(launch)]
fn previous(actions:&[f32],observations:&[f32],bases:&[f32],state:&[f32],output:&mut [f32],actors:usize){
    let a=ABSOLUTE_POS;
    if a<actors {
        let row=a*PREVIOUS;let p=a*6;let obs=a*WIDTH;
        for f in 0..PREVIOUS {output[row+f]=0.0f32;}
        output[row+actions[p] as usize]=1.0f32;
        output[row+11]=(actions[p+4]-64.0f32)/64.0f32;output[row+12]=(actions[p+5]-64.0f32)/64.0f32;
        output[row+15+actions[p+2] as usize]=1.0f32;output[row+23+actions[p+1] as usize]=1.0f32;
        let t=actions[p+3] as usize;
        if t>0 && t<TARGETS-1 {
            let mut at=obs+HERO_OFFSET+2*UNIT_COUNT*t;
            if t>=2 && t<2+CREEPS {at=obs+CREEP_OFFSET+(t-2)*2*UNIT_COUNT;}
            if t>=2+CREEPS && t<2+CREEPS+4 {at=obs+TOWER_OFFSET+(t-2-CREEPS)*2*UNIT_COUNT;}
            if t>=2+CREEPS+4 {at=obs+TREE_OFFSET+(t-2-CREEPS-4)*8;output[row+13]=observations[at];output[row+14]=observations[at+1];}
            else {output[row+13]=observations[at+UNIT_DX];output[row+14]=observations[at+UNIT_DY];}
        }
        if t==TARGETS-1 {
            // Native action_label uses the original snapshot coordinates.
            // Reconstructing absolute XY from FP16 loses up to four world units.
            let p=(a/2*UNITS+a%2)*STRIDE;
            output[row+13]=(bases[a%2*3]-state[p+UNIT_X])/2048.0f32;
            output[row+14]=(bases[a%2*3+1]-state[p+UNIT_Y])/2048.0f32;
        }
    }
}

impl World {
    pub fn observe(&self,previous:Tensor<2>)->Observed {
        self.observe_with_clock(previous,None)
    }
    /// Serving/evaluation can share an absolute game clock with native Dota.
    /// Training keeps global128-step zero-state BPTT through observe().
    pub fn observe_with_clock(&self,previous:Tensor<2>,origin:Option<f32>)->Observed {
        let metadata=tensor::<1>(self.meta.clone()).reshape([self.matches,META]);
        let finished=metadata.clone().slice([0..self.matches,2..3]);
        let counter=metadata.clone().slice([0..self.matches,6..7]);
        let period=if let Some(origin)=origin {
            let ticks=metadata.slice([0..self.matches,0..1])-(origin*30.).round();
            let group=(ticks.clone()/512.).floor();let prior=((ticks-4.)/512.).floor();
            (group.not_equal(prior).float()+counter.clone().equal_elem(0).float()).clamp(0.,1.)
        }else{counter.clone().remainder_scalar(128).equal_elem(0).float()};
        let actors=tensor::<1>(self.state.clone()).reshape([self.matches,UNITS,STRIDE]).slice([0..self.matches,0..2,0..STRIDE]).reshape([self.matches*2,STRIDE]);
        let dead=actors.clone().slice([0..self.matches*2,UNIT_HEALTH..UNIT_HEALTH+1]).lower_equal_elem(0.0f32).float().reshape([self.matches*2]);
        let respawn=actors.slice([0..self.matches*2,RESPAWN_RESET..RESPAWN_RESET+1]).reshape([self.matches*2]);
        let counter=counter.expand([self.matches,2]).reshape([self.matches*2]);
        let life_reset=dead+(respawn.clone()-counter).abs().lower_elem(0.5f32).float()*respawn.greater_elem(0.0f32).float();
        // Preserve global128-step zero-state BPTT boundaries. Life resets are
        // additional per-actor masks, not a replacement for segment resets.
        let reset=((finished+period).expand([self.matches,2]).reshape([self.matches*2])+life_reset).greater_elem(0.5).float();
        prepare::launch(&self.state.client,CubeCount::Static(self.matches.div_ceil(32) as u32,1,1),CubeDim::new_1d(32),
            argument(&self.state),argument(&self.meta),argument(&self.prototypes),argument(&self.shots),argument(&self.commands),
            argument(&self.tree_alive),self.matches,self.tree_count,self.engine.pre_game_seconds);
        let previous=primitive(previous);let n=self.matches*2;
        let out=primitive(Tensor::<2>::zeros([n,WIDTH],&self.device));
        let refs=primitive(Tensor::<2>::zeros([n,TARGETS],&self.device));
        let scale=primitive(Tensor::<1>::from_data(UNIT_SCALES,&self.device));
        observe::launch(&self.state.client,CubeCount::Static(n.div_ceil(32) as u32,1,1),CubeDim::new_1d(32),
            argument(&self.state),argument(&self.shots),argument(&self.meta),argument(&self.trees),argument(&self.tree_alive),argument(&self.geometry),argument(&self.fow_blockers),argument(&self.static_vision),argument(&self.tree_map),
            argument(&self.spells),argument(&self.items),argument(&self.hero),argument(&self.fountains),argument(&self.extra_vision),argument(&scale),argument(&previous),argument(&out),argument(&refs),n,self.tree_count,self.engine.extra_vision.len(),self.engine.pre_game_seconds);
        Observed {packed:tensor(out),refs,reset}
    }
}

#[cube(launch)]
fn prepare(state:&mut [f32],meta:&mut [f32],proto:&[f32],shots:&mut [f32],commands:&mut [f32],trees:&mut [f32],
    matches:usize,tree_count:usize,pregame:f32){
    let m=ABSOLUTE_POS;
    if m<matches && meta[m*META+2]>0.0f32 {
        initialize(state,proto,m);reset_attack_phase(meta,m);meta[m*META]=-pregame*30.0f32;meta[m*META+1]+=1.0f32;meta[m*META+2]=0.0f32;
        for i in 0..SHOTS*SHOT_FIELDS {shots[m*SHOTS*SHOT_FIELDS+i]=0.0f32;}
        for i in 0..COMMAND_WORDS {commands[m*COMMAND_WORDS+i]=0.0f32;}
        for i in 0..tree_count {trees[m*tree_count+i]=1.0f32;}
    }
}



