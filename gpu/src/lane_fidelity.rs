//! Arbitrary admitted lane fixtures, using the production GPU kernels.
//! Per-tick host readback belongs only to this diagnostic executable command.
use anyhow::{Context,Result,ensure};
use burn::tensor::{Tensor,TensorData};
use serde_json::{Value,json};
use std::{collections::BTreeMap,path::Path};
use crate::{contract::*,world::*,runtime::{device,primitive,tensor},sampling::Sampler};

fn number(v:&Value,k:&str)->Result<f32>{v[k].as_f64().with_context(||format!("required native field{k} missing" )).map(|x|x as f32)}
fn upload(values:Vec<f32>,d:&burn::tensor::Device)->burn_cubecl::tensor::CubeTensor{
    let n=values.len();primitive(Tensor::<1>::from_data(TensorData::new(values,[n]),d))
}
fn target_name(value:f32,names:&BTreeMap<usize,String>)->Option<String>{
    if value<0. {None}else{names.get(&(value as usize)).cloned()}
}
fn rows(state:&[f32],names:&BTreeMap<usize,String>,local:f32)->Vec<Value>{
    names.iter().map(|(&u,id)|{
        let p=u*STRIDE;let s=&state[p..p+STRIDE];
        let attacking=if s[WINDUP]>0. || (s[ATTACK_READY]>local&&s[TARGET]==s[WINDUP_TARGET]){s[WINDUP_TARGET]}else{-1.};
        json!({"id":id,"team":s[TEAM]as usize+2,"kind":if u<2{"hero"}else if s[KIND]==2.{"melee"}else if s[KIND]==3.{"ranged"}else{"siege"},
            "x":s[UNIT_X],"y":s[UNIT_Y],"z":s[UNIT_Z],"facing":s[FACING].to_degrees(),
            "alive":s[ACTIVE]>0.&&s[UNIT_HEALTH]>0.,"health":s[UNIT_HEALTH],"mana":s[UNIT_MANA],
            "level":s[UNIT_LEVEL],"attack_hand":s[ATTACK_HAND],"windup":s[WINDUP],"attack_ready":s[ATTACK_READY],
            "attack_target":target_name(attacking,names),
            "aggro_target":target_name(s[TARGET],names),
            "last_hits":s[UNIT_LAST_HITS],"denies":s[UNIT_DENIES],"raze_stacks":s[UNIT_RAZE_STACKS]})
    }).collect()
}

fn tracking_rows(shots:&[f32],names:&BTreeMap<usize,String>)->Vec<Value>{
    shots.chunks_exact(SHOT_FIELDS).enumerate().filter(|(_,s)|s[0]==1.).map(|(slot,s)|
        json!({"slot":slot,"source":target_name(s[7],names),"target":target_name(s[3],names),
            "x":s[1],"y":s[2],"speed":s[5],"damage":s[6],"target_generation":s[4],
            "source_kind":s[9],"z":null,"z_mode":"not simulated for tracking shots"})).collect()
}

pub fn replay(engine:&Path,reference:&Path,output:&Path)->Result<Value>{
    let captured:Value=serde_json::from_slice(&std::fs::read(reference)?)?;
    ensure!(captured["capture_passed"]==true && captured["schema"]==1,"requires admitted lane reference");
    let d=device();let mut world=World::load(engine,1,&d)?;
    let mut results=vec![];
    for case in captured["cases"].as_array().context("missing cases")?{
        let id_hash=crate::runtime::bytes_sha256(case["id"].as_str().context("caseid missing")?.as_bytes());
        let rng=Sampler::new(1,u32::from_str_radix(&id_hash[..8],16)?,&d);
        let natural=case["mode"]=="natural";
        let capture_projectiles=case["capture_projectiles"]==true;
        let record=captured["records"].as_array().context("missing records")?.iter().find(|r|r["id"]==case["id"]).context("missing native case")?;
        let initial=&record["frames"][0];let clock=(number(initial,"dota_time")?*30.).round();
        // Match actual day/night and the time0 boundary. Internal timers use a
        // nonnegative epoch, even for native starts earlier than training's-83s.
        world.engine.pre_game_seconds=83.0f32.max(-clock/30.+1.);
        world.reset();
        // World::reset initializes units/meta; episode rollover normally clears
        // shots inside simulate. Explicit diagnostic setup must do so too.
        world.shots=primitive(Tensor::<1>::zeros([SHOTS*SHOT_FIELDS],&d));
        world.shot_hits=primitive(Tensor::<1>::zeros([SHOTS*8],&d));
        world.tree_alive=primitive(Tensor::<1>::ones([world.tree_count],&d));
        let mut damage_rules=world.engine.hero.clone();
        if case["damage_mode"]=="fixed_base_roll"{for v in &mut damage_rules[27..33]{*v=0.;}}
        world.hero=upload(damage_rules,&d);
        let mut state=tensor::<1>(world.state.clone()).into_data().try_to_vec::<f32>()?;
        if !natural{for u in 0..UNITS{state[u*STRIDE+ACTIVE]=0.;}}
        let mut names=BTreeMap::<usize,String>::new();let mut slots=BTreeMap::<String,usize>::new();let mut generations=BTreeMap::<usize,f32>::new();let mut next_slot=6;
        for row in initial["units"].as_array().context("missing units")?{
            let id=row["id"].as_str().context("missing stable unit id")?.to_string();
            let kind=row["kind"].as_str().context("kind missing")?;
            let team=number(row,"team")?as usize-2;
            let u=if kind=="hero"{team}else{let u=next_slot;next_slot+=1;u};
            ensure!(u<CREEP_END,"too many fixture units");
            let p=u*STRIDE;
            if kind!="hero"{
                state[p..p+STRIDE].fill(0.);
                let proto=match kind{"melee"=>6,"ranged"=>7,"siege"=>8,_=>anyhow::bail!("unsupported kind{kind}")};
                state[p..p+UNIT_COUNT].copy_from_slice(&world.engine.prototypes[proto]);state[p+KIND]=(proto-4)as f32;
                state[p+LANE_POSITION]=if natural{world.lane_start(team)as f32}else{-1.};state[p+BOUNTY]=34.;state[p+XP_BOUNTY]=57.;
            }
            state[p+TEAM]=team as f32;state[p+ACTIVE]=1.;state[p+GENERATION]=1.;
            state[p+TARGET]=-1.;state[p+WINDUP_TARGET]=-1.;state[p+CREDIT]=-1.;state[p+TARGET_GENERATION]=1.;state[p+WINDUP_GENERATION]=1.;
            state[p+ORDER]=0.;state[p+WINDUP]=0.;state[p+ATTACK_READY]=0.;
            for (field,key)in [(UNIT_X,"x"),(UNIT_Y,"y"),(UNIT_Z,"z"),(UNIT_HEALTH,"health"),(UNIT_HEALTH_MAX,"max_health"),
                (UNIT_MANA,"mana"),(UNIT_MANA_MAX,"max_mana"),(UNIT_ATTACK_DAMAGE,"damage"),(UNIT_ATTACK_INTERVAL,"interval"),
                (UNIT_ATTACK_POINT,"attack_point"),(UNIT_ATTACK_RANGE,"range"),(UNIT_ARMOR,"armor")]{state[p+field]=number(row,key)?;}
            for(field,key)in[(UNIT_HEALTH_REGEN,"health_regen"),(UNIT_MANA_REGEN,"mana_regen"),(UNIT_PROJECTILE_SPEED,"projectile_speed"),
                (UNIT_MAGIC_RESIST,"magic_resist"),(UNIT_MOVE_SPEED,"move_speed"),(UNIT_LEVEL,"level"),
                (UNIT_LAST_HITS,"last_hits"),(UNIT_DENIES,"denies")]{if let Some(x)=row[key].as_f64(){state[p+field]=x as f32;}}
            state[p+FACING]=number(row,"facing")?.to_radians();state[p+ATTACK_AIM]=state[p+FACING];
            names.insert(u,id.clone());slots.insert(id,u);
            if natural && u>=6{generations.insert(u,1.);}
        }
        ensure!(names.contains_key(&0)&&names.contains_key(&1),"both native hero states required");
        if let Some(targets)=case["initial_aggro"].as_object(){
            ensure!(case["warm_start"]["initial_aggro_source_sha256"].is_string(),"initial aggro requires explicit native provenance");
            for (actor,target)in targets{
                let u=*slots.get(actor).context("initial aggro actor missing")?;
                let v=*slots.get(target.as_str().context("initial aggro target must be an identity")?).context("initial aggro target missing")?;
                ensure!(u>=6&&v>=6&&state[u*STRIDE+TEAM]!=state[v*STRIDE+TEAM],"warm combat requires opposing lane NPCs");
                state[u*STRIDE+TARGET]=v as f32;state[u*STRIDE+TARGET_GENERATION]=state[v*STRIDE+GENERATION];
            }
        }
        if let Some(goals)=case["initial_lane_goals"].as_object(){
            ensure!(case["warm_start"]["initial_aggro_source_sha256"].is_string(),"initial lane goals require native provenance");
            for (actor,goal)in goals{
                let u=*slots.get(actor).context("initial lane actor missing")?;let side=state[u*STRIDE+TEAM]as usize;
                ensure!(u>=6&&!world.engine.lane_routes.is_empty(),"warm goal requires an NPC and directed routes");
                let x=number(goal,"x")?;let y=number(goal,"y")?;
                let index=world.engine.lane_routes[side].iter().position(|p|(p.x-x).abs()<0.1&&(p.y-y).abs()<0.1).context("native lane goal absent from configured route")?;
                state[u*STRIDE+LANE_POSITION]=(world.lane_start(side)+index)as f32;
            }
        }
        world.state=upload(state.clone(),&d);let mut meta=vec![0.;META];meta[0]=clock;world.meta=upload(meta,&d);
        // Controlled native captures disable stock spawning. Match this setup
        // through the kernel specialization; keep the actual native day clock.
        let duration=case["duration_ticks"].as_u64().context("duration missing")?as usize;
        ensure!(duration<=1800,"bounded lane fixture required");
        let mut first=json!({"tick":0,"units":rows(&state,&names,world.engine.pre_game_seconds+clock/30.)});
        if capture_projectiles{first["projectiles"]=json!([]);}
        let mut frames=vec![first];
        let orders=case["orders"].as_array().context("orders missing")?;let mut executed=vec![];
        for tick in 0..duration{
            let mut queue=vec![0.;COMMAND_WORDS];let mut actor_counts=[0usize;2];
            for order in orders.iter().filter(|o|o["tick"].as_u64()==Some(tick as u64)){
                let actor=slots[order["actor"].as_str().context("actor missing")?];ensure!(actor<2,"hero orders only");
                ensure!(actor_counts[actor]<4,"too many same-tick orders");
                let k=(actor*4+actor_counts[actor])*COMMAND_FIELDS;actor_counts[actor]+=1;
                queue[k]=clock+tick as f32-1.;queue[k+1]=number(order,"action")?;
                queue[k+4]=order["target"].as_str().map(|id|slots[id]as f32).unwrap_or(-1.);
                queue[k+5]=order["x"].as_f64().unwrap_or(0.)as f32;queue[k+6]=order["y"].as_f64().unwrap_or(0.)as f32;
                queue[k+7]=1.;queue[k+8]=1.;
            }
            world.commands=upload(queue,&d);let _=world.step_ticks_with_waves(&rng,1,natural);
            if actor_counts.iter().sum::<usize>()>0{
                let remaining=tensor::<1>(world.commands.clone()).into_data().try_to_vec::<f32>()?;
                for actor in 0..2{for slot in 0..actor_counts[actor]{ensure!(remaining[(actor*4+slot)*COMMAND_FIELDS+7]==0.,"GPU command was not consumed at declaredtick{tick}");}}
                executed.extend(orders.iter().filter(|o|o["tick"].as_u64()==Some(tick as u64)).cloned());
            }
            state=tensor::<1>(world.state.clone()).into_data().try_to_vec::<f32>()?;
            if !natural{
                // Keep the guard: undeclared units must never enter a fixture.
                for u in 6..CREEP_END{if state[u*STRIDE+ACTIVE]>0.&&!names.contains_key(&u){
                    for &known in names.keys(){let p=u*STRIDE;let q=known*STRIDE;
                        ensure!((state[p+UNIT_X]-state[q+UNIT_X]).hypot(state[p+UNIT_Y]-state[q+UNIT_Y])>2500.,"undeclared wave entered controlledarena");
                    }
                }}
            }
            if natural{
                for u in 6..CREEP_END{let p=u*STRIDE;
                    if state[p+ACTIVE]>0.&&generations.get(&u).copied()!=Some(state[p+GENERATION]){
                        let id=format!("gpu-creep-{u}-{}",state[p+GENERATION]);names.insert(u,id);generations.insert(u,state[p+GENERATION]);
                    }
                }
            }
            let mut frame=json!({"tick":tick+1,"units":rows(&state,&names,world.engine.pre_game_seconds+(clock+tick as f32+1.)/30.)});
            if capture_projectiles{
                let shots=tensor::<1>(world.shots.clone()).into_data().try_to_vec::<f32>()?;
                frame["projectiles"]=json!(tracking_rows(&shots,&names));
            }
            frames.push(frame);
            let finished=tensor::<1>(world.meta.clone()).slice([2..3]).into_scalar::<f32>();
            ensure!(finished==0.,"fixture crossed an episode terminal at{}",tick+1);
        }
        results.push(json!({"id":case["id"],"frames":frames,"orders":executed,"rng_seed":u32::from_str_radix(&id_hash[..8],16)?,"native_clock_ticks":clock,"internal_epoch_seconds":world.engine.pre_game_seconds}));
    }
    let report=json!({"schema":1,"native_sha256":crate::runtime::file_sha256(reference)?,"environment_sha256":world.fingerprint,
        "records":results,"training_eligible":false,"scope":"ProductionGPU transitions with native initial public state and declared orders; diagnostic host setup/readback only. Naturalcreep birth identities require admission-time matching."});
    crate::artifacts::atomic_json(output,&report)?;
    Ok(json!({"output":output,"cases":results.len(),"training_updates":0}))
}
