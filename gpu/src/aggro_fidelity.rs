//! Host setup/readback for isolated native diagnostics only. Never used by learning.
use anyhow::{Context,Result,ensure};
use burn::tensor::{Tensor,TensorData};
use serde_json::{Value,json};
use std::path::Path;
use crate::{contract::*,runtime::{device,tensor,primitive},world::*,sampling::Sampler};

fn number(v:&Value,k:&str)->Result<f64>{v[k].as_f64().with_context(||format!("native field {k} missing"))}
fn slot(entity:&Value,units:&[Value])->i32{
    units.iter().position(|u|u["entity"]==*entity).map(|i|[0,1,6,7][i]).unwrap_or(-1)
}

pub fn validate(engine:&Path,reference:&Path)->Result<Value>{
    let captured:Value=serde_json::from_slice(&std::fs::read(reference)?)?;
    ensure!(captured["complete"]==true,"incomplete native capture");
    ensure!(captured["parameters"]["actor_mode"]=="bot","requires real Bot API orders");
    let suite=captured["suite"].as_str().context("suite missing")?;
    ensure!(matches!(suite,"radius"|"pull"|"priority"),"unsupported diagnostic suite");
    let orders=captured["bot_orders"].as_array().context("bot orders missing")?;
    ensure!(orders.iter().all(|o|o["accepted"]==true),"rejected native orders");
    ensure!(!captured["bot_frames"].as_array().context("Bot API frames missing")?.is_empty(),"missing paired Bot API frames");
    let frames=captured["frames"].as_array().context("frames missing")?;
    let mut cases=vec![];
    for case in captured["cases"].as_array().context("cases missing")? {
        let native:Vec<_>=frames.iter().filter(|f|f["case"]==case["case"]).collect();
        let origin=native.iter().copied().find(|f|number(f,"elapsed").unwrap_or(0.)>=if suite=="priority"{0.}else{0.89}).context("warm start missing")?;
        let elapsed=number(origin,"elapsed")?;
        let units=origin["units"].as_array().context("units missing")?;
        ensure!(units.len()==4,"requires two SF and two diagnostic creeps");
        let d=device();let mut world=World::load(engine,1,&d)?;let rng=Sampler::new(1,20261101,&d);
        let mut state=tensor::<1>(world.state.clone()).into_data().try_to_vec::<f32>()?;
        for u in 0..UNITS{state[u*STRIDE+ACTIVE]=0.;}
        let clock=-60.*30.;let local=world.engine.pre_game_seconds-60.;
        for (i,row)in units.iter().enumerate(){
            let u=[0,1,6,7][i];let p=u*STRIDE;
            if i>=2{
                for f in 0..STRIDE{state[p+f]=0.;}
                let prototype=if i==3||captured["kind"]=="melee"{6}else if captured["kind"]=="ranged"{7}else{8};
                state[p..p+UNIT_COUNT].copy_from_slice(&world.engine.prototypes[prototype]);
                state[p+KIND]=(prototype-4)as f32;state[p+TEAM]=if i==2{1.}else{0.};
                state[p+LANE_POSITION]=if i==2&&case["natural_wave"]==true{0.}else{-1.};
            }
            state[p+ACTIVE]=1.;state[p+TARGET]=-1.;state[p+WINDUP_TARGET]=-1.;state[p+CREDIT]=-1.;state[p+GENERATION]=1.;
            state[p+ORDER]=0.;state[p+WINDUP]=0.;state[p+ATTACK_READY]=0.;
            for(field,key)in[(UNIT_X,"x"),(UNIT_Y,"y"),(UNIT_Z,"z"),(UNIT_HEALTH,"health"),(UNIT_HEALTH_MAX,"max_health"),(UNIT_ATTACK_RANGE,"range"),(UNIT_ATTACK_POINT,"attack_point")]{state[p+field]=number(row,key)?as f32;}
            state[p+FACING]=(number(row,"facing")?as f32).to_radians();state[p+ATTACK_AIM]=state[p+FACING];
            state[p+UNIT_ATTACK_INTERVAL]=(1./number(row,"attacks_per_second")?)as f32;
            if i>=2 && suite!="priority"{
                let target=slot(&row["aggro_target"],units);state[p+TARGET]=target as f32;state[p+TARGET_GENERATION]=1.;
                let attack=slot(&row["attack_target"],units);state[p+WINDUP_TARGET]=attack as f32;state[p+WINDUP_GENERATION]=1.;
                // Native last_attack is the projectile/melee release time on GameTime.
                // Radius fixtures have already released once before the .9s warm start.
                let base=if captured["kind"]=="siege"&&i==2{world.engine.prototypes[8][UNIT_ATTACK_INTERVAL]}else{1.};
                let point=(state[p+UNIT_ATTACK_POINT]*state[p+UNIT_ATTACK_INTERVAL]/base*30.).ceil()/30.;
                let ready=local+ (number(row,"last_attack")?-number(origin,"game_time")?)as f32+state[p+UNIT_ATTACK_INTERVAL]-point;
                state[p+ATTACK_READY]=ready.max(local);
                if number(row,"last_attack")?==0. && attack>=0 {
                    let start=native.iter().copied().find(|f|number(f,"elapsed").unwrap_or(f64::INFINITY)<=elapsed && f["units"][i]["attack_target"]==row["attack_target"]).context("initial windup missing")?;
                    let attack_start=local+(number(start,"game_time")?-number(origin,"game_time")?)as f32;
                    state[p+WINDUP]=attack_start+point;state[p+ATTACK_READY]=attack_start+state[p+UNIT_ATTACK_INTERVAL];
                    ensure!(state[p+WINDUP]>local,"unreleased attack has expired windup");
                }
            }
        }
        // Controlled native fixtures add these exact unobstructed viewers. Keep
        // natural map sight for the radius suite; no hidden state in observations.
        if case["controlled_all_vision"]==true||case["controlled_target_vision"]==true{
            let mut vision:Vec<f32>=world.engine.extra_vision.iter().flat_map(|v|[v.x,v.y,v.team,v.day,v.night,if v.unobstructed{1.}else{0.}]).collect();
            if case["controlled_all_vision"]==true{for side in [0.,1.]{vision.extend([80.,-816.,side,2500.,2500.,1.]);}}
            else{vision.extend([state[STRIDE+UNIT_X],state[STRIDE+UNIT_Y],0.,128.,128.,1.]);}
            // step_ticks passes engine.extra_vision.len() as the launch count.
            while world.engine.extra_vision.len()<vision.len()/6{world.engine.extra_vision.push(StaticVision{x:0.,y:0.,team:0.,day:0.,night:0.,unobstructed:true});}
            let n=vision.len();world.extra_vision=primitive(Tensor::<1>::from_data(TensorData::new(vision,[n]),&d));
        }
        world.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
        let mut metadata=vec![0.;META];metadata[0]=clock;
        world.meta=primitive(Tensor::<1>::from_data(TensorData::new(metadata,[META]),&d));
        let pulses:Vec<f64>=orders.iter().filter(|o|o["case"]==case["case"]).map(|o|number(o,"elapsed")).collect::<Result<_>>()?;
        ensure!(pulses.len()==case["pulses"].as_array().context("pulses missing")?.len(),"native pulse count mismatch");
        let end=number(case,"duration").unwrap_or(7.).min(if suite=="radius"{4.}else{7.});
        let mut simulated=vec![];let mut gpu_frames=vec![];
        for tick in 0..((end-elapsed)*30.).ceil()as usize{
            let t=elapsed+tick as f64/30.;
            let pulse=pulses.iter().any(|p|(*p-t).abs()<1./60.+0.001);
            let moving=case["pull"]==true&&t>=1.2&&t<3.35;
            // step_ticks consumes commands before advancing meta. Native Think
            // orders here belong to this execution tick (no RPC offset).
            let mut commands=vec![0.;COMMAND_WORDS];commands[0]=clock+tick as f32-1.;
            commands[1]=if pulse{3.}else if moving{2.}else{4.};commands[4]=1.;commands[5]=-1200.;commands[6]=-816.;commands[7]=1.;commands[8]=1.;
            world.commands=primitive(Tensor::<1>::from_data(TensorData::new(commands,[COMMAND_WORDS]),&d));
            let _=world.step_ticks(&rng,1);
            let full=tensor::<1>(world.state.clone()).slice([0..8*STRIDE]).into_data().try_to_vec::<f32>()?;
            let s=&full[6*STRIDE..7*STRIDE];
            let time=local+(tick+1)as f32/30.;
            let attack=if s[WINDUP]>0.||(s[ATTACK_READY]>time&&s[TARGET]==s[WINDUP_TARGET]){s[WINDUP_TARGET]as i32}else{-1};
            simulated.push((t+1./30.,s[UNIT_X]as f64,s[UNIT_Y]as f64,s[TARGET]as i32,attack,s[WINDUP]as f64));
            let rows=[0,1,6,7].map(|u|{let p=u*STRIDE;json!({"slot":u,"x":full[p+UNIT_X],"y":full[p+UNIT_Y],"target":full[p+TARGET],"windup":full[p+WINDUP],"ready":full[p+ATTACK_READY],"pending":full[p+AGGRO_PENDING]})});
            gpu_frames.push(json!({"elapsed":t+1./30.,"units":rows}));
        }
        let mut phases=vec![];
        for phase in -1..=1{
            let mut count=0;let mut goal_errors=0;let mut attack_errors=0;let mut position_sum=0.;let mut position_max=0f64;
            for f in &native{
                let t=number(f,"elapsed")?;if t<elapsed+0.05||t>end-0.05{continue;}
                if let Some(s)=simulated.iter().min_by(|a,b|(a.0-t-phase as f64/30.).abs().total_cmp(&(b.0-t-phase as f64/30.).abs())){
                    let row=&f["units"][2];let native_units=f["units"].as_array().unwrap();
                    let position=((s.1-number(row,"x")?).powi(2)+(s.2-number(row,"y")?).powi(2)).sqrt();
                    position_sum+=position;position_max=position_max.max(position);count+=1;
                    if s.3!=slot(&row["aggro_target"],native_units){goal_errors+=1;}
                    if s.4!=slot(&row["attack_target"],native_units){attack_errors+=1;}
                }
            }
            phases.push(json!({"phase_ticks":phase,"samples":count,"aggro_target_errors":goal_errors,"attack_target_errors":attack_errors,"position_mean":position_sum/(count as f64).max(1.),"position_max":position_max,"passed":count>=30&&goal_errors<=2&&attack_errors<=2&&position_max<=20.}));
        }
        let native_events:Vec<_>=native.iter().filter(|f|number(f,"elapsed").unwrap_or(0.)>=elapsed).scan((-2,-2),|previous,f|{let row=&f["units"][2];let us=f["units"].as_array().unwrap();let pair=(slot(&row["aggro_target"],us),slot(&row["attack_target"],us));let changed=pair!=*previous;*previous=pair;Some(if changed{Some(json!({"elapsed":f["elapsed"],"target":pair.0,"attack":pair.1}))}else{None})}).flatten().collect();
        let gpu_events:Vec<_>=simulated.iter().scan((-2,-2),|previous,s|{let pair=(s.3,s.4);let changed=pair!=*previous;*previous=pair;Some(if changed{Some(json!({"elapsed":s.0,"target":s.3,"attack":s.4}))}else{None})}).flatten().collect();
        cases.push(json!({"case":case,"passed":phases.iter().any(|p|p["passed"]==true),"phases":phases,"native_events":native_events,"gpu_events":gpu_events,"gpu_frames":gpu_frames}));
    }
    Ok(json!({"passed":cases.iter().all(|c|c["passed"]==true),"reference":reference,"cases":cases,"training_eligible":false,"scope":"Four stock units; radius/pull warm start at .9s including old attack cycle; priority begins without target. Native pregame clock translated to -60s, preserving phase. Fixed sampling phase +/-1tick, <=2 target-transition samples, <=20unit position. RNG damage, manual NPC activation phase and full-lane navigation not certified."}))
}
