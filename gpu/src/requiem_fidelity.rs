//! Requiem diagnostic only: native stock spell vs the production GPU kernel.
use anyhow::{Context,Result,ensure};
use burn::tensor::{Tensor,TensorData};
use serde_json::{Value,json};
use std::path::Path;
use crate::{contract::*,runtime::{device,tensor,primitive},world::*,sampling::Sampler};
pub fn validate(engine:&Path,reference:&Path,fear_distance:f32,fear_from_caster:bool,fear_fixed_goal:bool)->Result<Value>{
    ensure!(fear_distance.is_finite()&&fear_distance>=0.&&fear_distance<=5000.,"invalid diagnostic fear lookahead");
    let native:Value=serde_json::from_slice(&std::fs::read(reference)?)?;
    ensure!(native["complete"]==true&&native["suite"]=="requiem"&&native["parameters"]["probe_magic"]!=true,"requires clean native Requiem capture without damage probes");
    let frames=native["frames"].as_array().context("frames missing")?;let mut results=vec![];
    for case in native["cases"].as_array().context("cases missing")?{
        let reference:Vec<_>=frames.iter().filter(|f|f["case"]==case["case"]).collect();let first=reference.first().context("case frames missing")?;
        let elapsed=first["elapsed"].as_f64().context("elapsed missing")?;
        let order=native["orders"].as_array().context("orders missing")?.iter().find(|o|o["case"]==case["case"]&&o["action"]=="requiem").context("cast order missing")?;
        let d=device();let mut world=World::load(engine,1,&d)?;let rng=Sampler::new(1,20261104,&d);
        if fear_distance>0. {
            world.engine.hero.resize(42,0.);world.engine.hero[39]=fear_distance;world.engine.hero[40]=if fear_from_caster{1.}else{0.};world.engine.hero[41]=if fear_fixed_goal{1.}else{0.};world.hero=primitive(Tensor::<1>::from_data(TensorData::new(world.engine.hero.clone(),[42]),&d));
        }
        let mut s=tensor::<1>(world.state.clone()).into_data().try_to_vec::<f32>()?;
        let source=first["units"][0]["team"].as_u64().context("caster team missing")?as usize-2;ensure!(source<2,"invalid caster team");let target=1-source;
        for u in 2..UNITS{s[u*STRIDE+ACTIVE]=0.;}
        for i in 0..2{let u=if i==0{source}else{target};let p=u*STRIDE;let row=&first["units"][i];
            for(field,key)in[(UNIT_X,"x"),(UNIT_Y,"y"),(UNIT_Z,"z"),(UNIT_HEALTH,"health"),(UNIT_HEALTH_MAX,"max_health"),(UNIT_MANA,"mana"),(UNIT_LEVEL,"level")]{s[p+field]=row[key].as_f64().with_context(||format!("missing {key}"))?as f32;}
            s[p+FACING]=if i==0{0.}else{std::f32::consts::PI};s[p+ATTACK_AIM]=s[p+FACING];
        }
        s[source*STRIDE+UNIT_SOULS]=case["souls"].as_f64().context("souls missing")?as f32;
        let initial_hp=s[target*STRIDE+UNIT_HEALTH]as f64;
        world.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
        let mut meta=vec![0.;META];meta[0]=-1800.;world.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
        let mut commands=vec![0.;COMMAND_WORDS];let k=source*4*COMMAND_FIELDS;commands[k]=-1800.+((order["elapsed"].as_f64().unwrap()-elapsed)*30.).round()as f32-1.;commands[k+1]=9.;commands[k+7]=1.;
        world.commands=primitive(Tensor::<1>::from_data(TensorData::new(commands,[COMMAND_WORDS]),&d));
        let mut gpu=vec![];let mut first_hit=None;let mut fear_start=None;let mut fear_end=None;let mut previous_hp=initial_hp;
        for tick in 0..209{
            let _=world.step_ticks(&rng,1);let state=tensor::<1>(world.state.clone()).slice([target*STRIDE..(target+1)*STRIDE]).into_data().try_to_vec::<f32>()?;
            let t=elapsed+(tick+1)as f64/30.;let timer=world.engine.pre_game_seconds-60.+tick as f32/30.;
            if (state[UNIT_HEALTH]as f64)<previous_hp-1. &&first_hit.is_none(){first_hit=Some(t);}
            if state[FEAR_END]>timer {if fear_start.is_none(){fear_start=Some(t);}fear_end=Some(t+1./30.);}
            previous_hp=state[UNIT_HEALTH]as f64;
            gpu.push(json!({"elapsed":t,"x":state[UNIT_X],"y":state[UNIT_Y],"health":state[UNIT_HEALTH],"fear_remaining":(state[FEAR_END]-timer).max(0.),"magic_resist":state[UNIT_MAGIC_RESIST]}));
        }
        let first_native_hit=reference.windows(2).find(|pair|pair[1]["units"][1]["health"].as_f64().unwrap()<pair[0]["units"][1]["health"].as_f64().unwrap()-1.).map(|pair|pair[1]["elapsed"].as_f64().unwrap());
        let native_fear:Vec<_>=reference.iter().filter(|f|f["units"][1]["modifiers"].as_array().is_some_and(|ms|ms.iter().any(|m|m["name"]=="modifier_nevermore_requiem_fear"))).collect();
        let native_fear_start=native_fear.first().map(|f|f["elapsed"].as_f64().unwrap());let native_fear_end=native_fear.last().map(|f|f["elapsed"].as_f64().unwrap()+1./30.);
        let mut phases=vec![];
        for phase in -1..=1{
            let mut hp_max=0f64;let mut position_max=0f64;
            for frame in &reference{let t=frame["elapsed"].as_f64().unwrap();if t<elapsed+0.05||t>6.9{continue;}
                let value=gpu.iter().min_by(|a,b|(a["elapsed"].as_f64().unwrap()-t-phase as f64/30.).abs().total_cmp(&(b["elapsed"].as_f64().unwrap()-t-phase as f64/30.).abs())).unwrap();let row=&frame["units"][1];
                hp_max=hp_max.max((value["health"].as_f64().unwrap()-row["health"].as_f64().unwrap()).abs());
                position_max=position_max.max(((value["x"].as_f64().unwrap()-row["x"].as_f64().unwrap()).powi(2)+(value["y"].as_f64().unwrap()-row["y"].as_f64().unwrap()).powi(2)).sqrt());
            }
            let timing=match(first_hit,first_native_hit){(None,None)=>true,(Some(g),Some(n))=>(g-n-phase as f64/30.).abs()<=1./30.+0.001,_=>false};
            let fear=match(fear_end,native_fear_end){(None,None)=>true,(Some(g),Some(n))=>(g-n-phase as f64/30.).abs()<=2./30.+0.001,_=>false};
            phases.push(json!({"phase_ticks":phase,"max_hp_error":hp_max,"max_position_error":position_max,"impact_timing_passed":timing,"fear_end_passed":fear,"passed":timing&&fear&&hp_max<=3.&&position_max<=20.}));
        }
        results.push(json!({"case":case,"passed":phases.iter().any(|p|p["passed"]==true),"native_first_hit":first_native_hit,"gpu_first_hit":first_hit,"native_fear_start":native_fear_start,"native_fear_end":native_fear_end,"gpu_fear_start":fear_start,"gpu_fear_end":fear_end,"phases":phases,"gpu_frames":gpu}));
    }
    Ok(json!({"passed":results.iter().all(|c|c["passed"]==true),"reference":reference,"scenarios":results,"training_eligible":false,"diagnostic_fear_distance":fear_distance,"diagnostic_fear_from_caster":fear_from_caster,"diagnostic_fear_fixed_goal":fear_fixed_goal,"scope":"Level6, Requiem1, Wraith band,2/8 souls; actual GPU physics vs stock native spell. Native pregame translated to -60s. Fixed phase +/-1tick; <=3HP,<=20units; damage probes excluded. Nonzero fear-distance overrides only this diagnostic, not the training engine."}))
}
