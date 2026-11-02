//! Differential Dota fixtures. CPU setup/readback is confined to this validation command.
use anyhow::{Result,ensure,Context};
use burn::tensor::{Tensor,TensorData};
use crate::{runtime::{device,tensor,primitive},world::*,contract::*,sampling::Sampler};
use std::path::Path;

/// Offline differential capture. These host reads are never used by training.
pub fn idle_lane(engine:&Path,reference:&Path,seconds:usize)->Result<serde_json::Value>{
    ensure!(seconds>0 && seconds<=180,"idle diagnostic must be1..180seconds");
    let captured:serde_json::Value=serde_json::from_slice(&std::fs::read(reference)?)?;
    ensure!(captured["complete"]==true && captured["idle_capture"]==true,"requires an idle native capture, without admitted attack pulses");
    let frames=captured["frames"].as_array().context("native frames missing")?;
    let d=device();let mut world=World::load(engine,1,&d)?;let rng=Sampler::new(1,123,&d);
    let mut state=tensor::<1>(world.state.clone()).into_data().try_to_vec::<f32>()?;
    for u in 0..2 {
        let frame=frames.iter().filter(|f|f["team"].as_u64()==Some(u as u64+2)).min_by(|a,b|a["time"].as_f64().unwrap().abs().total_cmp(&b["time"].as_f64().unwrap().abs())).context("initial native hero missing")?;
        ensure!(frame["time"].as_f64().unwrap().abs()<0.05,"native time0 missing");let hero=&frame["hero"];
        for (field,key)in[(UNIT_X,"x"),(UNIT_Y,"y"),(UNIT_Z,"z"),(UNIT_HEALTH,"health")] {state[u*STRIDE+field]=hero[key].as_f64().context("native hero field missing")?as f32;}
        state[u*STRIDE+FACING]=hero["facing"].as_f64().unwrap()as f32*std::f32::consts::PI/180.;
            state[u*STRIDE+ATTACK_AIM]=state[u*STRIDE+FACING];
    }
    world.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));world.meta=primitive(Tensor::<1>::zeros([META],&d));
    let mut samples=vec![];
    for tick in 0..seconds*30 {
        let _=world.step_ticks(&rng,1);
        if tick%30==0 {
            let s=tensor::<1>(world.state.clone()).into_data().try_to_vec::<f32>()?;let time=(tick+1)as f64/30.;let mut native=vec![];
            for team in 2..=3 {let f=frames.iter().filter(|f|f["team"].as_u64()==Some(team)).min_by(|a,b|(a["time"].as_f64().unwrap()-time).abs().total_cmp(&(b["time"].as_f64().unwrap()-time).abs())).context("native side missing")?;
                let creeps:Vec<_>=f["creeps"].as_array().context("native creeps missing")?.iter().filter(|c|c["team"].as_u64()==Some(team)).cloned().collect();
                native.push(serde_json::json!({"time":f["time"],"team":team,"hero":f["hero"],"creeps":creeps}));}
            let gpu:Vec<_>=s.chunks_exact(STRIDE).enumerate().filter(|(u,v)|*u<2 || (*u>=6 && *u<CREEP_END && v[ACTIVE]>0.)).map(|(u,v)|serde_json::json!({"id":u,"team":v[TEAM]+2.,"kind":v[KIND],"x":v[UNIT_X],"y":v[UNIT_Y],"health":v[UNIT_HEALTH],"max_health":v[UNIT_HEALTH_MAX],"target":v[TARGET],"level":v[UNIT_LEVEL]})).collect();
            samples.push(serde_json::json!({"time":time,"gpu":gpu,"native":native}));
        }
    }
    Ok(serde_json::json!({"reference":reference,"environment_sha256":world.fingerprint,"training_eligible":false,"scope":"Idle heroes at native time0 positions; natural GPU wave spawning, movement, targeting and combat; native samples matched by side/time. Host readback is diagnostic only.","samples":samples}))
}

/// Native Think frames and GPU post-tick frames can differ by one sampling tick.
/// A single fixed phase must explain both position and facing for each scenario.
pub fn movement(engine:&Path,reference:&Path)->Result<serde_json::Value>{
    let captured:serde_json::Value=serde_json::from_slice(&std::fs::read(reference)?)?;
    ensure!(captured["capture_passed"]==true && captured["suite"]=="movement","invalid movement capture");
    let mut results=vec![];
    for case in captured["scenarios"].as_array().context("scenarios missing")? {
        let d=device();let mut world=World::load(engine,1,&d)?;let rng=Sampler::new(1,123,&d);
        let first=case["orders"][0]["tick"].as_i64().context("order tick missing")?;let origin=first-6;
        let frames=case["frames"].as_array().context("frames missing")?;
        let mut state=tensor::<1>(world.state.clone()).into_data().try_to_vec::<f32>()?;
        for u in 2..UNITS {state[u*STRIDE+ACTIVE]=0.;}
        for u in 0..2 {
            let frame=frames.iter().find(|f|f["team"].as_u64()==Some(u as u64+2) && f["tick"].as_i64()==Some(origin)).context("initial frame missing")?;
            for (field,key) in [(UNIT_X,"x"),(UNIT_Y,"y"),(UNIT_Z,"z")] {state[u*STRIDE+field]=frame[key].as_f64().context("coordinate missing")? as f32;}
            state[u*STRIDE+FACING]=frame["facing"].as_f64().context("facing missing")? as f32*std::f32::consts::PI/180.;
            state[u*STRIDE+ATTACK_AIM]=state[u*STRIDE+FACING];
        }
        let frame=frames.iter().find(|f|f["team"]==2 && f["tick"].as_i64()==Some(first)).context("execution frame missing")?;
        let clock=(frame["time"].as_f64().context("clock missing")?*30.).round()as f32-6.;let mut meta=vec![0.;META];meta[0]=clock;
        let mut commands=vec![0.;COMMAND_WORDS];
        for (slot,order)in case["orders"].as_array().context("orders missing")?.iter().enumerate(){
            let k=slot*COMMAND_FIELDS;commands[k]=clock+(order["tick"].as_i64().unwrap()-origin)as f32;
            commands[k+1]=order["action"].as_u64().unwrap()as f32;commands[k+5]=order["x"].as_f64().unwrap_or(0.)as f32;commands[k+6]=order["y"].as_f64().unwrap_or(0.)as f32;commands[k+7]=1.;
        }
        world.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
        world.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));world.commands=primitive(Tensor::<1>::from_data(TensorData::new(commands,[COMMAND_WORDS]),&d));
        let mut simulated=vec![];
        for tick in origin+1..=first+65 {let _=world.step_ticks(&rng,1);let s=tensor::<1>(world.state.clone()).slice([0..STRIDE]).into_data().try_to_vec::<f32>()?;
            simulated.push((tick,s[UNIT_X]as f64,s[UNIT_Y]as f64,s[FACING]as f64*180./std::f64::consts::PI));}
        let mut phases=vec![];
        for phase in -1..=1 {
            let mut position_max=0f64;let mut position_sum=0.;let mut facing_max=0f64;let mut count=0;
            for f in frames.iter().filter(|f|f["team"]==2 && f["tick"].as_i64().is_some_and(|t|t>=first && t<=first+60)){
                let tick=f["tick"].as_i64().unwrap()+phase;
                if let Some((_,x,y,a))=simulated.iter().find(|v|v.0==tick){
                    let error=((x-f["x"].as_f64().unwrap()).powi(2)+(y-f["y"].as_f64().unwrap()).powi(2)).sqrt();
                    position_max=position_max.max(error);position_sum+=error;
                    let angle=(a-f["facing"].as_f64().unwrap()+180.).rem_euclid(360.)-180.;facing_max=facing_max.max(angle.abs());count+=1;
                }
            }
            phases.push(serde_json::json!({"phase_ticks":phase,"samples":count,"max_position_error":position_max,"mean_position_error":position_sum/(count as f64).max(1.),"max_facing_error_degrees":facing_max,"passed":count>=30 && position_max<=16. && facing_max<=15.}));
        }
        results.push(serde_json::json!({"name":case["name"],"passed":phases.iter().any(|v|v["passed"]==true),"phases":phases}));
    }
    Ok(serde_json::json!({"passed":results.iter().all(|v|v["passed"]==true),"reference":reference,"scenarios":results,"position_tolerance_units":16,"facing_tolerance_degrees":15,"sampling_phase_tolerance_ticks":1,"scope":"Direct movement,90/180-degree turns,reissue and stop; first60 ticks after each command. Validation readback only; no training."}))
}

pub fn combat(engine:&Path,reference:&Path)->Result<serde_json::Value>{
    let captured:serde_json::Value=serde_json::from_slice(&std::fs::read(reference)?)?;
    ensure!(captured["capture_passed"]==true,"Dota capture did not pass its own assertions");
    let mut results=vec![];let mut raze_stacks=0f32;let mut raze_until=0i64;
    for case in captured["scenarios"].as_array().context("scenarios missing")? {
        let d=device();let mut world=World::load(engine,1,&d)?;let rng=Sampler::new(1,123,&d);
        let first_tick=case["orders"][0]["tick"].as_i64().unwrap();let origin_tick=first_tick-6;
        let frames=case["frames"].as_array().unwrap();
        let mut state=tensor::<1>(world.state.clone()).into_data().try_to_vec::<f32>()?;
        for u in 2..UNITS {state[u*STRIDE+ACTIVE]=0.;}
        for u in 0..2 {
            let frame=frames.iter().filter(|v|v["team"].as_u64()==Some(u as u64+2)).min_by_key(|v|(v["tick"].as_i64().unwrap()-origin_tick).abs()).context("side missing")?;
            let p=u*STRIDE;
            for (field,key) in [(UNIT_X,"x"),(UNIT_Y,"y"),(UNIT_Z,"z"),(UNIT_HEALTH,"health"),(UNIT_MANA,"mana"),
                (UNIT_ATTACK_INTERVAL,"attack_interval"),(UNIT_ATTACK_POINT,"attack_point"),(UNIT_ATTACK_DAMAGE,"damage"),(UNIT_ARMOR,"armor")] {
                state[p+field]=frame[key].as_f64().with_context(||format!("missing {key}"))? as f32;
            }
            state[p+FACING]=frame["facing"].as_f64().unwrap() as f32*std::f32::consts::PI/180.;
            state[p+ATTACK_AIM]=state[p+FACING];
        }
        let frame=frames.iter().find(|f|f["team"]==2 && f["tick"].as_i64()==Some(first_tick)).context("execution frame missing")?;
        let initial_clock=(frame["time"].as_f64().unwrap()*30.).round() as f32-6.;
        if origin_tick>=raze_until {raze_stacks=0.;}
        if raze_stacks>0. {
            state[STRIDE+UNIT_RAZE_STACKS]=raze_stacks;
            state[STRIDE+STACK_END]=(initial_clock+world.engine.pre_game_seconds*30.+(raze_until-origin_tick) as f32)/30.;
        }
        let mut metadata=vec![0f32;META];metadata[0]=initial_clock;
        let mut commands=vec![0f32;COMMAND_WORDS];
        for (slot,order) in case["orders"].as_array().unwrap().iter().enumerate(){
            let start=slot*COMMAND_FIELDS;commands[start]=initial_clock+(order["tick"].as_i64().unwrap()-origin_tick) as f32;
            commands[start+1]=order["action"].as_u64().unwrap() as f32;commands[start+4]=1.;commands[start+7]=1.;
            commands[start+5]=order["x"].as_f64().unwrap_or(0.)as f32;commands[start+6]=order["y"].as_f64().unwrap_or(0.)as f32;
        }
        world.meta=primitive(Tensor::<1>::from_data(TensorData::new(metadata,[META]),&d));
        world.commands=primitive(Tensor::<1>::from_data(TensorData::new(commands,[COMMAND_WORDS]),&d));
        let mut previous_hp=state[STRIDE+UNIT_HEALTH];
        let mut releases=vec![];let mut release_count=0.;let mut impact=None;let mut spell_impact=None;let mut damage=None;
        let check_facing=case["name"].as_str().unwrap_or("").starts_with("attack_turn_");let mut gpu_facing=vec![];
        world.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
        for index in 0..80 {
            let _=world.step_ticks(&rng,1);
            if check_facing {gpu_facing.push((origin_tick+index+1,tensor::<1>(world.state.clone()).slice([FACING..FACING+1]).into_scalar::<f32>()as f64*180./std::f64::consts::PI));}
            let m=tensor::<1>(world.meta.clone()).into_data().try_to_vec::<f32>()?;
            if m[8]>release_count {releases.push((m[9]-initial_clock).round() as i64+origin_tick);release_count=m[8];}
            let hp=tensor::<1>(world.state.clone()).slice([STRIDE+UNIT_HEALTH..STRIDE+UNIT_HEALTH+1]).into_scalar::<f32>();
            if damage.is_none() && (m[13]>0. || m[12]>0.) {damage=Some(previous_hp-hp);}
            previous_hp=hp;
            if impact.is_none() && m[13]>0. {impact=Some((m[10]-initial_clock).round() as i64+origin_tick);}
            if spell_impact.is_none() && m[12]>0. {spell_impact=Some((m[11]-initial_clock).round() as i64+origin_tick);}
        }
        let actual_impact=impact.or(spell_impact);let expected_release=case["first_projectile_tick"].as_i64();let expected_impact=case["first_damage_tick"].as_i64();
        let release=releases.first().copied();let mut expected_releases=vec![];let mut tracking=false;
        for f in frames.iter().filter(|f|f["team"]==3){let active=!f["projectiles"].as_array().unwrap().is_empty();
            if active && !tracking {expected_releases.push(f["tick"].as_i64().unwrap());}tracking=active;}
        let expected_damage=expected_impact.and_then(|tick|{
            let hp=|t|frames.iter().find(|f|f["team"]==3 && f["tick"].as_i64()==Some(t)).and_then(|f|f["health"].as_f64());
            Some((hp(tick-1)?-hp(tick)?) as f32)
        });
        let spell=case["orders"].as_array().unwrap().iter().filter_map(|o|o["action"].as_u64()).find(|a|(6..=9).contains(a)).unwrap_or(3);
        if (6..=8).contains(&spell) {if let Some(tick)=expected_impact {
            if tick>=raze_until {raze_stacks=0.;}raze_stacks+=1.;raze_until=tick+(world.engine.spells[(spell-6) as usize][0][7]*30.).round() as i64;
        }}
        let agrees=|a:Option<i64>,b:Option<i64>|match(a,b){(Some(a),Some(b))=>(a-b).abs()<=1,(None,None)=>true,_=>false};
        let releases_agree=releases.len()==expected_releases.len() && releases.iter().zip(&expected_releases).all(|(a,b)|(a-b).abs()<=1);
        let damage_agrees=match(damage,expected_damage){(Some(a),Some(b))=>(a-b).abs()<=if spell>=6 {1.2}else{5.},(None,None)=>true,_=>false};
        let mut facing_phases=vec![];
        if check_facing {for phase in -1..=1 {let mut maximum=0f64;let mut samples=0;
            for frame in frames.iter().filter(|f|f["team"]==2 && f["tick"].as_i64().is_some_and(|t|t>=first_tick && t<=first_tick+20)) {
                if let Some((_,angle))=gpu_facing.iter().find(|v|v.0==frame["tick"].as_i64().unwrap()+phase){
                    maximum=maximum.max(((angle-frame["facing"].as_f64().unwrap()+180.).rem_euclid(360.)-180.).abs());samples+=1;
                }
            }
            facing_phases.push(serde_json::json!({"phase_ticks":phase,"samples":samples,"max_error_degrees":maximum,"passed":samples>=20 && maximum<=15.}));
        }}
        let facing_agrees=!check_facing || facing_phases.iter().any(|p|p["passed"]==true);
        results.push(serde_json::json!({"name":case["name"],"release_expected":expected_release,"release_gpu":release,
            "all_releases_expected":expected_releases,"all_releases_gpu":releases,"damage_expected":expected_damage,"damage_gpu":damage,
            "impact_expected":expected_impact,"impact_gpu":actual_impact,"facing_phases":facing_phases,"passed":agrees(release,expected_release)&&agrees(actual_impact,expected_impact)&&releases_agree&&damage_agrees&&facing_agrees}));
    }
    let passed=results.iter().all(|v|v["passed"]==true);
    Ok(serde_json::json!({"passed":passed,"tolerance_ticks":1,"reference":reference,"scenarios":results,
        "scope":"attack cycles, first-impact damage, raze stacks and cancellations; does not certify the full environment"}))
}

