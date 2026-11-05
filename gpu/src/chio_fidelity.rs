//! P15 movement differential. Host readback is restricted to this diagnostic.
use anyhow::{Context,Result,ensure};
use burn::tensor::{Tensor,TensorData};
use crate::{runtime::{device,primitive,tensor},world::*,contract::*};
use std::path::Path;

pub fn check(engine:&Path,reference:&Path,output:&Path)->Result<serde_json::Value> {
    let capture:serde_json::Value=serde_json::from_slice(&std::fs::read(reference)?)?;
    ensure!(capture["complete"]==true,"incomplete native capture");
    let frames=capture["frames"].as_array().context("missing frames")?;
    let orders=capture["orders"].as_array().context("missing orders")?;
    let d=device();let mut w=World::load(engine,1,&d)?;
    let initial=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>()?;
    let mut results=vec![];
    for case in capture["cases"].as_array().context("missing cases")? {
        let id=&case["case"];
        let rows:Vec<_>=frames.iter().filter(|f|&f["case"]==id).collect();
        let commands:Vec<_>=orders.iter().filter(|f|&f["case"]==id).collect();
        ensure!(commands.len()==21,"case {id}: expected 21 orders, got {}",commands.len());
        ensure!(commands.windows(2).all(|v|v[1]["tick"].as_i64().unwrap()-v[0]["tick"].as_i64().unwrap()==6),"decision cadence is not six ticks");
        ensure!(commands.iter().all(|o|o["tick"].as_i64().unwrap()-o["observation_tick"].as_i64().unwrap()==7),"command latency is not seven ticks");
        let first=commands[0]["tick"].as_i64().unwrap();
        let origin=*rows.iter().find(|f|f["tick"].as_i64()==Some(first-1)).context("missing pre-command frame")?;
        let mut s=initial.clone();
        for u in 0..UNITS {s[u*STRIDE+ACTIVE]=0.;}
        let actor=&origin["units"][0];
        for (field,key) in [(UNIT_X,"x"),(UNIT_Y,"y"),(UNIT_Z,"z"),(UNIT_MOVE_SPEED,"speed")] {s[field]=actor[key].as_f64().context("actor field missing")?as f32;}
        s[FACING]=actor["facing"].as_f64().unwrap()as f32*std::f32::consts::PI/180.;
        s[ACTIVE]=1.;s[UNIT_HEALTH]=1000.;s[KIND]=0.;
        let mut dest=(s[UNIT_X],s[UNIT_Y]);let mut samples=vec![];
        for f in rows.iter().filter(|f|f["tick"].as_i64().unwrap()>=first && f["case_tick"].as_i64().unwrap()<174) {
            let tick=f["tick"].as_i64().unwrap();
            // Measured colliders isolate direct locomotion from creep AI error.
            for (u,native) in f["units"].as_array().unwrap().iter().enumerate().skip(1) {
                let p=u*STRIDE;s[p+ACTIVE]=1.;s[p+UNIT_HEALTH]=1000.;s[p+KIND]=native["kind"].as_f64().unwrap()as f32;
                s[p+UNIT_X]=native["x"].as_f64().unwrap()as f32;s[p+UNIT_Y]=native["y"].as_f64().unwrap()as f32;
            }
            w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
            // Native frame at t precedes Bot API order at t; first movement is t+1.
            if tick>first {w.direct_fixture_step(dest.0,dest.1);}
            s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>()?;
            let native=&f["units"][0];let nx=native["x"].as_f64().unwrap();let ny=native["y"].as_f64().unwrap();
            let error=((s[UNIT_X]as f64-nx).powi(2)+(s[UNIT_Y]as f64-ny).powi(2)).sqrt();
            let nz=native["z"].as_f64().unwrap();let error_z=(s[UNIT_Z]as f64-nz).abs();
            samples.push(serde_json::json!({"tick":tick,"case_tick":f["case_tick"],"dota":[nx,ny],"sim":[s[UNIT_X],s[UNIT_Y]],"error":error,
                "dota_z":nz,"sim_z":s[UNIT_Z],"error_z":error_z,"error_3d":(error*error+error_z*error_z).sqrt()}));
            if let Some(order)=commands.iter().find(|o|o["tick"].as_i64()==Some(tick)) {
                dest=(order["x"].as_f64().unwrap()as f32,order["y"].as_f64().unwrap()as f32);
            }
        }
        let max=samples.iter().map(|x|x["error"].as_f64().unwrap()).fold(0f64,f64::max);
        let mean=samples.iter().map(|x|x["error"].as_f64().unwrap()).sum::<f64>()/samples.len()as f64;
        results.push(serde_json::json!({"case":id,"name":case["name"],"repeat":case["repeat_index"],"max_error":max,"mean_error":mean,"samples":samples}));
    }
    std::fs::write(output,serde_json::to_vec(&serde_json::json!({"engine_sha256":w.fingerprint,"reference":reference,"cases":results,
        "scope":"Production direct movement with native collider trajectories; fixed tick alignment, no fitted phase shift"}))?)?;
    Ok(serde_json::json!({"output":output,"cases":results.len()}))
}
