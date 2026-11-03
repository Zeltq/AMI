//! Optional ggd-v3 extension. No changes to simulation or teacher decisions.
use burn::tensor::Tensor;
use cubecl::prelude::*;
use crate::{contract::*,observe::{self,Observed},world::*,runtime::{argument,primitive,tensor}};

pub const FIELDS:usize=11;
pub const OFFSET:usize=observe::GGD_WIDTH;
pub const WIDTH:usize=OFFSET+(2+observe::CREEPS+4)*2*FIELDS;

pub fn columns(creeps:usize)->Vec<i32> {
    (0..2+creeps).chain(2+observe::CREEPS..2+observe::CREEPS+4)
        .flat_map(|slot|(OFFSET+slot*2*FIELDS..OFFSET+(slot+1)*2*FIELDS).map(|v|v as i32)).collect()
}

/// Dota attack animation rates, in cycles per base attack-speed second.
/// Measured separately for Radiant/Dire models; see reports/P06/curves.json.
#[cube]
fn animation_rate(kind:f32,team:f32)->f32 {
    if kind==0.0f32 {1.0f32/1.1f32}
    else if kind==1.0f32 {1.035952f32}
    else if kind==2.0f32 && team==1.0f32 {15.0f32/14.0f32}
    else if kind==3.0f32 && team==1.0f32 {30.0f32/29.0f32}
    else {1.0f32}
}

#[cube]
fn attack_cycle(kind:f32,team:f32,interval:f32,elapsed:f32)->f32 {
    let base=if kind==0.0f32{1.6f32}else if kind==1.0f32{0.9f32}else{1.0f32};
    ((elapsed-1.0f32/30.0f32).max(0.0f32)*animation_rate(kind,team)*base/interval.max(0.001f32)).min(1.0f32)
}

#[cube]
fn put(out:&mut [f32],at:usize,f:usize,value:f32,scale:f32){
    out[at+f]=value/scale;out[at+FIELDS+f]=1.0f32;
}

#[cube(launch)]
fn enrich(state:&mut [f32],meta:&[f32],refs:&[f32],forecasts:&[f32],out:&mut [f32],actors:usize,pregame:f32,#[comptime] honest:bool,#[comptime] oracle:bool){
    let a=ABSOLUTE_POS;
    if a<actors {
        let row=a*WIDTH;let m=a/2;let base=m*UNITS*STRIDE;
        let time=((meta[m*META]+pregame*30.0f32)/30.0f32).max(0.0f32);
        for slot in 0..2+observe::CREEPS+4 {
            let unit=if slot<2 {(a%2+slot)%2}else if slot<2+observe::CREEPS {refs[a*observe::TARGETS+slot]as usize}else{slot-2-observe::CREEPS+2};
            let obs=row+if slot<2 {observe::HERO_OFFSET+slot*2*UNIT_COUNT}else if slot<2+observe::CREEPS {observe::CREEP_OFFSET+(slot-2)*2*UNIT_COUNT}else{observe::TOWER_OFFSET+(slot-2-observe::CREEPS)*2*UNIT_COUNT};
            let at=row+OFFSET+slot*2*FIELDS;
            // Cached tower HP can remain known in fog; weapon state cannot.
            let visible=out[obs+UNIT_COUNT+if slot>=2+observe::CREEPS{UNIT_IS_ATTACKING}else{UNIT_HEALTH}]>0.0f32;
            if honest {out[obs+UNIT_ATTACK_CYCLE]=0.0f32;out[obs+UNIT_COUNT+UNIT_ATTACK_CYCLE]=0.0f32;}
            if unit<UNITS && visible {
                let p=base+unit*STRIDE;let kind=state[p+KIND];
                let base_interval=if kind==0.0f32 {1.6f32}else if kind==1.0f32 {0.9f32}else if kind==4.0f32 {3.0f32}else{1.0f32};
                let interval=state[p+UNIT_ATTACK_INTERVAL];
                let point=state[p+UNIT_ATTACK_POINT]*interval/base_interval;
                let target=crate::physics::observed_attack_target(state,base,p,time);
                let attacking=state[p+WINDUP]>time || (target>=0.0f32 && state[p+ATTACK_READY]>time);
                if honest {
                    if kind<4.0f32 {
                        put(out,at,0,if attacking{1.0f32}else{0.0f32},1.0f32);
                        let mut ready=0.0f32.runtime();
                        if state[p+ATTACK_READY]>0.0f32 {
                            let release=((state[p+ATTACK_READY]-interval+point)*30.0f32-0.0001f32).ceil()/30.0f32;
                            put(out,at,1,(time-release).clamp(0.0f32,3.0f32),3.0f32);
                            ready=(release+interval-point-time).clamp(0.0f32,3.0f32);
                        }
                        put(out,at,2,ready,3.0f32);
                    }
                    // Idle/walk, siege and T2 attack cycles were not calibrated.
                    if attacking && kind<4.0f32 && (kind!=1.0f32 || unit==2 || unit==4) {
                        let start=if unit<2 {meta[m*META+META_ATTACK_ANIMATION_START+unit]}
                            else if state[p+WINDUP]>time {state[p+WINDUP]-point}else{state[p+ATTACK_READY]-interval};
                        out[obs+UNIT_ATTACK_CYCLE]=attack_cycle(kind,state[p+TEAM],interval,time-start);
                        out[obs+UNIT_COUNT+UNIT_ATTACK_CYCLE]=1.0f32;
                    }
                }
                if oracle {
                    put(out,at,3,(state[p+ATTACK_READY]-time).max(0.0f32),3.0f32);
                    put(out,at,4,(state[p+WINDUP]-time).max(0.0f32),3.0f32);
                    if unit==a%2 {put(out,at,10,meta[m*META+META_ATTACK_POINT_ERROR+unit],1.0f32);}
                    let f=(a*UNITS+unit)*10;
                    if unit>=6 && unit<CREEP_END && forecasts[f]==state[p+GENERATION] && (forecasts[f+1]-time).abs()<0.0001f32 {
                        put(out,at,5,forecasts[f+2]-time,3.0f32);
                        put(out,at,6,forecasts[f+3],2000.0f32);
                        put(out,at,7,forecasts[f+4],200.0f32);
                        put(out,at,8,forecasts[f+5],200.0f32);
                        put(out,at,9,forecasts[f+7]-time,3.0f32);
                    }
                }
            }
        }
    }
}

#[cube(launch)]
fn curves(input:&[f32],out:&mut [f32],n:usize){
    let i=ABSOLUTE_POS;if i<n{out[i]=attack_cycle(input[i*4],input[i*4+1],input[i*4+2],input[i*4+3]);}
}

pub fn check(engine:&std::path::Path,fixture:&std::path::Path,output:&std::path::Path)->anyhow::Result<serde_json::Value>{
    use burn::tensor::{TensorData,DType};
    let rows:Vec<serde_json::Value>=serde_json::from_slice(&std::fs::read(fixture)?)?;
    anyhow::ensure!(!rows.is_empty(),"empty curve fixture");
    let d=crate::runtime::device();let n=rows.len();
    let input:Vec<f32>=rows.iter().flat_map(|r|["kind","team","interval","elapsed"].map(|k|r[k].as_f64().unwrap()as f32)).collect();
    let input=primitive(Tensor::<1>::from_data(TensorData::new(input,[n*4]),&d));let out=primitive(Tensor::<1>::zeros([n],&d));
    curves::launch(&out.client,CubeCount::Static(n.div_ceil(256)as u32,1,1),CubeDim::new_1d(256),argument(&input),argument(&out),n);
    let values=tensor::<1>(out).into_data().try_to_vec::<f32>()?;
    let mut groups=std::collections::BTreeMap::<String,(usize,f64,f64)>::new();
    for (r,v) in rows.iter().zip(&values){let e=(*v as f64-r["cycle"].as_f64().unwrap()).abs();let g=groups.entry(r["group"].as_str().unwrap().into()).or_default();g.0+=1;g.1=g.1.max(e);g.2+=e;}
    let curves:Vec<_>=groups.into_iter().map(|(g,(n,max,sum))|serde_json::json!({"group":g,"samples":n,"max_abs_error":max,"mean_abs_error":sum/n as f64})).collect();
    let mut w=World::load(engine,1,&d)?;let base=w.observe(observe::initial_previous(2,&d));
    let combat=crate::combat_observe::CombatObserver::new(&w).enrich(&w,&base);
    let teacher=crate::scripted_teacher::Teacher::new(&w);let _=teacher.actions(&w,&base);
    let initial=extend(&w,Observed{packed:combat.packed.clone(),refs:combat.refs.clone(),reset:combat.reset.clone()},&teacher,true,false).packed.into_data().try_to_vec::<f32>()?;
    let mut initial_known=0;
    for a in 0..2 {for slot in 0..2+observe::CREEPS+4 {let at=a*WIDTH+OFFSET+slot*2*FIELDS;
        anyhow::ensure!(initial[at]==0.,"zero windup is not an active attack at episode start");
        initial_known+=usize::from(initial[at+FIELDS]>0.);
    }}
    let empty=extend(&w,Observed{packed:combat.packed.clone(),refs:combat.refs.clone(),reset:combat.reset.clone()},&teacher,false,false);
    let old=combat.packed.cast(DType::F16).cast(DType::F32).into_data().try_to_vec::<f32>()?;
    let new=empty.packed.cast(DType::F16).cast(DType::F32).into_data().try_to_vec::<f32>()?;
    let mut equal=0;let mut zeros=0;
    for a in 0..2 {for f in 0..observe::GGD_WIDTH {equal+=usize::from(old[a*observe::GGD_WIDTH+f].to_bits()==new[a*WIDTH+f].to_bits());}for f in OFFSET..WIDTH {zeros+=usize::from(new[a*WIDTH+f]==0.);}}
    anyhow::ensure!(equal==old.len()&&zeros==2*(WIDTH-OFFSET),"zero ablation changed legacy input");
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>()?;
    state[ATTACK_READY]=1.7;state[UNIT_ATTACK_INTERVAL]=1.2;state[UNIT_ATTACK_POINT]=0.5;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
    let mut meta=tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>()?;meta[0]=(1.5-w.engine.pre_game_seconds)*30.;
    w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
    let base=w.observe(observe::initial_previous(2,&d));
    let combat=crate::combat_observe::CombatObserver::new(&w).enrich(&w,&base);
    let base_values=combat.packed.clone().into_data().try_to_vec::<f32>()?;
    let public=extend(&w,combat,&teacher,true,true).packed.into_data().try_to_vec::<f32>()?;
    let ready=public[OFFSET+2]*3.;let hidden=public[OFFSET+3]*3.;let since=public[OFFSET+1]*3.;
    anyhow::ensure!((ready-0.225).abs()<0.0001 && (since-0.6).abs()<0.0001 && (hidden-0.2).abs()<0.0001,"public release rounding differs from Dota adapter formula");
    let mut fog_count=0;
    for a in 0..2 {for t in 0..4 {
        let at=a*observe::GGD_WIDTH+observe::TOWER_OFFSET+t*2*UNIT_COUNT;
        if base_values[at+UNIT_COUNT+UNIT_IS_ATTACKING]==0. {
            let at=a*WIDTH+OFFSET+(2+observe::CREEPS+t)*2*FIELDS;
            anyhow::ensure!(public[at..at+2*FIELDS].iter().all(|v|*v==0.),"hidden tower timing leaked");fog_count+=1;
        }
    }}
    anyhow::ensure!(fog_count>0,"fixture has no hidden towers");
    let result=serde_json::json!({"device":"production CubeCL GPU kernel","curves":curves,"legacy_prefix_equal":{"numerator":equal,"denominator":old.len()},"extension_zero":{"numerator":zeros,"denominator":2*(WIDTH-OFFSET)},"initial_inactive":{"numerator":initial_known,"denominator":initial_known},"public_clock_check":{"ready":ready,"oracle_ready":hidden,"since_attack":since,"passed":true},"hidden_towers_masked":fog_count,"fixture":fixture});
    crate::artifacts::atomic_json(output,&result)?;Ok(result)
}

pub fn extend(w:&World,obs:Observed,teacher:&crate::scripted_teacher::Teacher,honest:bool,oracle:bool)->Observed {
    let n=w.matches*2;
    assert_eq!(obs.packed.dims()[1],observe::GGD_WIDTH);
    let packed=primitive(Tensor::cat(vec![obs.packed.cast(burn::tensor::DType::F32),Tensor::zeros([n,WIDTH-OFFSET],&w.device)],1));
    enrich::launch(&w.state.client,CubeCount::Static(n.div_ceil(32)as u32,1,1),CubeDim::new_1d(32),
        argument(&w.state),argument(&w.meta),argument(&obs.refs),argument(&teacher.forecasts),argument(&packed),n,w.engine.pre_game_seconds,honest,oracle);
    Observed{packed:tensor(packed),refs:obs.refs,reset:obs.reset}
}

// Diagnostic trace for the first four matches. It never feeds the policy.
const TRACE_WIDTH:usize=17;
pub struct Trace {buffer:burn_cubecl::tensor::CubeTensor,actors:usize,file:std::io::BufWriter<std::fs::File>}
impl Trace {
    pub fn new(w:&World,root:&std::path::Path)->anyhow::Result<Self>{
        let actors=(w.matches*2).min(8);
        crate::artifacts::atomic_json(&root.join("timing-trace-header.json"),&serde_json::json!({"actors":actors,"columns":["time","action","target","own_hp","ready_in","windup_in","order_target","lh","deny","target_hp","target_team","forecast_hp","hit_in","damage_min","damage_max","death_in","forecast_known"]}))?;
        Ok(Self{buffer:primitive(Tensor::<3>::zeros([128,actors,TRACE_WIDTH],&w.device)),actors,file:std::io::BufWriter::new(std::fs::File::create(root.join("timing-trace.jsonl"))?)})
    }
    pub fn record(&self,w:&World,t:&crate::scripted_teacher::Teacher,refs:&burn_cubecl::tensor::CubeTensor,actions:burn::tensor::Tensor<2,burn::tensor::Int>,step:usize){
        let actions=primitive(actions.float());
        trace::launch(&w.state.client,CubeCount::Static(1,1,1),CubeDim::new_1d(32),argument(&w.state),argument(&w.meta),argument(refs),argument(&actions),argument(&t.forecasts),argument(&self.buffer),self.actors,step,w.engine.pre_game_seconds);
    }
    pub fn flush(&mut self,steps:usize)->anyhow::Result<()>{
        use std::io::Write;
        let values=tensor::<3>(self.buffer.clone()).slice([0..steps,0..self.actors,0..TRACE_WIDTH]).into_data().try_to_vec::<f32>()?;
        for (i,row) in values.chunks_exact(TRACE_WIDTH).enumerate(){serde_json::to_writer(&mut self.file,&serde_json::json!({"actor":i%self.actors,"v":row}))?;self.file.write_all(b"\n")?;}
        self.file.flush()?;Ok(())
    }
}

#[cube(launch)]
fn trace(state:&[f32],meta:&[f32],refs:&[f32],actions:&[f32],forecasts:&[f32],out:&mut [f32],actors:usize,step:usize,pregame:f32){
    let a=ABSOLUTE_POS;if a<actors {
        let base=a/2*UNITS*STRIDE;let p=base+a%2*STRIDE;let at=(step*actors+a)*TRACE_WIDTH;let time=((meta[a/2*META]+pregame*30.0f32)/30.0f32).max(0.0f32);
        for f in 0..TRACE_WIDTH{out[at+f]=0.0f32;}
        let target=if actions[a*6]==3.0f32{refs[a*observe::TARGETS+actions[a*6+3]as usize]}else{state[p+TARGET]};
        out[at]=meta[a/2*META]/30.0f32;out[at+1]=actions[a*6];out[at+2]=target;out[at+3]=state[p+UNIT_HEALTH];
        out[at+4]=(state[p+ATTACK_READY]-time).max(0.0f32);out[at+5]=(state[p+WINDUP]-time).max(0.0f32);out[at+6]=state[p+TARGET];
        out[at+7]=state[p+UNIT_LAST_HITS];out[at+8]=state[p+UNIT_DENIES];
        if target>=0.0f32 && target<UNITS as f32 {
            let u=target as usize;let q=base+u*STRIDE;let f=(a*UNITS+u)*10;
            out[at+9]=state[q+UNIT_HEALTH];out[at+10]=state[q+TEAM];
            if (forecasts[f+1]-time).abs()<0.0001f32 && forecasts[f]==state[q+GENERATION] && u>=6 {
                out[at+11]=forecasts[f+3];out[at+12]=forecasts[f+2]-time;out[at+13]=forecasts[f+4];out[at+14]=forecasts[f+5];out[at+15]=forecasts[f+7]-time;out[at+16]=1.0f32;
            }
        }
    }
}
