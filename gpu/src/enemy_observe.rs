//! ggd-v4 public spell observations. Never reads enemy cooldowns.
use burn::tensor::{Tensor,DType};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use crate::{contract::*,observe::{self,Observed},world::*,runtime::{argument,primitive,tensor}};

pub const FIELDS:usize=16;
pub const OFFSET:usize=crate::timing_observe::WIDTH;
pub const WIDTH:usize=OFFSET+4*FIELDS;
const MEMORY:usize=8; // previous spell, onset, last observation, four onsets, episode

pub fn columns()->Vec<i32>{(OFFSET..WIDTH).map(|v|v as i32).collect()}

/// Keep P08's Experience module untouched; its public storage supports any width.
pub fn experience(actors:usize,d:&burn::tensor::Device)->crate::experience::Experience {
    use crate::experience::{Experience,HORIZON,SIGNALS};
    Experience{observations:(0..actors).step_by(32).map(|a|primitive(Tensor::<3>::zeros([32.min(actors-a),HORIZON,WIDTH],(d,DType::F16)))).collect(),
        signals:primitive(Tensor::<3>::zeros([actors,HORIZON,SIGNALS],d)),actors}
}

#[cube]
fn put(out:&mut [f32],at:usize,f:usize,value:f32,scale:f32){out[at+f]=value/scale;out[at+FIELDS+f]=1.0f32;}

#[cube(launch)]
fn enrich(state:&[f32],meta:&[f32],memory:&mut [f32],out:&mut [f32],actors:usize,pregame:f32){
    let a=ABSOLUTE_POS;
    if a<actors {
        let m=a/2;let time=(meta[m*META]+pregame*30.0f32)/30.0f32;
        for slot in 0..2 {
            let unit=(a%2+slot)%2;let p=(m*UNITS+unit)*STRIDE;
            let at=a*WIDTH+OFFSET+slot*2*FIELDS;let mem=(a*2+slot)*MEMORY;
            if memory[mem+7]!=meta[m*META+1] {
                for f in 0..MEMORY {memory[mem+f]=-1.0f32;}
                memory[mem+7]=meta[m*META+1];
            }
            let visible=out[a*WIDTH+observe::HERO_OFFSET+slot*2*UNIT_COUNT+UNIT_COUNT+UNIT_HEALTH]>0.0f32;
            if !visible {memory[mem]=-1.0f32;memory[mem+1]=-1.0f32;memory[mem+2]=-1.0f32;}
            else {
                if memory[mem+2]>=0.0f32 && time-memory[mem+2]>0.2f32 {memory[mem]=-1.0f32;memory[mem+1]=-1.0f32;}
                let spell=if state[p+CAST_END]>time {state[p+CAST]}else{0.0f32};
                for i in 0..5 {put(out,at,i,if spell==i as f32{1.0f32}else{0.0f32},1.0f32);}
                if spell!=memory[mem] {
                    memory[mem+1]=-1.0f32;
                    if spell>0.0f32 && memory[mem]>=0.0f32 {memory[mem+1]=time;memory[mem+2+spell as usize]=time;}
                }
                if spell==0.0f32 {put(out,at,5,0.0f32,1.0f32);}
                else if memory[mem+1]>=0.0f32 {put(out,at,5,((time-memory[mem+1])/if spell==4.0f32{1.67f32}else{0.55f32}).clamp(0.0f32,1.0f32),1.0f32);}
                memory[mem]=spell;memory[mem+2]=time;
                for i in 0..4 {
                    if memory[mem+3+i]>=0.0f32 {put(out,at,6+i,(time-memory[mem+3+i]).clamp(0.0f32,60.0f32),60.0f32);}
                    put(out,at,10+i,crate::physics::spell_level(state[p+UNIT_LEVEL],i)as f32,4.0f32);
                }
                put(out,at,14,state[p+UNIT_RAZE_STACKS],16.0f32);
                put(out,at,15,(state[p+STACK_END]-time).max(0.0f32),10.0f32);
            }
        }
    }
}

pub struct Observer {memory:CubeTensor}
impl Observer {
    pub fn new(w:&World)->Self {Self{memory:primitive(Tensor::<1>::full([w.matches*4*MEMORY],-1.,&w.device))}}
    pub fn extend(&self,w:&World,obs:Observed)->Observed {
        assert_eq!(obs.packed.dims()[1],OFFSET);
        let n=w.matches*2;let packed=primitive(Tensor::cat(vec![obs.packed.cast(DType::F32),Tensor::zeros([n,WIDTH-OFFSET],&w.device)],1));
        enrich::launch(&w.state.client,CubeCount::Static(n.div_ceil(32)as u32,1,1),CubeDim::new_1d(32),argument(&w.state),argument(&w.meta),argument(&self.memory),argument(&packed),n,w.engine.pre_game_seconds);
        Observed{packed:tensor(packed),refs:obs.refs,reset:obs.reset}
    }
}

/// Replay measured public events through the production observer, without a policy.
/// This checks encoding and causal memory; it does not claim trajectory equivalence.
pub fn check(engine:&std::path::Path,fixture:&std::path::Path,output:&std::path::Path)->anyhow::Result<serde_json::Value>{
    use burn::tensor::TensorData;
    let rows:Vec<serde_json::Value>=serde_json::from_slice(&std::fs::read(fixture)?)?;
    anyhow::ensure!(!rows.is_empty(),"empty public event fixture");
    let d=crate::runtime::device();let mut w=World::load(engine,1,&d)?;let observer=Observer::new(&w);
    let mut max=0f64;let mut fields=0usize;let mut mismatched_known=0usize;
    for row in &rows {
        let time=row["time"].as_f64().unwrap()as f32;
        let mut state=vec![0f32;UNITS*STRIDE];let mut meta=vec![0f32;META];meta[0]=(time-w.engine.pre_game_seconds)*30.;
        let mut packed=vec![0f32;2*OFFSET];
        for u in 0..2 {
            let r=&row["units"][u];let p=u*STRIDE;
            let visible=r["visible"].as_bool().unwrap();let spell=r["spell"].as_f64().unwrap()as f32;
            state[p+UNIT_LEVEL]=7.;state[p+CAST]=spell;state[p+CAST_END]=if spell>0.{time+1.}else{0.};
            state[p+UNIT_RAZE_STACKS]=r["stacks"].as_f64().unwrap()as f32;state[p+STACK_END]=time+r["remaining"].as_f64().unwrap()as f32;
            packed[observe::HERO_OFFSET+u*2*UNIT_COUNT+UNIT_COUNT+UNIT_HEALTH]=if visible{1.}else{0.};
        }
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
        w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
        let obs=Observed{packed:Tensor::from_data(TensorData::new(packed,[2,OFFSET]),&d),refs:primitive(Tensor::<1>::zeros([2*observe::TARGETS],&d)),reset:Tensor::zeros([2],&d)};
        let values=observer.extend(&w,obs).packed.into_data().try_to_vec::<f32>()?;
        if row["compare"]==false {continue;}
        let expected=row["expected"].as_array().unwrap();
        for (i,e) in expected.iter().enumerate(){
            let error=(values[OFFSET+i]as f64-e.as_f64().unwrap()).abs();max=max.max(error);fields+=1;
            if i%(2*FIELDS)>=FIELDS && error>0. {mismatched_known+=1;}
        }
    }
    let mut actual_casts=vec![];
    for spell in 1..=4 {
        let mut world=World::load(engine,1,&d)?;let rng=crate::sampling::Sampler::new(1,20261103,&d);
        let mut s=tensor::<1>(world.state.clone()).into_data().try_to_vec::<f32>()?;
        for u in 2..UNITS {s[u*STRIDE+ACTIVE]=0.;}
        for u in 0..2 {let p=u*STRIDE;s[p+UNIT_LEVEL]=7.;s[p+UNIT_HEALTH]=1000.;s[p+UNIT_MANA]=1000.;s[p+UNIT_X]=80.+if u==1{[200.,450.,700.,300.][spell-1]}else{0.};s[p+UNIT_Y]=-816.;s[p+FACING]=if u==1{std::f32::consts::PI}else{0.};s[p+ATTACK_AIM]=s[p+FACING];}
        world.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));world.meta=primitive(Tensor::<1>::zeros([META],&d));
        let enemy=Observer::new(&world);let combat=crate::combat_observe::CombatObserver::new(&world);let teacher=crate::scripted_teacher::Teacher::new(&world);
        let read=|| {let o=world.observe(observe::initial_previous(2,&d));enemy.extend(&world,crate::timing_observe::extend(&world,combat.enrich(&world,&o),&teacher,true,false)).packed.into_data().try_to_vec::<f32>().unwrap()};
        let _=read();let o=world.observe(observe::initial_previous(2,&d));
        let actions=Tensor::<2,burn::tensor::Int>::from_data([[0,0,0,0,64,64],[spell as i32+5,0,0,0,64,64]],&d);let _=world.advance(actions,&o.refs,&rng);
        let mut samples=vec![];
        for tick in 0..75 {let _=world.step_ticks_with_waves(&rng,1,false);let values=read();samples.push(serde_json::json!({"tick":tick,"values":&values[OFFSET..WIDTH]}));}
        actual_casts.push(serde_json::json!({"spell":spell,"samples":samples}));
    }
    let result=serde_json::json!({"frames":rows.len(),"compared_frames":fields/(4*FIELDS),"fields":fields,"known_mismatches":mismatched_known,"max_abs_error":max,"tolerance":0.0001,"passed":mismatched_known==0&&max<=0.0001,"scope":"Recorded Dota Bot API events through production GPU public observer; actual simulator cast traces included separately","actual_casts":actual_casts});
    crate::artifacts::atomic_json(output,&result)?;anyhow::ensure!(result["passed"]==true,"public observation parity failed: {result}");Ok(result)
}
