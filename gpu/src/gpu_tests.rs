use burn::tensor::{Tensor,TensorData,Int};
use crate::{runtime::{device,primitive,tensor},world::*,contract::*,observe::{initial_previous,TERRAIN_OFFSET,WIDTH},sampling::Sampler};
use std::sync::Mutex;
static GPU:Mutex<()>=Mutex::new(());
use cubecl::frontend::CompilationArg;

#[test]
fn p15_first_pregame_death_terminates_chio_without_changing_legacy_rule(){
    let _g=GPU.lock().unwrap();
    for chio in [false,true] {
        let mut w=arena();w.engine.chio_direct=chio;w.engine.decision_ticks=if chio{6}else{4};let d=w.device.clone();
        let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[UNIT_HEALTH]=1.;s[DAMAGE]=100.;s[CREDIT]=6.;s[CREDIT_TEAM]=1.;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let mut meta=vec![0f32;META];meta[0]=-30.;
        w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));let rng=Sampler::new(1,23,&d);
        let _=w.step_ticks_with_waves(&rng,1,false);let meta=tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(meta[0],-29.);assert_eq!(meta[2],if chio{1.}else{0.});assert_eq!(meta[17],if chio{-1.}else{0.});
    }
}

#[test]
fn p15_reward_events_capture_simultaneous_purchase_use_and_flask_damage(){
    use crate::{chio_reward as r,chio_events as e,chio_actions as c};
    let _g=GPU.lock().unwrap();let mut w=arena();w.engine.chio_direct=true;w.engine.decision_ticks=6;let d=w.device.clone();let rng=Sampler::new(1,21,&d);
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[ITEM_START+4]=1.;s[UNIT_GOLD]=1000.;s[UNIT_MANA]=100.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let before=r::snapshot(&w);let obs=w.observe(initial_previous(2,&d));let mask=c::observe(&w,&obs);
    let _=c::enqueue(&w,&obs,&mask,Tensor::<2,Int>::from_data([[11,0,0,0],[1,0,0,0]],&d),Tensor::<1>::from_data([3f32,0.],&d));
    let _=w.step_ticks_with_waves(&rng,8,false);let after=r::snapshot(&w);
    let events=tensor::<1>(w.chio_events.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(events[e::MANGO],1.);assert_eq!(events[e::PURCHASES+4],1.);assert_eq!(events[e::SPENT],w.engine.items[4][0]);
    assert!((events[e::GOLD]-8./30.*100./60.).abs()<1e-6);
    let snapshot=tensor::<2>(after.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(snapshot[17],1.);
    let reward=r::transition(&before,&after,false,&d);let parts=reward.components.into_data().try_to_vec::<f32>().unwrap();
    assert!((parts[1]-0.006*events[e::GOLD]).abs()<1e-6);assert!((parts[10]+0.1).abs()<1e-6);
    // A real ordinary hero projectile interrupts Flask. Preserve native stats.
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[UNIT_HEALTH]=300.;s[FLASK_END]=w.engine.pre_game_seconds+8./30.+8.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s.clone(),[UNITS*STRIDE]),&d));
    let mut shots=vec![0f32;SHOTS*SHOT_FIELDS];shots[0]=1.;shots[1]=s[UNIT_X];shots[2]=s[UNIT_Y];shots[3]=0.;shots[4]=s[GENERATION];shots[5]=1200.;shots[6]=100.;shots[7]=1.;shots[8]=1.;
    w.shots=primitive(Tensor::<1>::from_data(TensorData::new(shots,[SHOTS*SHOT_FIELDS]),&d));
    let before=r::snapshot(&w);let _=w.step_ticks_with_waves(&rng,1,false);let after=r::snapshot(&w);
    let snap=tensor::<2>(after.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(snap[25],1.);assert_eq!(snap[19],0.);assert!(snap[28]>snap[30]);assert!((snap[31]-8.).abs()<1e-4);
    let reward=r::transition(&before,&after,true,&d);let parts=reward.components.clone().into_data().try_to_vec::<f32>().unwrap();assert!(parts[12]<0. && parts[12]>=-0.400001);
    let path=data_root().join("runs/p15/reward-unit");std::fs::create_dir_all(&path).unwrap();
    let mut recorder=r::Recorder::new(&w,true);recorder.store(&before,&after,&reward,Tensor::ones([2],&d),0);recorder.flush(&path,0,0,1).unwrap();
}

fn chio_arena()->World {
    let mut w=arena();w.engine.chio_direct=true;w.engine.decision_ticks=6;let d=w.device.clone();
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    s[UNIT_LEVEL]=7.;s[UNIT_MANA]=2000.;s[UNIT_MANA_MAX]=2000.;
    for item in [4,5,6] {s[ITEM_START+item]=1.;s[ITEM_READY+item]=0.;}
    for (u,team,kind,hp,maxhp,x,y) in [(6,1.,2.,100.,100.,0.,0.),(7,0.,2.,50.,100.,0.,100.),(4,1.,1.,1000.,1000.,300.,0.)] {
        let p=u*STRIDE;s[p+ACTIVE]=1.;s[p+TEAM]=team;s[p+KIND]=kind;s[p+UNIT_HEALTH]=hp;s[p+UNIT_HEALTH_MAX]=maxhp;
        s[p+UNIT_X]=x;s[p+UNIT_Y]=y;s[p+GENERATION]=3.;s[p+TARGET]=-1.;s[p+LANE_POSITION]=-1.;s[p+ATTACK_READY]=10000.;
    }
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    w.tree_count=3;w.trees=primitive(Tensor::<1>::from_data([-150f32,0.,0.,-190.,0.,0.,-100.,0.,0.],&d));
    w.tree_alive=primitive(Tensor::<1>::ones([3],&d));w
}

#[test]
fn p15_teacher_translation_handles_thresholds_targets_items_and_pending_orders(){
    use crate::{chio_actions as ca,chio_teacher::Translator};
    let _g=GPU.lock().unwrap();let mut w=chio_arena();let d=w.device.clone();let translator=Translator::new(&w);
    let translate=|w:&World,raw:[i32;6]| {
        let obs=w.observe(initial_previous(2,&d));let mask=ca::observe(w,&obs);
        translator.labels(w,&obs,&mask,Tensor::<2,Int>::from_data([raw,[0,0,0,0,64,64]],&d)).into_data().try_to_vec::<i32>().unwrap()[..4].to_vec()
    };
    for (raw,expected) in [
        ([1,0,0,0,68,64],[1,0,0,0]),([1,0,0,0,69,64],[0,0,0,0]),([1,0,0,0,64,69],[0,9,0,0]),
        ([3,0,0,2,64,64],[2,0,0,0]),([3,0,0,3,64,64],[3,0,0,1]),([3,0,0,1,64,64],[4,0,0,0]),
        ([3,0,0,260,64,64],[5,0,0,0]),([3,0,0,261,64,64],[1,0,0,0]),
        ([10,0,5,0,64,64],[10,0,0,0]),([10,0,4,0,64,64],[11,0,0,0]),([10,0,6,0,64,64],[12,0,0,0]),
        ([10,0,3,0,64,64],[1,0,0,0]),([10,0,7,0,64,64],[1,0,0,0]),([5,0,0,0,64,64],[1,0,0,0]),
        ([0,0,0,0,64,64],[1,0,0,0])
    ] {assert_eq!(translate(&w,raw),expected,"raw {raw:?}");}
    let mut queue=vec![0f32;COMMAND_WORDS];queue[0]=6.;queue[1]=3.;queue[4]=6.;queue[7]=1.;queue[8]=3.;
    w.commands=primitive(Tensor::<1>::from_data(TensorData::new(queue.clone(),[COMMAND_WORDS]),&d));
    assert_eq!(translate(&w,[0,0,0,0,64,64]),vec![2,0,0,0]);
    queue[8]=2.;w.commands=primitive(Tensor::<1>::from_data(TensorData::new(queue,[COMMAND_WORDS]),&d));
    assert_eq!(translate(&w,[0,0,0,0,64,64]),vec![1,0,0,0]);
    w.commands=primitive(Tensor::<1>::zeros([COMMAND_WORDS],&d));
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[CAST]=2.;s[CAST_END]=1000.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    assert_eq!(translate(&w,[0,0,0,0,64,64]),vec![7,0,0,0]);assert_eq!(translate(&w,[4,0,0,0,64,64]),vec![1,0,0,0]);
}

#[test]
fn p15_teacher_memory_reconciles_the_actual_direct_command_and_script_purchase(){
    use crate::{chio_actions as ca,chio_teacher::Translator};
    let _g=GPU.lock().unwrap();let w=chio_arena();let d=w.device.clone();let teacher=crate::scripted_teacher::Teacher::new(&w);
    let obs=w.observe(initial_previous(2,&d));let mask=ca::observe(&w,&obs);let before=teacher.command_snapshot();
    let raw=Tensor::<2,Int>::from_data([[1,4,0,0,100,64],[0,0,0,0,64,64]],&d);
    let label=Translator::new(&w).labels(&w,&obs,&mask,raw.clone());let executed=ca::enqueue(&w,&obs,&mask,label,Tensor::zeros([2],&d));
    teacher.reconcile_commands(&w,&obs,before,raw,executed);
    let memory=tensor::<1>(teacher.memory.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(memory[12],2.);
    let queue=tensor::<1>(w.commands.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(queue[1],2.);assert_eq!(queue[2],0.);
}

fn chio_observation(w:&World,history:&crate::chio_observe::Observer,previous:Tensor<2,Int>)->crate::observe::Observed {
    let base=w.observe(initial_previous(2,&w.device));let t=crate::scripted_teacher::Teacher::new(w);
    let combat=crate::combat_observe::CombatObserver::new(w);let enemy=crate::enemy_observe::Observer::new(w);
    let v4=enemy.extend(w,crate::timing_observe::extend(w,combat.enrich(w,&base),&t,true,false));
    let masks=crate::chio_actions::observe(w,&base);history.extend(w,&base,v4,previous,&masks)
}

#[test]
fn p15_hp_history_is_causal_idempotent_and_uses_twelve_decisions(){
    use crate::chio_observe as c;
    let _g=GPU.lock().unwrap();let mut w=chio_arena();let d=w.device.clone();let history=c::Observer::new(&w);
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[UNIT_HEALTH_MAX]=1000.;
    for step in 0..14 {
        s[UNIT_HEALTH]=500.+step as f32*10.;w.state=primitive(Tensor::<1>::from_data(TensorData::new(s.clone(),[UNITS*STRIDE]),&d));
        let mut meta=vec![0f32;META];meta[0]=(step*6)as f32;meta[6]=step as f32;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
        let obs=chio_observation(&w,&history,c::initial_previous(2,&d));let data=obs.packed.clone().into_data().try_to_vec::<f32>().unwrap();
        for lag in 0..12 {
            let known=lag<=step;assert_eq!(data[c::HERO_OFFSET+c::HERO_FIELDS+lag],if known{1.}else{0.});
            let expected=if known{(500.+(step-lag)as f32*10.)/1000.}else{0.};assert!((data[c::HERO_OFFSET+lag]-expected).abs()<1e-6);
        }
        let again=chio_observation(&w,&history,c::initial_previous(2,&d)).packed.into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(&data[c::HERO_OFFSET..c::WIDTH],&again[c::HERO_OFFSET..c::WIDTH]);
    }
}

#[test]
fn p15_hp_history_hides_fog_and_resets_reused_unit_generations(){
    use crate::chio_observe as c;
    let _g=GPU.lock().unwrap();let mut w=chio_arena();let d=w.device.clone();let history=c::Observer::new(&w);
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[STRIDE+UNIT_HEALTH_MAX]=1000.;s[STRIDE+UNIT_HEALTH]=800.;
    for step in 0..5 {
        s[STRIDE+UNIT_X]=if step==0 || step==4 {200.}else{6000.};s[STRIDE+UNIT_Y]=if step==0 || step==4 {0.}else{6000.};
        if step>0 {s[STRIDE+UNIT_HEALTH]=if step==4{500.}else{123.+step as f32};}
        if step==4 {s[6*STRIDE+GENERATION]+=1.;s[6*STRIDE+UNIT_HEALTH]=40.;}
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s.clone(),[UNITS*STRIDE]),&d));
        let mut meta=vec![0f32;META];meta[6]=step as f32;meta[0]=(step*6)as f32;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
        let data=chio_observation(&w,&history,c::initial_previous(2,&d)).packed.into_data().try_to_vec::<f32>().unwrap();
        let at=c::HERO_OFFSET+2*c::HERO_FIELDS;
        if step>0 && step<4 {assert_eq!(data[at],0.);assert_eq!(data[at+c::HERO_FIELDS],0.);}
        if step==4 {
            assert_eq!(data[at],0.5);assert!((data[at+4]-0.8).abs()<1e-6);
            assert_eq!(&data[at+c::HERO_FIELDS..at+c::HERO_FIELDS+5],&[1.,0.,0.,0.,1.]);
            let creep=c::CREEP_OFFSET;assert!((data[creep]-0.4).abs()<1e-6);
            assert!(data[creep+c::UNIT_FIELDS+1..creep+c::UNIT_FIELDS+12].iter().all(|v|*v==0.));
        }
    }
}

#[test]
fn p15_geometry_previous_cast_and_six_item_inputs_follow_contract(){
    use crate::chio_observe as c;
    let _g=GPU.lock().unwrap();let mut w=chio_arena();let d=w.device.clone();let history=c::Observer::new(&w);
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[UNIT_X]=0.;s[UNIT_Y]=0.;s[FACING]=0.;
    s[STRIDE+UNIT_X]=450.;s[STRIDE+UNIT_Y]=0.;s[STRIDE+FACING]=std::f32::consts::PI;
    s[CAST]=3.;s[CAST_END]=1000.;s[UNIT_MANA]=0.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let mut meta=vec![0f32;META];meta[6]=1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
    let previous=Tensor::<2,Int>::from_data([[0,9,0,0],[4,0,0,0]],&d);
    let obs=chio_observation(&w,&history,previous);let data=obs.packed.clone().into_data().try_to_vec::<f32>().unwrap();
    let enemy=c::HERO_OFFSET+2*c::HERO_FIELDS;
    assert_eq!(&data[enemy+12..enemy+15],&[1.,1.,1.]);assert_eq!(data[enemy+15],1.);assert_eq!(data[enemy+16],0.);
    assert_eq!(&data[c::HERO_OFFSET+17..c::HERO_OFFSET+20],&[1.,1.,1.]);
    assert_eq!(&data[c::HERO_OFFSET+20..c::HERO_OFFSET+25],&[1.,0.,0.,1.,0.]);
    assert_eq!(data[c::PREVIOUS_OFFSET],1.);assert_eq!(data[c::PREVIOUS_OFFSET+13+9],1.);assert_eq!(data[c::PREVIOUS_OFFSET+49],1.);
    let (b,_)=c::decode(&obs,crate::observe::CREEPS,crate::observe::TREES,crate::observe::PROJECTILES);
    assert_eq!(b.items.dims(),[2,6,16]);assert_eq!(b.enemy_items.dims(),[2,6,16]);assert_eq!(b.previous.dims(),[2,c::PREVIOUS]);
    let before=b.items.into_data().try_to_vec::<f32>().unwrap();
    let mut changed=data;for actor in 0..2 {for item in [3,7] {for f in 0..16 {changed[actor*c::WIDTH+crate::observe::ITEM_OFFSET+item*16+f]=1234.;}}}
    let obs=crate::observe::Observed{packed:Tensor::<2>::from_data(TensorData::new(changed,[2,c::WIDTH]),&d),refs:obs.refs,reset:obs.reset};
    let (b,_)=c::decode(&obs,crate::observe::CREEPS,crate::observe::TREES,crate::observe::PROJECTILES);
    assert_eq!(b.items.into_data().try_to_vec::<f32>().unwrap(),before);
}

#[test]
fn p15_common_economy_thresholds_level_transition_and_delivery_counts(){
    let _g=GPU.lock().unwrap();let mut w=chio_arena();let d=w.device.clone();
    let original=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    // level, missing HP, Flask, Tango charges, Mango, mana, max mana, buy
    let cases=[
        (4.,0.,0.,0.,0.,500.,500.,4.),(4.,300.,1.,0.,0.,500.,500.,1.),
        (4.,301.,1.,0.,0.,500.,500.,4.),(4.,12.,0.,1.,0.,500.,500.,1.),
        (4.,13.,0.,1.,0.,500.,500.,4.),(4.,701.,2.,0.,0.,500.,500.,1.),
        (5.,701.,2.,0.,0.,500.,500.,4.),(5.,900.,3.,0.,0.,500.,500.,1.),
        (4.,0.,1.,0.,2.,140.,500.,1.),(4.,0.,1.,0.,2.,139.,500.,3.),
        (4.,0.,0.,3.,0.,250.,250.,1.),(4.,0.,1.,0.,0.,449.,500.,1.),
        (5.,0.,1.,0.,0.,449.,500.,3.),(4.,0.,1.,0.,3.,0.,500.,1.),
        (5.,0.,1.,0.,3.,0.,500.,3.),(4.,100.,0.,3.,0.,500.,500.,1.),
    ];
    for (i,(level,missing,flask,tango,mango,mana,max_mana,expected)) in cases.into_iter().enumerate() {
        let mut s=original.clone();s[UNIT_LEVEL]=level;s[UNIT_HEALTH_MAX]=1000.;s[UNIT_HEALTH]=1000.-missing;
        s[UNIT_MANA_MAX]=max_mana;s[UNIT_MANA]=mana;s[UNIT_GOLD]=10000.;s[ITEM_START+4]=mango;s[ITEM_START+5]=flask;s[ITEM_START+6]=tango;
        // Owned quantities must work in all four storage locations.
        let location=[ITEM_START,BACKPACK_START,STASH_START,COURIER_START][i%4];
        if location!=ITEM_START {for item in [4,5,6] {s[location+item]=s[ITEM_START+item];s[ITEM_START+item]=0.;}}
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
        let economy=crate::chio_economy::Economy::new(&w);let obs=w.observe(initial_previous(2,&d));
        assert_eq!(economy.choose(&w,&obs).into_data().try_to_vec::<f32>().unwrap()[0],expected,"case {i}");
    }
}

#[test]
fn p15_common_economy_pregame_once_pending_purchase_and_remaining_heal(){
    let _g=GPU.lock().unwrap();let mut w=chio_arena();let d=w.device.clone();
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[UNIT_GOLD]=10000.;
    for item in [4,5,6] {s[ITEM_START+item]=0.;}
    let mut meta=vec![0f32;META];meta[0]=-30.;
    w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta.clone(),[META]),&d));
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s.clone(),[UNITS*STRIDE]),&d));
    let economy=crate::chio_economy::Economy::new(&w);
    let buy=|w:&World| {let obs=w.observe(initial_previous(2,&d));economy.choose(w,&obs).into_data().try_to_vec::<f32>().unwrap()[0]};
    assert_eq!(buy(&w),5.);
    let mut q=vec![0f32;COMMAND_WORDS];q[2]=5.;q[7]=1.;w.commands=primitive(Tensor::<1>::from_data(TensorData::new(q,[COMMAND_WORDS]),&d));
    assert_eq!(buy(&w),0.);
    w.commands=primitive(Tensor::<1>::zeros([COMMAND_WORDS],&d));s[STASH_START+6]=3.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s.clone(),[UNITS*STRIDE]),&d));let _=buy(&w);
    s[STASH_START+6]=0.;w.state=primitive(Tensor::<1>::from_data(TensorData::new(s.clone(),[UNITS*STRIDE]),&d));assert_eq!(buy(&w),4.);
    meta[1]=1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));assert_eq!(buy(&w),5.);
    // Ongoing healing reduces the need for a second Flask.
    s[ITEM_START+5]=1.;s[UNIT_HEALTH_MAX]=1000.;s[UNIT_HEALTH]=500.;s[FLASK_END]=w.engine.pre_game_seconds+8.;
    w.meta=primitive(Tensor::<1>::zeros([META],&d));w.state=primitive(Tensor::<1>::from_data(TensorData::new(s.clone(),[UNITS*STRIDE]),&d));assert_eq!(buy(&w),1.);
    s[FLASK_END]=0.;w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));assert_eq!(buy(&w),4.);
}

#[test]
fn p15_all_thirteen_actions_map_to_the_delayed_executor(){
    use crate::chio_actions as c;
    let _g=GPU.lock().unwrap();let mut w=chio_arena();let n=2;
    for kind in 0..13 {
        w.commands=primitive(Tensor::<1>::zeros([COMMAND_WORDS],&w.device));
        let obs=w.observe(initial_previous(n,&w.device));let masks=c::observe(&w,&obs);
        let mask=masks.types.clone().int().into_data().try_to_vec::<i32>().unwrap();assert!(mask[kind]!=0,"type {kind} unexpectedly unavailable");
        let chosen=Tensor::<2,Int>::from_data([[kind as i32,9,0,1],[1,0,0,0]],&w.device);
        let _=c::enqueue(&w,&obs,&masks,chosen,Tensor::zeros([n],&w.device));
        let q=tensor::<1>(w.commands.clone()).into_data().try_to_vec::<f32>().unwrap();
        let expected=if kind==0{2.}else if kind==1{4.}else if kind<=5{3.}else if kind<=9{kind as f32}else{10.};
        assert_eq!(q[0],6.);assert_eq!(q[1],expected);assert_eq!(q[7],1.);assert_eq!(q[2],0.);
        if kind==0 {assert!((q[5]+200.).abs()<0.001);assert!((q[6]-2000.).abs()<0.001);}
        if kind==2 {assert_eq!(q[4],6.);assert_eq!(q[8],3.);}
        if kind==3 {assert_eq!(q[4],7.);assert_eq!(q[8],3.);}
        if kind==4 {assert_eq!(q[4],1.);}
        if kind==5 {assert_eq!(q[4],4.);}
        if kind==10 {assert_eq!(q[3],5.);assert_eq!(q[4],0.);}
        if kind==11 {assert_eq!(q[3],4.);assert_eq!(q[4],0.);}
        if kind==12 {assert_eq!(q[3],6.);assert_eq!(q[4],(UNITS+1)as f32);}
    }
    assert_eq!(tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap()[7],0.);
}

#[test]
fn p15_masks_cover_deny_visibility_items_cast_and_pending_continuation(){
    use crate::chio_actions as c;
    let _g=GPU.lock().unwrap();let mut w=chio_arena();let d=w.device.clone();
    let get=|w:&World| {let obs=w.observe(initial_previous(2,&w.device));c::observe(w,&obs).types.int().into_data().try_to_vec::<i32>().unwrap().into_iter().map(|v|v!=0).collect::<Vec<_>>()};
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    s[7*STRIDE+UNIT_HEALTH]=50.01;s[FLASK_END]=1000.;s[ITEM_READY+4]=1000.;s[ITEM_START+6]=0.;
    s[STRIDE+ACTIVE]=0.;s[4*STRIDE+ACTIVE]=0.;s[6*STRIDE+ACTIVE]=0.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s.clone(),[UNITS*STRIDE]),&d));
    let mask=get(&w);for kind in [2,3,4,5,10,11,12] {assert!(!mask[kind],"type {kind} mask leaked");}
    for spell in 1..=4 {
        s[CAST]=spell as f32;s[CAST_END]=1000.;s[UNIT_MANA]=0.;s[COOLDOWNS+spell-1]=1000.;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s.clone(),[UNITS*STRIDE]),&d));
        let mask=get(&w);let enabled:Vec<_>=mask[..13].iter().enumerate().filter_map(|(i,&v)|v.then_some(i)).collect();
        assert_eq!(enabled,vec![1,5+spell]);
    }
    // A first cast in flight closes the mask before the simulator starts it.
    s[CAST]=0.;s[CAST_END]=0.;s[UNIT_MANA]=2000.;for spell in 0..4{s[COOLDOWNS+spell]=0.;}
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s.clone(),[UNITS*STRIDE]),&d));
    let mut q=vec![0f32;COMMAND_WORDS];q[0]=6.;q[1]=6.;q[7]=1.;
    w.commands=primitive(Tensor::<1>::from_data(TensorData::new(q,[COMMAND_WORDS]),&d));
    let mask=get(&w);assert_eq!(mask[..13].iter().filter(|&&v|v).count(),2);assert!(mask[1]&&mask[6]);
    // After completion an already queued continuation must not keep all heads closed.
    s[COOLDOWNS]=1000.;w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let mask=get(&w);assert!(mask[0]);assert!(!mask[6]);
}

#[test]
fn p15_repeated_attack_orders_preserve_windup_for_all_four_target_types(){
    use crate::chio_actions as c;
    let _g=GPU.lock().unwrap();
    for kind in 2..=5 {
        let mut w=chio_arena();let d=w.device.clone();let rng=Sampler::new(1,17,&d);
        let target=match kind{2=>6,3=>7,4=>1,_=>4};
        let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        if target>=2 {s[target*STRIDE+UNIT_HEALTH]=10000.;s[target*STRIDE+UNIT_HEALTH_MAX]=20000.;}
        let initial_hp=s[target*STRIDE+UNIT_HEALTH];
        s[target*STRIDE+UNIT_X]=100.;s[target*STRIDE+UNIT_Y]=0.;s[target*STRIDE+UNIT_HEALTH_REGEN]=0.;
        // Exclude interference while retaining the selected actor and target.
        for u in 1..UNITS {if u!=target{s[u*STRIDE+ACTIVE]=0.;}}
        s[target*STRIDE+UNIT_ATTACK_DAMAGE]=0.;s[target*STRIDE+UNIT_MOVE_SPEED]=0.;s[target*STRIDE+ATTACK_READY]=10000.;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
        let mut released=false;
        for step in 0..8 {
            let obs=w.observe(initial_previous(2,&d));let mask=c::observe(&w,&obs);
            let allowed=mask.types.clone().int().into_data().try_to_vec::<i32>().unwrap();
            assert!(allowed[kind as usize]>0,"attack type {kind} invalid at step {step}: {allowed:?}");
            // One creep row remains after excluding unrelated entities.
            let a=Tensor::<2,Int>::from_data([[kind,0,0,0],[1,0,0,0]],&d);
            let _=c::enqueue(&w,&obs,&mask,a,Tensor::zeros([2],&d));let _=w.step_ticks_with_waves(&rng,6,false);
            let state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();released|=state[target*STRIDE+UNIT_HEALTH]<initial_hp-1.;
        }
        assert!(released,"type {kind} never released an attack under six-tick repetition");
        assert_eq!(tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap()[7],0.,"attack type {kind}");
    }
}

#[test]
fn p15_decision_clock_queue_and_recurrent_reset_are_profile_scoped(){
    let _g=GPU.lock().unwrap();
    for ticks in [4usize,6] {
        let mut w=arena();w.engine.chio_direct=ticks==6;w.engine.decision_ticks=ticks;
        let profile=if ticks==6{"chio-v1"}else{"ggd-v4"};assert!(w.validate_profile(profile).is_ok());
        assert!(w.validate_profile(if ticks==6{"ggd-v4"}else{"chio-v1"}).is_err());
        assert_eq!(w.decision_hz(),if ticks==6{5.}else{7.5});
        let rng=Sampler::new(1,20261103,&w.device);let obs=w.observe(initial_previous(2,&w.device));
        let actions=Tensor::<2,Int>::from_data([[1,0,0,0,127,64],[0,0,0,0,64,64]],&w.device);
        let _=w.advance(actions,&obs.refs,&rng);
        let meta=tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(meta[0],ticks as f32);assert_eq!(meta[6],1.);
        let queue=tensor::<1>(w.commands.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(queue[0],6.);assert_eq!(queue[7],1.);
        assert_eq!(tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap()[UNIT_X],-200.);
        let _=w.step_ticks_with_waves(&rng,7-ticks,false);
        assert_eq!(tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap()[ORDER],0.);
        let _=w.step_ticks_with_waves(&rng,1,false);
        assert_eq!(tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap()[ORDER],1.);
        let mut meta=vec![0f32;META];meta[0]=(127*ticks)as f32;meta[6]=127.;
        w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&w.device));
        assert_eq!(w.observe(initial_previous(2,&w.device)).reset.into_data().try_to_vec::<f32>().unwrap(),vec![0.,0.]);
        let obs=w.observe(initial_previous(2,&w.device));let t=crate::scripted_teacher::Teacher::new(&w);
        let actions=Tensor::<2,Int>::from_data([[4,0,0,0,64,64];2],&w.device);
        let _=w.advance_teacher(actions,&obs.refs,&rng,&t,false);
        assert_eq!(w.observe(initial_previous(2,&w.device)).reset.into_data().try_to_vec::<f32>().unwrap(),vec![1.,1.]);
        let meta=tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(meta[0],(128*ticks)as f32);
        let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[UNIT_HEALTH]=0.;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&w.device));
        let mut meta=meta;meta[6]=129.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&w.device));
        assert_eq!(w.observe(initial_previous(2,&w.device)).reset.into_data().try_to_vec::<f32>().unwrap(),vec![1.,0.]);
        assert_eq!(crate::trace::Trace::new(&w).header(&w,0)["decision_ticks"],ticks);
    }
}

#[test]
fn p15_directional_move_stops_at_bodies_without_sliding(){
    let _g=GPU.lock().unwrap();
    for kind in [0.,2.] {
        let mut w=arena();w.engine.chio_direct=true;
        let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        s[UNIT_X]=0.;s[FACING]=0.;s[STRIDE+UNIT_X]=90.;s[STRIDE+KIND]=kind;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&w.device));
        for _ in 0..90 {w.direct_fixture_step(2000.,0.);}
        let end=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(end[UNIT_Y],0.);assert_eq!(end[FACING],0.);
        assert!(end[UNIT_X]>0. && end[UNIT_X]<60.,"passed through body: {}",end[UNIT_X]);
        let x=end[UNIT_X];w.direct_fixture_step(2000.,0.);
        assert_eq!(tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap()[UNIT_X],x);
    }
}

#[test]
fn p15_directional_move_does_not_recover_out_of_blocked_cells(){
    let _g=GPU.lock().unwrap();let mut w=arena();w.engine.chio_direct=true;
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[UNIT_X]=0.;s[FACING]=0.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&w.device));
    w.geometry=primitive(Tensor::<1>::zeros([512*512*2],&w.device));
    for _ in 0..30 {w.direct_fixture_step(2000.,0.);}
    let end=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!((end[UNIT_X],end[UNIT_Y]),(0.,0.));
}

#[test]
fn p15_directional_turns_match_native_ticks_within_one_step(){
    let _g=GPU.lock().unwrap();
    let reference:serde_json::Value=serde_json::from_str(include_str!("../../tests/fixtures/chio-directional-turns.json")).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let mut w=arena();w.engine.chio_direct=true;
        let frames=case["frames"].as_array().unwrap();let start=&frames[0];
        let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        s[UNIT_X]=start[1].as_f64().unwrap()as f32;s[UNIT_Y]=start[2].as_f64().unwrap()as f32;
        s[FACING]=start[3].as_f64().unwrap()as f32*std::f32::consts::PI/180.;s[STRIDE+ACTIVE]=0.;s[UNIT_MOVE_SPEED]=305.;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&w.device));
        for f in frames.iter().skip(1) {
            w.direct_fixture_step(2080.,-816.);
            let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
            let error=((s[UNIT_X]as f64-f[1].as_f64().unwrap()).powi(2)+(s[UNIT_Y]as f64-f[2].as_f64().unwrap()).powi(2)).sqrt();
            assert!(error<=10.2,"{} tick {} error {error}",case["name"],f[0]);
        }
    }
}

#[test]
fn p15_flag_preserves_normal_walk_and_other_state_bitwise(){
    let _g=GPU.lock().unwrap();let a=arena();let mut b=arena();b.engine.chio_direct=true;
    let mut s=tensor::<1>(a.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    s[ORDER]=1.;s[DEST_X]=1200.;s[DEST_Y]=300.;
    // Place an enemy in the path so this also exercises normal path/body avoidance.
    let mut a=a;
    a.state=primitive(Tensor::<1>::from_data(TensorData::new(s.clone(),[UNITS*STRIDE]),&a.device));
    b.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&b.device));
    let ra=Sampler::new(1,20261103,&a.device);let rb=Sampler::new(1,20261103,&b.device);
    for _ in 0..20 {let _=a.step_ticks_with_waves(&ra,4,false);let _=b.step_ticks_with_waves(&rb,4,false);}
    for (left,right) in [(a.state,b.state),(a.meta,b.meta),(a.shots,b.shots),(a.rewards,b.rewards)] {
        assert_eq!(tensor::<1>(left).into_data().try_to_vec::<f32>().unwrap(),tensor::<1>(right).into_data().try_to_vec::<f32>().unwrap());
    }
}

#[cubecl::prelude::cube(launch)]
fn p09_events(state:&mut [f32],out:&mut [f32],kind:f32,spell:usize){
    crate::teacher_combat::damage(out,0,0,kind,17.0f32,5.0f32,spell);
    // Zero or overkill damage must not replace the last effective source.
    crate::teacher_combat::damage(out,0,0,1.0f32,0.0f32,5.0f32,0);
    crate::teacher_combat::death(out,state,0,0);
    crate::teacher_combat::finish(out,state,0,-1.0f32);
}

#[test]
fn p09_death_sources_distances_and_reward_snapshots(){
    use cubecl::prelude::{CubeCount,CubeDim};
    use crate::{runtime::argument,teacher_combat as c};
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    state[UNIT_X]=0.;state[UNIT_Y]=0.;state[2*STRIDE+UNIT_X]=3.;state[2*STRIDE+UNIT_Y]=4.;state[4*STRIDE+UNIT_X]=6.;state[4*STRIDE+UNIT_Y]=8.;state[CREDIT]=1.;
    state[UNIT_LAST_HITS]=13.;state[UNIT_DENIES]=7.;state[UNIT_LEVEL]=3.;state[XP]=29.;state[UNIT_SOULS]=11.;state[UNIT_RAZE_STACKS]=2.;state[STACK_END]=10000.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state.clone(),[UNITS*STRIDE]),&d));
    for (kind,spell,cause) in [(0.,0,0),(0.,1,1),(2.,0,2),(1.,0,3),(0.,4,4)] {
        let out=primitive(Tensor::<1>::zeros([2*c::FIELDS],&d));
        p09_events::launch(&w.state.client,CubeCount::Static(1,1,1),CubeDim::new_1d(1),argument(&w.state),argument(&out),kind,spell);
        let data=tensor::<1>(out).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(data[c::DEATH_COUNTS+cause],1.);assert_eq!(data[c::DEATH_OWN_DISTANCE+cause],5.);assert_eq!(data[c::DEATH_ENEMY_DISTANCE+cause],10.);assert_eq!(data[c::FIRST_SOURCE],(cause+1)as f32);assert_eq!(data[c::FIELDS+c::HERO_KILLS],1.);
        assert_eq!(data[c::OUTCOME_REASON],3.);assert_eq!(data[c::OUTCOME_RESULT],-1.);
    }
    assert_eq!(tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap(),state);
    let before=w.observe(initial_previous(2,&d)).packed.into_data().try_to_vec::<f32>().unwrap();
    // observe refreshes fog proxy units; snapshot after that intentional update.
    state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    let recorder=crate::demo_telemetry::Recorder::new(&w);recorder.record(&w,0,0);recorder.record(&w,0,1);
    let directory=std::env::temp_dir().join(format!("p09-telemetry-{}",std::process::id()));std::fs::create_dir_all(&directory).unwrap();recorder.flush(&directory,0,0,1).unwrap();
    use std::io::Read;let mut bytes=vec![];flate2::read::GzDecoder::new(std::fs::File::open(directory.join("telemetry-000000.f32.gz")).unwrap()).read_to_end(&mut bytes).unwrap();
    let start=4+u32::from_le_bytes(bytes[..4].try_into().unwrap())as usize;let data:Vec<_>=bytes[start..].chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect();
    assert_eq!(&data[1..3],&[13.,7.]);assert_eq!(&data[5..7],&[3.,29.]);assert_eq!(&data[8..10],&[11.,2.]);
    let xp=tensor::<1>(w.experience.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(data[7],29.+xp[0]+xp[1]);
    let fields=crate::demo_telemetry::FIELDS;
    assert_eq!(&data[..fields],&data[fields..2*fields]);
    for side in 0..2 {let p=side*STRIDE;let at=26+side*4;
        assert_eq!(&data[at..at+4],&[state[p+UNIT_HEALTH],state[p+UNIT_HEALTH_MAX],state[p+UNIT_MANA],state[p+UNIT_MANA_MAX]]);
    }
    assert_eq!(tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap(),state);
    assert_eq!(w.observe(initial_previous(2,&d)).packed.into_data().try_to_vec::<f32>().unwrap(),before);
    let mut expired=state.clone();expired[STACK_END]=0.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(expired,[UNITS*STRIDE]),&d));recorder.record(&w,0,0);recorder.flush(&directory,1,0,1).unwrap();
    let mut bytes=vec![];flate2::read::GzDecoder::new(std::fs::File::open(directory.join("telemetry-000001.f32.gz")).unwrap()).read_to_end(&mut bytes).unwrap();
    let start=4+u32::from_le_bytes(bytes[..4].try_into().unwrap())as usize;assert_eq!(f32::from_le_bytes(bytes[start+9*4..start+10*4].try_into().unwrap()),0.);
    std::fs::remove_file(directory.join("telemetry-000001.f32.gz")).unwrap();
    std::fs::remove_file(directory.join("telemetry-000000.f32.gz")).unwrap();std::fs::remove_dir(directory).unwrap();
}

#[test]
fn p07_uphill_impacts_use_seeded_rng_only_for_eligible_ranged_attacks(){
    let _g=GPU.lock().unwrap();
    let run=|enabled:bool,uphill:bool,kind:usize| {
        let mut w=arena();w.engine.uphill_miss=enabled;let d=w.device.clone();let rng=Sampler::new(1,7,&d);
        let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();let q=6*STRIDE;
        s[q+ACTIVE]=1.;s[q+UNIT_HEALTH]=100000.;s[q+UNIT_HEALTH_MAX]=100000.;s[q+UNIT_X]=200.;s[q+TEAM]=1.;s[q+KIND]=2.;s[q+ATTACK_READY]=10000.;s[q+TARGET]=-1.;s[q+LANE_POSITION]=-1.;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
        let mut geometry=vec![1f32;512*512*2];let cell=|x:f32|((256*512+((x+8192.)/32.)as usize)*2)as usize;
        geometry[cell(-200.)]=if uphill{4.}else{3.};geometry[cell(200.)]=3.;
        w.geometry=primitive(Tensor::<1>::from_data(TensorData::new(geometry,[512*512*2]),&d));
        let mut shots=vec![0f32;SHOTS*SHOT_FIELDS];
        for i in 0..240 {let k=i*SHOT_FIELDS;shots[k]=1.;shots[k+1]=200.;shots[k+3]=6.;shots[k+5]=1200.;shots[k+6]=100.;shots[k+9]=kind as f32;}
        w.shots=primitive(Tensor::<1>::from_data(TensorData::new(shots,[SHOTS*SHOT_FIELDS]),&d));
        let _=w.step_ticks_with_waves(&rng,1,false);
        let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        let bytes=rng.client.read_one(rng.state.clone()).unwrap();let seed=u32::from_le_bytes(bytes[..4].try_into().unwrap());
        (((100000.-s[q+UNIT_HEALTH])/(100.*w.engine.combat[kind*5+2])).round()as usize,seed)
    };
    let initial=7u32.wrapping_add(2654435761)|1;let mut seed=initial;let mut hits=0;
    for _ in 0..240 {seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;hits+=usize::from((seed>>8)as f32/16777216.>=0.25);}
    assert!(hits>150 && hits<210);
    assert_eq!(run(true,true,0),(hits,seed));assert_eq!(run(true,true,0),(hits,seed));
    assert_eq!(run(true,true,3),(hits,seed));assert_eq!(run(true,true,1),(hits,seed));
    for (enabled,uphill,kind) in [(false,true,0),(true,false,0),(true,false,1)] {assert_eq!(run(enabled,uphill,kind),(240,initial));}
}

#[test]
fn p07_actual_enemy_raze_updates_public_cast_clock_and_debuff(){
    let _g=GPU.lock().unwrap();let w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let observer=crate::enemy_observe::Observer::new(&w);let combat=crate::combat_observe::CombatObserver::new(&w);let teacher=crate::scripted_teacher::Teacher::new(&w);
    let read=|| {let obs=w.observe(initial_previous(2,&d));observer.extend(&w,crate::timing_observe::extend(&w,combat.enrich(&w,&obs),&teacher,true,false)).packed.into_data().try_to_vec::<f32>().unwrap()};
    let at=crate::enemy_observe::OFFSET+2*crate::enemy_observe::FIELDS;
    assert_eq!(read()[at],1.);
    let obs=w.observe(initial_previous(2,&d));let actions=Tensor::<2,Int>::from_data([[0,0,0,0,64,64],[7,0,0,0,64,64]],&d);let _=w.advance(actions,&obs.refs,&rng);
    let mut casting=0;let mut final_row=vec![];
    for _ in 0..30 {let _=w.step_ticks_with_waves(&rng,1,false);let r=read();if r[at+2]==1. {casting+=1;assert_eq!(r[at+crate::enemy_observe::FIELDS+7],1.);}final_row=r;}
    assert!(casting>=15 && casting<=18,"unexpected cast phase length: {casting}");
    assert_eq!(final_row[at],1.);assert!(final_row[at+7]>0.);
    assert_eq!(final_row[crate::enemy_observe::OFFSET+14],1./16.);
    assert!(final_row[crate::enemy_observe::OFFSET+15]>0.);
}

#[test]
fn dagger_reconciles_actual_attack_item_and_continue_without_touching_equal_commands(){
    let _g=GPU.lock().unwrap();let w=arena();let d=w.device.clone();
    let teacher=crate::scripted_teacher::Teacher::new(&w);let obs=w.observe(initial_previous(2,&d));
    let snapshot=teacher.command_snapshot();let labels=teacher.actions(&w,&obs);
    let memory=tensor::<1>(teacher.memory.clone()).into_data().try_to_vec::<f32>().unwrap();
    let forecasts=tensor::<1>(teacher.forecasts.clone()).into_data().try_to_vec::<f32>().unwrap();
    teacher.reconcile_commands(&w,&obs,snapshot,labels.clone(),labels.clone());
    assert_eq!(tensor::<1>(teacher.memory.clone()).into_data().try_to_vec::<f32>().unwrap(),memory);
    assert_eq!(tensor::<1>(teacher.forecasts.clone()).into_data().try_to_vec::<f32>().unwrap(),forecasts);
    let snapshot=teacher.command_snapshot();let labels=teacher.actions(&w,&obs);
    let executed=Tensor::<2,Int>::from_data([[3,0,0,1,64,64],[10,0,5,0,64,64]],&d);
    teacher.reconcile_commands(&w,&obs,snapshot,labels,executed);
    let after=tensor::<1>(teacher.memory.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(after[1],1.);assert!(after[3]>w.engine.pre_game_seconds);assert_eq!(after[12],3.);
    assert_eq!(after[16+1],-1.);assert!((after[16+14]-w.engine.pre_game_seconds-0.4).abs()<0.001);
    assert_eq!(after[15],memory[15],"observation-derived retreat hysteresis retained");
    let snapshot=teacher.command_snapshot();let labels=teacher.actions(&w,&obs);
    let executed=Tensor::<2,Int>::from_data([[0,0,0,0,64,64],[0,0,0,0,64,64]],&d);
    teacher.reconcile_commands(&w,&obs,snapshot,labels,executed);
    let continued=tensor::<1>(teacher.memory.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(&continued[1..6],&after[1..6],"CONTINUE retains the pending real attack");
    assert_eq!(continued[16+14],after[16+14],"CONTINUE retains the real item delay");
}

#[test]
fn diagnostic_forward_move_is_once_masked_and_only_current_radiant_after_zero(){
    let _g=GPU.lock().unwrap();let d=device();let mut w=World::load(&engine(),2,&d).unwrap();
    let mut meta=vec![0f32;2*META];meta[0]=-1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta.clone(),[2*META]),&d));
    let actions=||Tensor::<2>::from_data([[0f32,2.,3.,4.,64.,64.];4],&d);
    let learn=Tensor::<2>::from_data([[1.],[1.],[0.],[1.]],&d);
    let mut mask=vec![1f32;44];mask[1]=0.;
    let finished=primitive(Tensor::<2>::zeros([2,crate::evaluation::FIELDS],&d));let marks=primitive(Tensor::<2>::zeros([4,3],&d));
    let inject=|a,mask:Vec<f32>|crate::departure::inject_forward_once(&w,a,learn.clone(),Tensor::from_data(TensorData::new(mask,[4,11]),&d),&finished,&marks);
    let before=actions().into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(inject(actions(),mask.clone()).into_data().try_to_vec::<f32>().unwrap(),before);
    meta[0]=0.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[2*META]),&d));
    let inject=|a,mask:Vec<f32>|crate::departure::inject_forward_once(&w,a,learn.clone(),Tensor::from_data(TensorData::new(mask,[4,11]),&d),&finished,&marks);
    assert_eq!(inject(actions(),mask.clone()).into_data().try_to_vec::<f32>().unwrap(),before);
    mask[1]=1.;let mut expected=before.clone();expected[0]=1.;expected[4]=127.;expected[5]=128.;
    assert_eq!(inject(actions(),mask.clone()).into_data().try_to_vec::<f32>().unwrap(),expected);
    assert_eq!(inject(actions(),mask).into_data().try_to_vec::<f32>().unwrap(),before);
    assert_eq!(tensor::<2>(marks).into_data().try_to_vec::<f32>().unwrap(),vec![1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]);
}

fn data_root()->std::path::PathBuf {
    std::env::var_os("SF_DUEL_DATA_ROOT").map(std::path::PathBuf::from)
        .unwrap_or_else(||std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf())
}
fn engine()->std::path::PathBuf {data_root().join("runs/gpu/dota-calibration-003/engine.json")}
fn arena()->World {
    let d=device();let mut w=World::load(&engine(),1,&d).unwrap();
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    for u in 2..UNITS {state[u*STRIDE+ACTIVE]=0.;}
    for u in 0..2 {
        let p=u*STRIDE;state[p+UNIT_X]=if u==0{-200.}else{200.};state[p+UNIT_Y]=0.;state[p+UNIT_Z]=0.;
        state[p+FACING]=if u==0{0.}else{std::f32::consts::PI};
        state[p+ATTACK_AIM]=state[p+FACING];
    }
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
    w.meta=primitive(Tensor::<1>::zeros([META],&d));
    w.fow_blockers=primitive(Tensor::<1>::zeros([512*512],&d));w.geometry=primitive(Tensor::<1>::ones([512*512*2],&d));w.movement_geometry=None;w.tree_map=primitive(Tensor::<1>::zeros([512*512],&d));w
}
fn command(w:&World,rng:&Sampler,action:i32){
    action_command(w,rng,[action,0,0,1,64,64]);
}
fn action_command(w:&World,rng:&Sampler,action:[i32;6]){
    let obs=w.observe(initial_previous(2,&w.device));
    let a=Tensor::<2,Int>::from_data([action,[0,0,0,0,64,64]],&w.device);
    let _=w.advance(a,&obs.refs,rng);
}

#[test]
fn selfplay_metadata_weapon_phase_cannot_pollute_outcomes_before_a_terminal(){
    let _g=GPU.lock().unwrap();let w=arena();let rng=Sampler::new(1,7,&w.device);
    action_command(&w,&rng,[3,0,0,1,64,64]);let _=w.step_ticks(&rng,20);
    let m=tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert!(m[META_ATTACK_NOMINAL_END]>0.);
    assert!(m[18..32].iter().all(|x|*x==0.),"weapon phase is not a selfplay metric");
}

#[test]
fn selfplay_metadata_terminal_accumulation_and_episode_reset_keep_weapon_state_separate(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    for victim in [0usize,1]{
        let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        s[victim*STRIDE+UNIT_HEALTH]=1.;s[victim*STRIDE+DAMAGE]=2.;s[victim*STRIDE+CREDIT]=(1-victim)as f32;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
        let mut m=tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap();m[0]=1.;
        for i in META_ATTACK_POINT_ERROR..META{m[i]=0.25+(i-META_ATTACK_POINT_ERROR)as f32;}
        w.meta=primitive(Tensor::<1>::from_data(TensorData::new(m,[META]),&d));
        let(_,done)=w.step_ticks(&rng,1);assert_eq!(done.sum().into_scalar::<f32>(),2.);
        let before=tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap();
        let mut signals=vec![0.;2*crate::experience::SIGNALS];signals[9]=1.;signals[11]=1.;signals[crate::experience::SIGNALS+9]=1.;
        crate::selfplay::record_outcomes(&w,Tensor::<2>::from_data(TensorData::new(signals,[2,crate::experience::SIGNALS]),&d));
        let after=tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(&before[META_ATTACK_POINT_ERROR..],&after[META_ATTACK_POINT_ERROR..]);
        assert_eq!(after[18],if victim==0{0.}else{1.});assert_eq!(after[19],1.);assert_eq!(after[20],0.);
        let _=w.observe(initial_previous(2,&d));
        let reset=tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap();
        assert!(reset[META_ATTACK_POINT_ERROR..].iter().all(|x|*x==0.));
        assert_eq!(&reset[18..32],&after[18..32],"episode reset preserves accumulated results");
    }
}

#[test]
fn serving_memory_clock_crosses_native_tick_buckets_without_changing_training_resets(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();
    for(tick,counter,clock_reset,training_reset)in[(-100.,128.,0.,1.),(-98.,5.,1.,0.),(-94.,6.,0.,0.),(414.,9.,1.,0.),(-2490.,0.,1.,1.)]{
        let mut meta=vec![0.;META];meta[0]=tick;meta[6]=counter;
        w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta.clone(),[META]),&d));
        assert_eq!(w.observe_with_clock(initial_previous(2,&d),Some(-3.3)).reset.into_data().try_to_vec::<f32>().unwrap(),vec![clock_reset;2]);
        w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
        assert_eq!(w.observe(initial_previous(2,&d)).reset.into_data().try_to_vec::<f32>().unwrap(),vec![training_reset;2]);
    }
}

#[test]
fn damage_matches_native_integral_post_mitigation_events_on_both_sides(){
    let _g=GPU.lock().unwrap();let root=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let reference=root.join("tests/fixtures/native-damage-rounding.json");
    let captured:serde_json::Value=serde_json::from_slice(&std::fs::read(&reference).unwrap()).unwrap();
    let output=std::env::temp_dir().join(format!("sf-native-damage-{}.json",std::process::id()));
    crate::lane_fidelity::replay(&engine(),&reference,&output).unwrap();
    let simulated:serde_json::Value=serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    for native in captured["records"].as_array().unwrap(){
        let gpu=simulated["records"].as_array().unwrap().iter().find(|r|r["id"]==native["id"]).unwrap();
        for target in ["victim",if native["id"].as_str().unwrap().ends_with("-2"){"radiant"}else{"dire"}]{
            let event=native["damage_events"].as_array().unwrap().iter().find(|e|e["target"]==target).unwrap();
            let mut first=None;
            for pair in gpu["frames"].as_array().unwrap().windows(2){
                let hp=|f:&serde_json::Value|f["units"].as_array().unwrap().iter().find(|u|u["id"]==target).unwrap()["health"].as_f64().unwrap();
                let drop=hp(&pair[0])-hp(&pair[1]);if first.is_none()&&drop>2.{first=Some(drop);}
            }
            assert!((first.unwrap()-event["damage"].as_f64().unwrap()).abs()<0.1,"{} {target}",native["id"]);
        }
    }
    std::fs::remove_file(output).unwrap();
}

#[test]
fn measured_static_sight_is_phase_and_team_specific_and_keeps_dynamic_sight(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    for u in 0..UNITS{s[u*STRIDE+ACTIVE]=0.;}
    for u in 0..2{s[u*STRIDE+UNIT_X]=0.;s[u*STRIDE+UNIT_Y]=0.;}
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s.clone(),[UNITS*STRIDE]),&d));
    w.engine.extra_vision=serde_json::from_value(serde_json::json!([
        {"x":-896.,"y":-896.,"team":0.,"day":2000.,"night":2000.,"unobstructed":true},
        {"x":-896.,"y":-896.,"team":1.,"day":2000.,"night":2000.,"unobstructed":true}])).unwrap();
    w.extra_vision=primitive(Tensor::<1>::from_data([-896f32,-896.,0.,2000.,2000.,1.,-896.,-896.,1.,2000.,2000.,1.],&d));
    let index=(((8192.-896.)/32.)as usize)*512+((8192.-896.)/32.)as usize;
    let mut map=vec![0f32;512*512];map[index]=f32::from_bits((63u32<<6)|1|(1<<4));
    w.static_vision=primitive(Tensor::<1>::from_data(TensorData::new(map,[512*512]),&d));
    for(tick,expected)in[(-1.,[1.,0.]),(1.,[0.,1.]),(9001.,[0.,0.])]{
        let mut m=vec![0.;META];m[0]=tick;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(m,[META]),&d));
        let obs=w.observe(initial_previous(2,&d)).packed.into_data().try_to_vec::<f32>().unwrap();
        assert_eq!([obs[TERRAIN_OFFSET+128],obs[WIDTH+TERRAIN_OFFSET+128]],expected,"phase tick{tick}");
    }
    // A covered negative cell must still receive sight from a live allied creep.
    let p=6*STRIDE;s[p+ACTIVE]=1.;s[p+KIND]=2.;s[p+TEAM]=0.;s[p+UNIT_X]=-896.;s[p+UNIT_Y]=-896.;s[p+UNIT_HEALTH]=100.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let obs=w.observe(initial_previous(2,&d)).packed.into_data().try_to_vec::<f32>().unwrap();
    assert_eq!([obs[TERRAIN_OFFSET+128],obs[WIDTH+TERRAIN_OFFSET+128]],[1.,0.]);
    // Unsupported cells retain the old building-ray behavior.
    w.static_vision=primitive(Tensor::<1>::zeros([512*512],&d));
    let obs=w.observe(initial_previous(2,&d)).packed.into_data().try_to_vec::<f32>().unwrap();
    assert_eq!([obs[TERRAIN_OFFSET+128],obs[WIDTH+TERRAIN_OFFSET+128]],[1.,1.]);
}

#[test]
fn native_raze_boundaries_include_each_target_hull_on_both_sides(){
    let _g=GPU.lock().unwrap();
    let reference=data_root().join("runs/gpu/overnight-8h-20261102-001/raze-hull16-qualified.native.json");
    let output=std::env::temp_dir().join(format!("sf-raze-hull-{}.json",std::process::id()));
    crate::lane_fidelity::replay(&engine(),&reference,&output).unwrap();
    let n:serde_json::Value=serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let g:serde_json::Value=serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();std::fs::remove_file(output).unwrap();
    for(case,nr)in n["cases"].as_array().unwrap().iter().zip(n["records"].as_array().unwrap()){
        let target=&case["raze_probe"]["target"];let gr=g["records"].as_array().unwrap().iter().find(|r|r["id"]==case["id"]).unwrap();
        let last=|r:&serde_json::Value|r["frames"].as_array().unwrap().last().unwrap()["units"].as_array().unwrap().iter().find(|u|u["id"]==*target).unwrap()["raze_stacks"].as_f64().unwrap();
        assert_eq!(last(nr),last(gr),"native hullboundary{}",case["id"]);
    }
}

#[test]
fn controlled_fixtures_disable_waves_while_production_keeps_both_spawn_boundaries(){
    let _g=GPU.lock().unwrap();
    for boundary in [0.,900.]{for waves in [false,true]{
        let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
        let mut meta=vec![0.;META];meta[0]=boundary-1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
        if waves{let _=w.step_ticks(&rng,2);}else{let _=w.step_ticks_with_waves(&rng,2,false);}
        let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        let active=(6..CREEP_END).filter(|u|s[u*STRIDE+ACTIVE]>0.).count();
        assert_eq!(active,if waves{8}else{0},"boundary{boundary},waves{waves}");
    }}
}

#[test]
fn sf_projectile_profile_uses_actual_origin_and_optional_creation_delay(){
    let _g=GPU.lock().unwrap();
    for deferred in [false,true]{
        let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
        let profile=[1f32,if deferred{1.}else{0.},1.,2.,2.,0.,0.,0.,0.,100.,0.,1.,100.,0.,0.,100.,0.,1.,100.,0.];
        w.sf_projectile_animation=primitive(Tensor::<1>::from_data(profile,&d));
        let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[ORDER]=3.;s[TARGET]=1.;s[TARGET_GENERATION]=0.;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
        let mut launch=None;
        for _ in 0..30{
            let _=w.step_ticks(&rng,1);
            if tensor::<1>(w.state.clone()).slice([ATTACK_HAND..ATTACK_HAND+1]).into_scalar::<f32>()!=0.{launch=Some(tensor::<1>(w.shots.clone()).into_data().try_to_vec::<f32>().unwrap());break;}
        }
        let shots=launch.expect("SF did not release a projectile");let shot=shots.chunks_exact(SHOT_FIELDS).find(|s|s[0]==1.).unwrap();
        assert!((shot[1]-if deferred{-100.}else{-60.}).abs()<0.001,"creation delay{deferred}:x{}",shot[1]);
        assert!(shot[2].abs()<0.001);
    }
}

#[test]
fn fresh_attack_turn_matches_native_side_and_back_windup_start(){
    let _g=GPU.lock().unwrap();
    // Native attachment captures: 90deg command at12 starts attack sequence15.
    // The separately recorded180deg attack retains its four-tick readiness.
    for (angle,start)in[(std::f32::consts::FRAC_PI_2,3),(std::f32::consts::PI,4)]{
        let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
        let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[FACING]=angle;s[ATTACK_AIM]=angle;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
        let mut meta=vec![0.;META];meta[0]=1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
        let mut orders=vec![0.;COMMAND_WORDS];orders[1]=3.;orders[4]=1.;orders[7]=1.;orders[8]=0.;
        w.commands=primitive(Tensor::<1>::from_data(TensorData::new(orders,[COMMAND_WORDS]),&d));
        for tick in 1..=start{
            let _=w.step_ticks(&rng,1);let windup=tensor::<1>(w.state.clone()).slice([WINDUP..WINDUP+1]).into_scalar::<f32>();
            assert_eq!(windup>0.,tick==start,"angle{angle},tick{tick}");
        }
    }
}

#[test]
fn sf_windup_fraction_matches_thirteen_direct_native_event_pairs(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let expected:serde_json::Value=serde_json::from_str(include_str!("../../tests/fixtures/native-sf-windup-level6.json")).unwrap();
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[UNIT_LEVEL]=6.;s[FACING]=-std::f32::consts::FRAC_PI_2;s[ATTACK_AIM]=s[FACING];
    s[STRIDE+UNIT_X]=7000.;s[STRIDE+UNIT_Y]=7000.;let p=6*STRIDE;s[p..p+STRIDE].fill(0.);s[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[8]);
    s[p+KIND]=4.;s[p+ACTIVE]=1.;s[p+TEAM]=1.;s[p+UNIT_X]=184.;s[p+UNIT_Y]=0.;s[p+UNIT_Z]=0.;s[p+TARGET]=-1.;s[p+LANE_POSITION]=-1.;s[p+GENERATION]=1.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let mut meta=vec![0.;META];meta[0]=-1800.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
    let mut commands=vec![0.;COMMAND_WORDS];commands[0]=-1789.;commands[1]=3.;commands[4]=6.;commands[7]=1.;commands[8]=1.;
    w.commands=primitive(Tensor::<1>::from_data(TensorData::new(commands,[COMMAND_WORDS]),&d));
    let mut starts=vec![];let mut releases=vec![];let mut winding=false;let mut hand=0.;
    for tick in 1..=400{
        let _=w.step_ticks(&rng,1);let s=tensor::<1>(w.state.clone()).slice([0..STRIDE]).into_data().try_to_vec::<f32>().unwrap();
        if s[WINDUP]>0. && !winding{starts.push(tick);}winding=s[WINDUP]>0.;
        if s[ATTACK_HAND]!=hand{releases.push(tick);hand=s[ATTACK_HAND];}
    }
    let pairs=expected["events"].as_array().unwrap();assert_eq!(releases.len(),pairs.len());
    for (i,e)in pairs.iter().enumerate(){
        assert_eq!(starts[i],e["start"].as_u64().unwrap()as i32,"native start{i}");
        assert_eq!(releases[i],e["release"].as_u64().unwrap()as i32,"native release{i}");
    }
}

#[test]
fn episode_reset_clears_both_heroes_pending_attack_fraction(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    for mode in 0..3{
        let mut m=vec![0.;META];m[0]=-900.;m[2]=1.;m[META_ATTACK_POINT_ERROR..META_ATTACK_ANIMATION_START+2].fill(0.5);
        w.meta=primitive(Tensor::<1>::from_data(TensorData::new(m,[META]),&d));
        if mode==0{w.reset();}else if mode==1{let _=w.observe(initial_previous(2,&d));}else{let _=w.step_ticks(&rng,1);}
        let m=tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap();
        assert!(m[META_ATTACK_POINT_ERROR..META_ATTACK_ANIMATION_START+2].iter().all(|v|*v==0.),"reset mode{mode}");
    }
}

#[test]
fn native_cancellations_do_not_spend_or_clear_the_sf_windup_fraction(){
    let _g=GPU.lock().unwrap();
    let reference=data_root().join("runs/gpu/lane-conformance-001/windup-cancellation-001/native.json");
    let output=std::env::temp_dir().join(format!("sf-windup-cancellation-{}.json",std::process::id()));
    crate::lane_fidelity::replay(&engine(),&reference,&output).unwrap();
    let native:serde_json::Value=serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let gpu:serde_json::Value=serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();std::fs::remove_file(output).unwrap();
    for (case,record)in native["cases"].as_array().unwrap().iter().zip(native["records"].as_array().unwrap()){
        let actor=case["orders"][0]["actor"].as_str().unwrap();let mut start=0;let mut expected=vec![];
        for event in record["attack_events"].as_array().unwrap().iter().filter(|e|e["actor"]==actor){
            if event["event"]=="start"{start=event["tick"].as_i64().unwrap();}
            if event["event"]=="release"{expected.push(event["tick"].as_i64().unwrap()-start);}
        }
        let trace=gpu["records"].as_array().unwrap().iter().find(|r|r["id"]==case["id"]).unwrap();
        let mut winding=false;let mut hand=0.;let mut actual=vec![];
        for frame in trace["frames"].as_array().unwrap(){
            let u=frame["units"].as_array().unwrap().iter().find(|u|u["id"]==actor).unwrap();let active=u["windup"].as_f64().unwrap()>0.;
            if active && !winding{start=frame["tick"].as_i64().unwrap();}winding=active;
            let next=u["attack_hand"].as_f64().unwrap();if next!=hand{actual.push(frame["tick"].as_i64().unwrap()-start);hand=next;}
        }
        assert_eq!(actual,expected,"native cancelled/completed windup durations for{}",case["id"]);
    }
}

#[test]
fn mixed_previous_actions_match_native_contract_after_fp16(){
    let _g=GPU.lock().unwrap();let d=device();
    // Generated by the production Python native encoder, including variable
    // targets, both sides and non-grid coordinates. No GPU-kernel copy oracle.
    let fixture:serde_json::Value=serde_json::from_str(include_str!("../../tests/fixtures/gpu-previous-actions.json")).unwrap();
    let cases=fixture["cases"].as_array().unwrap();let n=cases.len();
    let mut packed=vec![0f32;n*crate::observe::WIDTH];let mut state=vec![0f32;n/2*UNITS*STRIDE];
    let mut actions=vec![];let mut expected=vec![];
    for (actor,case)in cases.iter().enumerate(){
        for entry in case["packed_nonzero"].as_array().unwrap(){packed[actor*crate::observe::WIDTH+entry[0].as_u64().unwrap()as usize]=entry[1].as_f64().unwrap()as f32;}
        let p=(actor/2*UNITS+actor%2)*STRIDE;
        for axis in 0..2{state[p+axis]=case["own_xy"][axis].as_f64().unwrap()as f32;}
        actions.extend(case["action"].as_array().unwrap().iter().map(|x|x.as_i64().unwrap()as i32));
        expected.extend(case["expected"].as_array().unwrap().iter().map(|x|x.as_f64().unwrap()as f32));
    }
    let bases:Vec<f32>=fixture["bases"].as_array().unwrap().iter().map(|x|x.as_f64().unwrap()as f32).collect();
    let bases=primitive(Tensor::<1>::from_data(TensorData::new(bases,[6]),&d));
    let state=primitive(Tensor::<1>::from_data(TensorData::new(state,[n/2*UNITS*STRIDE]),&d));
    let observed=crate::observe::Observed{packed:Tensor::<2>::from_data(TensorData::new(packed,[n,crate::observe::WIDTH]),&d).cast(burn::tensor::DType::F16),refs:bases.clone(),reset:Tensor::zeros([n],&d)};
    let actual=observed.previous(Tensor::<2,Int>::from_data(TensorData::new(actions,[n,6]),&d),&bases,&state)
        .cast(burn::tensor::DType::F16).cast(burn::tensor::DType::F32).into_data().try_to_vec::<f32>().unwrap();
    for (i,(a,e))in actual.iter().zip(&expected).enumerate(){assert_eq!(a.to_bits(),e.to_bits(),"actor{},field{}",i/PREVIOUS,i%PREVIOUS);}
}

#[test]
fn natural_creeps_follow_their_own_directed_route_without_crossing_routes(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    w.lane_lengths=[2,2];w.lane=primitive(Tensor::<1>::from_data([-2900f32,100.,0.,-2400.,400.,0.,2900.,100.,0.,2400.,700.,0.],&d));
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    for hero in 0..2{let p=hero*STRIDE;s[p+UNIT_X]=if hero==0{-7000.}else{7000.};s[p+UNIT_Y]=7000.;}
    for (unit,team,x)in[(6,0,-3000f32),(7,1,3000f32)]{
        let p=unit*STRIDE;s[p..p+STRIDE].fill(0.);s[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[6]);
        s[p+KIND]=2.;s[p+ACTIVE]=1.;s[p+TEAM]=team as f32;s[p+UNIT_X]=x;s[p+UNIT_Y]=0.;
        s[p+TARGET]=-1.;s[p+LANE_POSITION]=w.lane_start(team)as f32;s[p+FACING]=if team==0{0.}else{std::f32::consts::PI};
    }
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let mut meta=vec![0.;META];meta[0]=-900.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
    let _=w.step_ticks(&rng,30);let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(s[6*STRIDE+LANE_POSITION],1.);assert_eq!(s[7*STRIDE+LANE_POSITION],3.);
    assert!(s[6*STRIDE+UNIT_X]>-3000. && s[6*STRIDE+UNIT_Y]>0.);
    assert!(s[7*STRIDE+UNIT_X]<3000. && s[7*STRIDE+UNIT_Y]>0.);
}

#[test]
fn repeated_sf_attacks_keep_the_native_fractional_period(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    s[STRIDE+UNIT_HEALTH]=100000.;s[STRIDE+UNIT_HEALTH_MAX]=100000.;s[ORDER]=3.;s[TARGET]=1.;s[TARGET_GENERATION]=0.;
    let interval=s[UNIT_ATTACK_INTERVAL];w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    // Native GetLastAttackTime in corrected-core30-002 advances35.294151ticks;
    // acceptance here is the existing one-tick event tolerance over12seconds.
    let mut releases=vec![];let mut hand=0.;
    for tick in 1..=360 {
        let _=w.step_ticks(&rng,1);let next=tensor::<1>(w.state.clone()).slice([ATTACK_HAND..ATTACK_HAND+1]).into_scalar::<f32>();
        if next!=hand{releases.push(tick);hand=next;}
    }
    assert!(releases.len()>=10,"missing continuous attacks: {releases:?}");
    for (i,&tick)in releases.iter().enumerate(){let expected=releases[0]as f32+i as f32*interval*30.;assert!((tick as f32-expected).abs()<=1.01,"release{i}: actual{tick},expected{expected},all{releases:?}");}
}

#[test]
fn native_melee_radius_aggro_preserves_target_retention(){
    let _g=GPU.lock().unwrap();
    let reference=data_root().join("runs/gpu/dota-server-aggro-014-melee-clean/server-scenarios.json");
    let report=crate::aggro_fidelity::validate(&engine(),&reference).unwrap();
    let failed:Vec<_>=report["cases"].as_array().unwrap().iter().filter(|r|r["passed"]!=true)
        .map(|r|r["case"]["radius"].clone()).collect();
    assert!(failed.is_empty(),"native melee radius control failed at{failed:?}");
}

#[test]
fn native_lane_acquisition_is_nearest_not_an_unconditional_creep_priority(){
    let _g=GPU.lock().unwrap();
    for (x,bait,expected)in[(100.,300.,0.),(100.,180.,7.),(200.,280.,7.),(400.,600.,7.),(100.,500.,0.),(200.,450.,0.)] {
        let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[UNIT_X]=0.;s[STRIDE+UNIT_X]=7000.;s[STRIDE+UNIT_Y]=7000.;
        for (unit,side,pos)in[(6,1.,x),(7,0.,bait)] {let p=unit*STRIDE;s[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[6]);s[p+KIND]=2.;s[p+ACTIVE]=1.;s[p+TEAM]=side;s[p+UNIT_X]=pos;s[p+UNIT_Y]=0.;s[p+TARGET]=-1.;s[p+LANE_POSITION]=-1.;}
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let _=w.step_ticks(&rng,1);
        assert_eq!(tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap()[6*STRIDE+TARGET],expected);
    }
}

#[test]
fn creep_auto_chase_waits_for_windup_and_backswing(){
    let _g=GPU.lock().unwrap();
    for windup in [false,true] {
        let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let time=w.engine.pre_game_seconds-30.;let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        for (unit,side,x)in[(6,1.,0.),(7,0.,300.)] {let p=unit*STRIDE;s[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[6]);s[p+KIND]=2.;s[p+ACTIVE]=1.;s[p+TEAM]=side;s[p+UNIT_X]=x;s[p+UNIT_Y]=0.;s[p+TARGET]=-1.;}
        let p=6*STRIDE;s[p+TARGET]=7.;s[p+WINDUP_TARGET]=7.;s[p+ATTACK_READY]=time+1.;s[p+WINDUP]=if windup{time+0.3}else{0.};s[7*STRIDE+UNIT_MOVE_SPEED]=0.;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let mut meta=vec![0.;META];meta[0]=-900.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));let _=w.step_ticks(&rng,8);
        assert_eq!(tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap()[p+UNIT_X],0.);
        let _=w.step_ticks(&rng,32);assert_ne!(tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap()[p+UNIT_X],0.);
    }
}

#[test]
fn natural_lane_goal_prioritizes_creeps_while_goal_less_npc_selects_nearest(){
    let _g=GPU.lock().unwrap();
    for (natural,forced,expected)in[(false,false,0.),(true,false,7.),(true,true,0.)]{
        let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        s[UNIT_X]=0.;s[STRIDE+UNIT_X]=7000.;s[STRIDE+UNIT_Y]=7000.;
        for(unit,side,x)in[(6,1.,100.),(7,0.,500.)]{let p=unit*STRIDE;s[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[6]);s[p+KIND]=2.;s[p+TEAM]=side;s[p+ACTIVE]=1.;s[p+UNIT_X]=x;s[p+UNIT_Y]=0.;s[p+TARGET]=-1.;s[p+LANE_POSITION]=-1.;s[p+UNIT_MOVE_SPEED]=0.;}
        let p=6*STRIDE;s[p+LANE_POSITION]=if natural{0.}else{-1.};if forced{s[p+TARGET]=0.;s[p+AGGRO_END]=1000.;}
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let _=w.step_ticks(&rng,1);
        assert_eq!(tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap()[p+TARGET],expected);
    }
}

#[test]
fn attacking_flag_uses_own_action_but_other_units_attack_activity(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    s[UNIT_X]=-400.;s[STRIDE+UNIT_X]=600.;s[STRIDE+ORDER]=3.;s[STRIDE+TARGET]=0.;w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let mut meta=vec![0.;META];meta[0]=1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));let _=w.step_ticks(&rng,1);
    let b=w.observe(initial_previous(2,&d)).decode().heroes.into_data().try_to_vec::<f32>().unwrap();assert_eq!(b[2*UNIT_COUNT+UNIT_IS_ATTACKING],0.);assert_eq!(b[4*UNIT_COUNT+UNIT_IS_ATTACKING],1.);
}

#[test]
fn tower_protection_is_authoritative_in_physics_and_known_only_after_sight(){
    let _g=GPU.lock().unwrap();let d=device();let mut w=World::load(&engine(),1,&d).unwrap();let rng=Sampler::new(1,7,&d);
    let masks=w.observe(initial_previous(2,&d)).decode().attack_mask.int().into_data().try_to_vec::<i32>().unwrap();let target=2+crate::observe::CREEPS+2;
    assert_eq!(masks[target],1);assert_eq!(masks[target+1],1,"native unseen T2 protection flag defaults false");
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[4*STRIDE+INVULNERABLE],1.);
    for u in 2..UNITS {if u!=4 {s[u*STRIDE+ACTIVE]=0.;}}
    s[UNIT_X]=-400.;s[UNIT_Y]=0.;s[FACING]=0.;s[ATTACK_AIM]=0.;s[ORDER]=3.;s[TARGET]=4.;s[STRIDE+UNIT_X]=7000.;s[STRIDE+UNIT_Y]=7000.;
    s[4*STRIDE+UNIT_X]=300.;s[4*STRIDE+UNIT_Y]=0.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    w.fow_blockers=primitive(Tensor::<1>::zeros([512*512],&d));w.geometry=primitive(Tensor::<1>::ones([512*512*2],&d));w.movement_geometry=None;w.tree_map=primitive(Tensor::<1>::zeros([512*512],&d));
    let mut meta=vec![0.;META];meta[0]=-30.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));let _=w.step_ticks(&rng,10);
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[4*STRIDE+UNIT_HEALTH],1800.);assert_eq!(s[WINDUP],0.);assert!(s[UNIT_X]>-400.);
    assert_eq!(w.observe(initial_previous(2,&d)).decode().attack_mask.int().into_data().try_to_vec::<i32>().unwrap()[target],0);
    let mut meta=tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap();meta[0]=0.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));let _=w.step_ticks(&rng,40);
    assert!(hp(&w,4)<1800.);assert_eq!(w.observe(initial_previous(2,&d)).decode().attack_mask.int().into_data().try_to_vec::<i32>().unwrap()[target],1);
}

#[test]
fn lane_creep_attack_readiness_does_not_wait_for_its_visible_turn(){
    let _g=GPU.lock().unwrap();
    for kind in [2,3,4] {
        let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();let p=6*STRIDE;
        s[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[kind+4]);s[p+KIND]=kind as f32;s[p+TEAM]=1.;s[p+ACTIVE]=1.;s[p+UNIT_X]=-100.;s[p+UNIT_Y]=0.;s[p+TARGET]=0.;s[p+AGGRO_END]=1000.;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let mut meta=vec![0.;META];meta[0]=-900.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));let _=w.step_ticks(&rng,1);
        let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert!(s[p+WINDUP]>0.);assert_eq!(s[p+WINDUP_TARGET],0.);assert!((s[p+FACING].abs()-0.3351032).abs()<0.0001);
    }
}

#[test]
fn native_attack_reach_counts_both_hulls_and_chasing_is_not_an_attack_target(){
    let _g=GPU.lock().unwrap();
    for distance in [573.,580.] {
        let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        s[STRIDE+UNIT_X]=s[UNIT_X]+distance;s[ORDER]=3.;s[TARGET]=1.;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let _=w.step_ticks(&rng,1);
        let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        let b=w.observe(initial_previous(2,&d)).decode().heroes.into_data().try_to_vec::<f32>().unwrap();
        if distance==573. {assert_eq!(s[UNIT_X],-200.);assert!(s[WINDUP]>0.);assert_eq!(b[UNIT_ATTACK_TARGET_ENEMY],1.);}
        else{assert!(s[UNIT_X]>-200.);assert_eq!(s[WINDUP],0.);assert_eq!(b[UNIT_ATTACK_TARGET_ENEMY],0.);}
    }
}

#[test]
fn queued_aggro_waits_for_the_old_attack_and_cannot_be_refreshed_before_three_seconds(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let time=w.engine.pre_game_seconds-30.;
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    for (unit,side,x)in[(6,1.,100.),(7,0.,180.)] {let p=unit*STRIDE;s[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[6]);s[p+ACTIVE]=1.;s[p+KIND]=2.;s[p+TEAM]=side;s[p+UNIT_X]=x;s[p+UNIT_Y]=0.;s[p+TARGET]=-1.;}
    let p=6*STRIDE;s[p+TARGET]=7.;s[p+WINDUP_TARGET]=7.;s[p+WINDUP]=time+0.3;s[p+ATTACK_READY]=time+1.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let mut meta=vec![0.;META];meta[0]=-900.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
    action_command(&w,&rng,[3,0,0,1,64,64]);let _=w.step_ticks(&rng,4);
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[p+TARGET],7.);assert_eq!(s[p+AGGRO_PENDING],1.);let ready=s[p+AGGRO_READY];
    let _=w.step_ticks(&rng,24);let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[p+TARGET],0.);assert_eq!(s[p+AGGRO_PENDING],0.);assert_eq!(s[p+AGGRO_PREVIOUS],8.);
    action_command(&w,&rng,[3,0,0,1,64,64]);let _=w.step_ticks(&rng,4);let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[p+AGGRO_READY],ready);
}

#[test]
fn expired_aggro_restores_target_but_keeps_old_chase_for_half_a_second(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let time=w.engine.pre_game_seconds-30.;
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[UNIT_X]=-800.;
    for (unit,side,x)in[(6,1.,-400.),(7,0.,100.)] {let p=unit*STRIDE;s[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[6]);s[p+ACTIVE]=1.;s[p+KIND]=2.;s[p+TEAM]=side;s[p+UNIT_X]=x;s[p+UNIT_Y]=0.;s[p+TARGET]=-1.;}
    let p=6*STRIDE;s[p+TARGET]=0.;s[p+AGGRO_END]=time-0.1;s[p+AGGRO_PREVIOUS]=8.;s[p+FACING]=std::f32::consts::PI;s[p+ATTACK_AIM]=std::f32::consts::PI;s[7*STRIDE+UNIT_MOVE_SPEED]=0.;s[STRIDE+UNIT_X]=7000.;s[STRIDE+UNIT_Y]=7000.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let mut meta=vec![0.;META];meta[0]=-900.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));let _=w.step_ticks(&rng,1);
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[p+TARGET],7.);assert!(s[p+AGGRO_RECOVER]>time+0.4);assert!(s[p+UNIT_X]< -400.);
    let _=w.step_ticks(&rng,8);let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[p+TARGET],7.);let x=s[p+UNIT_X];assert!(x< -480.);
    let _=w.step_ticks(&rng,24);let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[p+TARGET],7.);assert!(s[p+UNIT_X]>x,"restored creep target must eventually reverse the chase");
}

#[test]
fn siege_aggro_can_retarget_after_animation_while_weapon_cooldown_continues(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let time=w.engine.pre_game_seconds-30.;
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();let p=6*STRIDE;
    s[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[8]);s[p+ACTIVE]=1.;s[p+KIND]=4.;s[p+TEAM]=1.;s[p+UNIT_X]=-100.;s[p+UNIT_Y]=0.;s[p+TARGET]=1.;s[p+WINDUP_TARGET]=1.;s[p+ATTACK_READY]=time+1.;s[p+AGGRO_PENDING]=1.;s[p+AGGRO_END]=time+2.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let mut meta=vec![0.;META];meta[0]=-900.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));let _=w.step_ticks(&rng,10);
    assert_eq!(tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap()[p+TARGET],1.);
    let _=w.step_ticks(&rng,5);let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[p+TARGET],0.);assert_eq!(s[p+WINDUP],0.);assert!(s[p+ATTACK_READY]>time+0.9);
}

#[test]
fn hero_attack_orders_never_provoke_an_inactive_fog_cache(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();let p=FOG_START*STRIDE;
    s[p+ACTIVE]=0.;s[p+TEAM]=1.;s[p+KIND]=1.;s[p+UNIT_HEALTH]=1800.;s[p+UNIT_X]=-100.;s[p+UNIT_Y]=0.;s[p+TARGET]=-1.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let mut commands=vec![0.;COMMAND_WORDS];commands[1]=3.;commands[4]=1.;commands[7]=1.;w.commands=primitive(Tensor::<1>::from_data(TensorData::new(commands,[COMMAND_WORDS]),&d));
    let mut meta=vec![0.;META];meta[0]=1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));let _=w.step_ticks(&rng,1);
    assert_eq!(tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap()[p+TARGET],-1.);
}

#[test]
fn shop_stock_is_bounded_and_remote_purchases_wait_in_stash(){
    let _g=GPU.lock().unwrap();let mut w=arena();let rng=Sampler::new(1,7,&w.device);
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();state[UNIT_GOLD]=10000.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&w.device));
    for _ in 0..8 {action_command(&w,&rng,[0,3,0,0,64,64]);}
    for _ in 0..3 {command(&w,&rng,0);}
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(s[STASH_START+4],4.);assert_eq!(s[STOCK_START+4],4.);assert_eq!(s[ITEM_START+4],0.);
    assert!((s[UNIT_GOLD]-(10000.-4.*65.+44./30.*100./60.)).abs()<0.1);
}

#[test]
fn requiem_launches_real_lines_and_cannot_hit_same_unit_twice_per_line(){
    let _g=GPU.lock().unwrap();
    for distance in [150.,400.] {
    let mut w=arena();let rng=Sampler::new(1,7,&w.device);
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    state[UNIT_LEVEL]=6.;state[UNIT_SOULS]=2.;state[UNIT_MANA]=267.;
    state[STRIDE+UNIT_X]=state[UNIT_X]+distance;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&w.device));
    command(&w,&rng,9);let _=w.step_ticks(&rng,55);
    let shots=tensor::<1>(w.shots.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(shots.chunks_exact(SHOT_FIELDS).filter(|p|p[0]==2.).count(),2);
    let _=w.step_ticks(&rng,55);let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert!(s[STRIDE+UNIT_HEALTH]<540. && s[STRIDE+UNIT_HEALTH]>500.,"one forward line should hit once: {}",s[STRIDE+UNIT_HEALTH]);
    assert_eq!(s[STRIDE+MRES_REDUCTION],0.,"current native Requiem does not amplify subsequent magic damage");
    }
}
fn hp(w:&World,u:usize)->f32 {tensor::<1>(w.state.clone()).slice([u*STRIDE+UNIT_HEALTH..u*STRIDE+UNIT_HEALTH+1]).into_scalar()}

#[test]
fn many_full_rotations_do_not_change_the_motion_or_observation_heading(){
    let _g=GPU.lock().unwrap();let mut outcomes=vec![];
    for turns in [0.,20.] {let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[FACING]=std::f32::consts::FRAC_PI_2+turns*std::f32::consts::PI;s[ATTACK_AIM]=s[FACING];w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
        action_command(&w,&rng,[2,0,0,0,64,80]);let _=w.step_ticks(&rng,4);let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();outcomes.push([s[UNIT_X],s[UNIT_Y],s[FACING].cos(),s[FACING].sin()]);assert!(s[FACING].abs()<=std::f32::consts::PI);
    }
    assert!(outcomes[0].iter().zip(outcomes[1]).all(|(a,b)|(a-b).abs()<0.0001));
}

#[test]
fn attack_turn_readiness_does_not_snap_the_observed_facing_during_windup(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    s[FACING]=std::f32::consts::PI;s[ATTACK_AIM]=s[FACING];s[ORDER]=3.;s[TARGET]=1.;w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let mut meta=vec![0.;META];meta[0]=1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
    let _=w.step_ticks(&rng,4);let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert!(s[WINDUP]>0.);assert!(s[FACING]>1.);assert!(s[ATTACK_AIM].abs()<0.001);
    let _=w.step_ticks(&rng,8);let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert!(s[FACING].abs()<0.001);
}

#[test]
fn casting_cannot_track_a_stored_attack_target_through_fog(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    s[STRIDE+UNIT_X]=7000.;s[STRIDE+UNIT_Y]=7000.;s[TARGET]=1.;s[CAST_END]=1000.;s[CAST]=3.;s[ORDER]=0.;w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let _=w.step_ticks(&rng,1);
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[TARGET],-1.);assert_eq!(s[FACING],0.);assert!(s[CAST_END]>0.);
}

#[test]
fn flagbearer_spawns_replace_one_melee_and_keep_native_type_features(){
    let _g=GPU.lock().unwrap();let d=device();
    for seconds in [90,120,180,480] {
        let mut w=World::load(&engine(),1,&d).unwrap();let rng=Sampler::new(1,7,&d);let mut meta=vec![0.;META];meta[0]=(seconds*30)as f32;
        w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));let _=w.step_ticks(&rng,1);
        let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        let flags:Vec<_>=(6..CREEP_END).filter(|u|s[u*STRIDE+ACTIVE]>0. && s[u*STRIDE+FLAGBEARER]>0.).collect();assert_eq!(flags.len(),if seconds<120{0}else{2});
        for side in 0..2 {assert_eq!((6..CREEP_END).filter(|u|s[u*STRIDE+ACTIVE]>0. && s[u*STRIDE+TEAM]==side as f32).count(),4);}
        for u in flags {let p=u*STRIDE;let expected=if seconds<450{0.4}else{0.424};assert!((s[p+UNIT_MAGIC_RESIST]-expected).abs()<0.000001);assert_eq!(s[p+XP_BOUNTY],60.);assert!((78. ..=88.).contains(&s[p+BOUNTY]));assert_eq!(s[p+BOUNTY]%2.,0.);}
        if seconds>=120 {let seen=w.observe(initial_previous(2,&d)).decode().creeps.into_data().try_to_vec::<f32>().unwrap();
            let flag=seen.chunks_exact(2*UNIT_COUNT).find(|v|v[UNIT_MAGIC_RESIST]>0.35).unwrap();assert_eq!(&flag[UNIT_CREEP_MELEE..=UNIT_CREEP_SIEGE],&[0.,0.,0.]);assert!((flag[UNIT_HEALTH_REGEN]*20.-3.5).abs()<0.001);
        }
    }
}

#[test]
fn flagbearer_regen_is_shared_with_creeps_and_lingers_without_buffing_heroes(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[UNIT_HEALTH]=300.;s[STRIDE+UNIT_X]=7000.;s[STRIDE+UNIT_Y]=7000.;
    for (unit,x)in[(6,0.),(7,500.)]{let p=unit*STRIDE;s[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[6]);s[p+UNIT_X]=x;s[p+UNIT_Y]=0.;s[p+TEAM]=0.;s[p+KIND]=2.;s[p+ACTIVE]=1.;s[p+TARGET]=-1.;s[p+UNIT_HEALTH]=100.;}
    s[6*STRIDE+FLAGBEARER]=1.;w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let mut meta=vec![0.;META];meta[0]=1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));let _=w.step_ticks(&rng,1);
    assert!((hp(&w,7)-(100.+3.5/30.)).abs()<0.001);assert!((hp(&w,0)-(300.+2.35/30.)).abs()<0.001);
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[6*STRIDE+ACTIVE]=0.;w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let _=w.step_ticks(&rng,20);
    let before=hp(&w,7);let _=w.step_ticks(&rng,4);assert!((hp(&w,7)-before-0.5*4./30.).abs()<0.001);
}

#[test]
fn flagbearer_hero_last_hit_credits_its_gold_and_sixty_xp(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();let p=6*STRIDE;
    s[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[6]);s[p+UNIT_X]=0.;s[p+UNIT_Y]=0.;s[p+UNIT_HEALTH]=1.;s[p+ACTIVE]=1.;s[p+KIND]=2.;s[p+TEAM]=1.;s[p+FLAGBEARER]=1.;s[p+DAMAGE]=100.;s[p+CREDIT]=0.;s[p+BOUNTY]=84.;s[p+XP_BOUNTY]=60.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let mut meta=vec![0.;META];meta[0]=1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));let _=w.step_ticks(&rng,1);
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[UNIT_LAST_HITS],1.);assert_eq!(s[UNIT_XP_TO_NEXT],180.);assert!((s[UNIT_GOLD]-(95.+84.+100./60./30.)).abs()<0.001);
}

#[test]
fn tower_armor_changes_physical_damage_and_observed_regen_with_presence(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    s[UNIT_HEALTH]=300.;s[STRIDE+UNIT_LEVEL]=2.;let tower=2*STRIDE;s[tower+ACTIVE]=1.;s[tower+UNIT_X]=723.;s[tower+UNIT_Y]=0.;s[tower+TOWER_TIER]=1.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let mut shots=vec![0.;SHOTS*SHOT_FIELDS];shots[..10].copy_from_slice(&[1.,-200.,0.,0.,0.,900.,100.,1.,1.,0.]);w.shots=primitive(Tensor::<1>::from_data(TensorData::new(shots,[SHOTS*SHOT_FIELDS]),&d));
    let _=w.step_ticks(&rng,1);let obs=w.observe(initial_previous(2,&d)).packed;
    assert!((obs.clone().slice([0..1,UNIT_ARMOR..UNIT_ARMOR+1]).into_scalar::<f32>()*20.-7.25).abs()<0.001);
    assert!((obs.slice([0..1,UNIT_HEALTH_REGEN..UNIT_HEALTH_REGEN+1]).into_scalar::<f32>()*20.-3.35).abs()<0.001);
    let expected=300.+3.35/30.-(100f32/(1.+0.06*7.25)).ceil();assert!((hp(&w,0)-expected).abs()<0.001);
}

#[test]
fn tower_aura_uses_the_strongest_tier_and_expires_each_linger_independently(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    s[STRIDE+UNIT_X]=7000.;s[STRIDE+UNIT_Y]=7000.;
    for (unit,tier)in[(2,1.),(3,2.)]{let p=unit*STRIDE;s[p+ACTIVE]=1.;s[p+UNIT_X]=723.;s[p+UNIT_Y]=0.;s[p+TOWER_TIER]=tier;}
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let _=w.step_ticks(&rng,1);
    let armor=|w:&World|w.observe(initial_previous(2,&d)).packed.slice([0..1,UNIT_ARMOR..UNIT_ARMOR+1]).into_scalar::<f32>()*20.;
    assert!((armor(&w)-11.75).abs()<0.001);
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[3*STRIDE+ACTIVE]=0.;w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let _=w.step_ticks(&rng,10);assert!((armor(&w)-11.75).abs()<0.001);let _=w.step_ticks(&rng,10);assert!((armor(&w)-9.75).abs()<0.001);
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[2*STRIDE+ACTIVE]=0.;w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let _=w.step_ticks(&rng,20);assert!((armor(&w)-6.75).abs()<0.001);
}

#[test]
fn casting_after_movement_keeps_turning_and_resumes_the_suspended_order(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let mut commands=vec![0.;COMMAND_WORDS];commands[1]=2.;commands[5]=-200.;commands[6]=256.;commands[7]=1.;
    commands[COMMAND_FIELDS]=1.;commands[COMMAND_FIELDS+1]=8.;commands[COMMAND_FIELDS+7]=1.;
    w.commands=primitive(Tensor::<1>::from_data(TensorData::new(commands,[COMMAND_WORDS]),&d));
    let _=w.step_ticks(&rng,12);let casting=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert!(casting[CAST_END]>0.);assert!((casting[FACING]-std::f32::consts::FRAC_PI_2).abs()<0.001);assert!((casting[UNIT_Y]-305./30.).abs()<0.001);
    let _=w.step_ticks(&rng,12);let after=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(after[CAST_END],0.);assert_eq!(after[ORDER],2.);assert!(after[UNIT_Y]>casting[UNIT_Y]);
}

#[test]
fn sf_can_start_moving_while_still_turning_toward_a_sideways_order(){
    let _g=GPU.lock().unwrap();let w=arena();let rng=Sampler::new(1,7,&w.device);
    action_command(&w,&rng,[2,0,0,0,64,80]);let _=w.step_ticks(&rng,4);
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert!(s[UNIT_Y]>0.);assert!(s[FACING]>0. && s[FACING]<std::f32::consts::FRAC_PI_2);
}

#[test]
fn physics_navigation_is_separate_from_the_bot_api_observation(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let mut api=vec![1.;512*512*2];for cell in api.chunks_exact_mut(2){cell[1]=0.;}
    w.geometry=primitive(Tensor::<1>::from_data(TensorData::new(api,[512*512*2]),&d));
    let before=w.observe(initial_previous(2,&d)).packed;
    w.movement_geometry=Some(primitive(Tensor::<1>::ones([512*512*2],&d)));
    let after=w.observe(initial_previous(2,&d)).packed;assert_eq!((before-after).abs().max().into_scalar::<f32>(),0.);
    for _ in 0..4 {action_command(&w,&rng,[2,0,0,0,100,64]);}
    let x=tensor::<1>(w.state.clone()).slice([UNIT_X..UNIT_X+1]).into_scalar::<f32>();assert!(x> -200.,"physical walkability must allow motion despite API-false terrain");
}

#[test]
fn blocked_spawn_recovery_leaves_its_cell_without_entering_a_new_wall(){
    let _g=GPU.lock().unwrap();
    for start in [16.,-200.] {
        let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let mut geometry=vec![1.;512*512*2];
        for y in 0..512 {geometry[(y*512+256)*2+1]=0.;}
        w.geometry=primitive(Tensor::<1>::from_data(TensorData::new(geometry,[512*512*2]),&d));
        let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();state[UNIT_X]=start;state[UNIT_Y]=16.;state[STRIDE+UNIT_X]=7000.;state[STRIDE+UNIT_Y]=7000.;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
        action_command(&w,&rng,[2,0,0,0,100,64]);for _ in 0..5 {let _=w.step_ticks(&rng,30);}
        let state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert!(state[UNIT_X]<0.,"crossed or remained inside wall from{start}:{}",state[UNIT_X]);
    }
}

#[test]
fn native_dire_ranged_creep_leaves_its_spawn_instead_of_stalling(){
    let _g=GPU.lock().unwrap();let d=device();let mut w=World::load(&engine(),1,&d).unwrap();let rng=Sampler::new(1,7,&d);
    w.meta=primitive(Tensor::<1>::zeros([META],&d));let _=w.step_ticks(&rng,1);
    let initial=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    let unit=(6..CREEP_END).find(|u|initial[u*STRIDE+ACTIVE]>0. && initial[u*STRIDE+TEAM]==1. && initial[u*STRIDE+KIND]==3.).unwrap();
    let p=unit*STRIDE;let x=initial[p+UNIT_X];let y=initial[p+UNIT_Y];
    for _ in 0..5 {let _=w.step_ticks(&rng,30);}
    let state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    let moved=((state[p+UNIT_X]-x).powi(2)+(state[p+UNIT_Y]-y).powi(2)).sqrt();
    assert!(moved>500.,"Dire ranged creep stalled at({},{}) after5seconds; moved{moved}",state[p+UNIT_X],state[p+UNIT_Y]);
}

#[test]
fn lane_creeps_reconsider_chased_hero_but_keep_in_range_and_forced_targets(){
    let _g=GPU.lock().unwrap();
    for (forced,creep_x,expected)in[(false,200.,7.),(false,100.,0.),(true,200.,0.)] {
        let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        s[UNIT_X]=0.;s[STRIDE+UNIT_X]=7000.;s[STRIDE+UNIT_Y]=7000.;
        for (unit,team,x)in[(6,1.,creep_x),(7,0.,300.)] {
            let p=unit*STRIDE;s[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[6]);s[p+UNIT_X]=x;s[p+UNIT_Y]=0.;s[p+KIND]=2.;s[p+TEAM]=team;s[p+ACTIVE]=1.;s[p+TARGET]=-1.;
        }
        s[6*STRIDE+TARGET]=0.;s[6*STRIDE+AGGRO_END]=if forced {1000.}else{0.};
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
        let _=w.step_ticks(&rng,1);let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(s[6*STRIDE+TARGET],expected);
    }
}

#[test]
fn native_melee_and_ranged_creeps_have_distinct_acquisition_ranges(){
    let _g=GPU.lock().unwrap();
    for (kind,expected) in [(2,-1.),(3,0.)] {
        let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
        let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();let p=6*STRIDE;
        state[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[if kind==2 {6}else{7}]);
        state[p+UNIT_X]=350.;state[p+UNIT_Y]=0.;state[p+TEAM]=1.;state[p+KIND]=kind as f32;state[p+ACTIVE]=1.;state[p+TARGET]=-1.;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));let mut meta=vec![0f32;META];meta[0]=1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
        let _=w.step_ticks(&rng,1);let state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(state[p+TARGET],expected);
    }
}

#[test]
fn native_max_resources_floor_total_attributes_without_rounding_attack_speed(){
    let _g=GPU.lock().unwrap();
    for (level,hp,mana)in[(2,626.,315.),(3,692.,339.),(6,868.,423.)] {
        let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[UNIT_LEVEL]=level as f32;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let _=w.step_ticks(&rng,1);
        let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[UNIT_HEALTH_MAX],hp);assert_eq!(s[UNIT_MANA_MAX],mana);
        let speed=1.6/(1.+(25.+5.+3.6*(level-1)as f32+6.)/100.);assert!((s[UNIT_ATTACK_INTERVAL]-speed).abs()<0.000001);
    }
}

#[test]
fn necromastery_attack_bonus_scales_at_level_six_as_in_native_records(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();state[UNIT_LEVEL]=6.;state[UNIT_SOULS]=10.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));let _=w.step_ticks(&rng,1);
    let state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert!((state[UNIT_ATTACK_DAMAGE]-85.).abs()<0.01);
}

#[test]
fn authored_fow_walls_block_ground_sight_but_team_revealers_open_it(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();w.engine.extra_vision.clear();
    let mut blockers=vec![0.;512*512];for y in 0..512 {blockers[y*512+256]=1.;}
    w.fow_blockers=primitive(Tensor::<1>::from_data(TensorData::new(blockers,[512*512]),&d));
    let seen=w.observe(initial_previous(2,&d)).decode().hero_valid.int().into_data().try_to_vec::<i32>().unwrap();assert_eq!(seen,vec![1,0,1,0]);
    w.engine.extra_vision=vec![StaticVision{x:0.,y:0.,team:0.,day:1000.,night:1000.,unobstructed:true}];
    w.extra_vision=primitive(Tensor::<1>::from_data([0.,0.,0.,1000.,1000.,1.],&d));
    let seen=w.observe(initial_previous(2,&d)).decode().hero_valid.int().into_data().try_to_vec::<i32>().unwrap();assert_eq!(seen,vec![1,1,1,0]);
}

#[test]
fn permanent_base_sources_share_only_their_teams_current_vision(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    state[UNIT_X]=-7000.;state[UNIT_Y]=-7000.;state[STRIDE+UNIT_X]=1000.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
    w.engine.extra_vision=vec![StaticVision{x:0.,y:0.,team:0.,day:1500.,night:600.,unobstructed:false}];
    w.extra_vision=primitive(Tensor::<1>::from_data([0.,0.,0.,1500.,600.,0.],&d));
    for (ticks,visible) in [(-1.,0),(1.,1),(9001.,0)] {
        let mut meta=vec![0.;META];meta[0]=ticks;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
        let b=w.observe(initial_previous(2,&d)).decode();let seen=b.hero_valid.int().into_data().try_to_vec::<i32>().unwrap();
        let mask=b.attack_mask.int().into_data().try_to_vec::<i32>().unwrap();assert_eq!(seen[1],visible);assert_eq!(mask[1],visible);assert_eq!(seen[3],0,"source must not reveal the enemy team");
    }
}

#[test]
fn day_night_boundaries_and_native_tower_tiers_control_shared_vision(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();state[STRIDE+UNIT_X]=1000.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state.clone(),[UNITS*STRIDE]),&d));
    for (ticks,visible) in [(-1.,0),(0.,0),(1.,1),(8999.,1),(9000.,1),(9001.,0),(18000.,0)] {
        let mut meta=vec![0.;META];meta[0]=ticks;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
        let seen=w.observe(initial_previous(2,&d)).decode().hero_valid.int().into_data().try_to_vec::<i32>().unwrap();assert_eq!(seen[1],visible,"clock tick{ticks}");
    }
    // A remote allied T2 shares1100 night vision; T1 shares only800.
    state[UNIT_X]=-7000.;state[UNIT_Y]=-7000.;
    for (tower,visible) in [(2,0),(3,1)] {
        let mut current=state.clone();let p=tower*STRIDE;current[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[tower]);
        current[p+UNIT_X]=0.;current[p+UNIT_Y]=0.;current[p+TEAM]=0.;current[p+KIND]=1.;current[p+ACTIVE]=1.;
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(current,[UNITS*STRIDE]),&d));
        let mut meta=vec![0.;META];meta[0]=9001.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
        let seen=w.observe(initial_previous(2,&d)).decode().hero_valid.int().into_data().try_to_vec::<i32>().unwrap();assert_eq!(seen[1],visible,"tower{tower}");
    }
}

#[test]
fn native_height_levels_block_uphill_vision_and_produce_the_correct_physical_z(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let mut geometry=vec![1f32;512*512*2];
    for y in 0..512 {for x in 0..512 {geometry[(y*512+x)*2]=if x<256 {4.}else{3.};}}
    w.geometry=primitive(Tensor::<1>::from_data(TensorData::new(geometry,[512*512*2]),&d));
    let valid=w.observe(initial_previous(2,&d)).decode().hero_valid.int().into_data().try_to_vec::<i32>().unwrap();assert_eq!(valid,vec![1,0,1,1]);
    let rng=Sampler::new(1,7,&d);action_command(&w,&rng,[2,0,0,0,128,64]);let _=w.step_ticks(&rng,12);
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(state[UNIT_Z],0.);
    state[UNIT_X]=100.;w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));let _=w.step_ticks(&rng,1);
    let state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(state[UNIT_Z],128.);
}

#[test]
fn observation_known_flags_and_free_scroll_follow_the_native_contract(){
    let _g=GPU.lock().unwrap();let w=arena();let b=w.observe(initial_previous(2,&w.device)).decode();
    assert_eq!(b.action_mask.clone().int().into_data().try_to_vec::<i32>().unwrap()[5],0,"native Action_HoldPosition is unavailable");
    let heroes=b.heroes.into_data().try_to_vec::<f32>().unwrap();
    for field in UNIT_TEAM_RELATIVE..UNIT_COUNT {assert_eq!(heroes[UNIT_COUNT+field],0.);}
    let items=b.items.into_data().try_to_vec::<f32>().unwrap();let tp=3*16;
    assert!((items[tp]-0.1).abs()<1e-6);assert!((items[tp+4]-(93.-83.)/120.).abs()<1e-5);assert_eq!(items[tp+7],0.75);
    let empty=4*16;assert_eq!(items[empty+12],0.);assert_eq!(items[empty+14],0.);
    let initial=World::load(&engine(),1,&w.device).unwrap();let hero=initial.observe(initial_previous(2,&w.device)).decode().heroes.into_data().try_to_vec::<f32>().unwrap();
    // First native decision of train match-0001-attempt-09, including aura.
    assert!((hero[UNIT_HEALTH_REGEN]*20.-31.45).abs()<0.02);assert!((hero[UNIT_MANA_REGEN]*10.-18.66).abs()<0.02);
}

#[test]
fn move_clears_attack_target_and_cannot_be_cancelled_by_its_old_targets_fog(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    command(&w,&rng,3);let _=w.step_ticks(&rng,6);
    let prior=w.observe(initial_previous(2,&d)).decode().heroes.into_data().try_to_vec::<f32>().unwrap();assert_eq!(prior[UNIT_ATTACK_TARGET_ENEMY],1.);
    action_command(&w,&rng,[2,0,0,0,0,64]);let _=w.step_ticks(&rng,8);
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(state[TARGET],-1.);
    state[STRIDE+UNIT_X]=7000.;state[STRIDE+UNIT_Y]=7000.;w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));let _=w.step_ticks(&rng,1);
    let state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(state[ORDER],2.);
}

#[test]
fn public_tower_orders_work_in_fog_and_hidden_hp_uses_each_observers_memory(){
    let _g=GPU.lock().unwrap();let d=device();let mut w=World::load(&engine(),1,&d).unwrap();
    w.fow_blockers=primitive(Tensor::<1>::zeros([512*512],&d));w.geometry=primitive(Tensor::<1>::ones([512*512*2],&d));w.tree_map=primitive(Tensor::<1>::zeros([512*512],&d));
    let first=w.observe(initial_previous(2,&d));
    let tower_target=2+crate::observe::CREEPS+2;
    assert_ne!(first.decode().attack_mask.int().into_data().try_to_vec::<i32>().unwrap()[tower_target],0);
    let initial=first.packed.slice([0..1]).into_data().try_to_vec::<f32>().unwrap();
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();state[4*STRIDE+UNIT_HEALTH]=1.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state.clone(),[UNITS*STRIDE]),&d));
    let hidden=w.observe(initial_previous(2,&d)).packed.slice([0..1]).into_data().try_to_vec::<f32>().unwrap();assert_eq!(initial,hidden);
    state[UNIT_X]=state[4*STRIDE+UNIT_X]+400.;state[UNIT_Y]=state[4*STRIDE+UNIT_Y];
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
    let seen=w.observe(initial_previous(2,&d)).decode().towers.into_data().try_to_vec::<f32>().unwrap();assert_eq!(seen[2*2*UNIT_COUNT+UNIT_HEALTH],1./2000.);
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();state[UNIT_X]=-6700.;state[UNIT_Y]=-6700.;state[4*STRIDE+UNIT_HEALTH]=42.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
    let remembered=w.observe(initial_previous(2,&d)).decode().towers.into_data().try_to_vec::<f32>().unwrap();assert_eq!(remembered[2*2*UNIT_COUNT+UNIT_HEALTH],1./2000.);
    let rng=Sampler::new(1,7,&d);action_command(&w,&rng,[3,0,0,tower_target as i32,64,64]);let _=w.step_ticks(&rng,12);
    let state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(state[ORDER],3.);assert_eq!(state[TARGET],4.);
}

#[test]
fn hold_attacks_in_range_without_chasing_and_move_clears_hold(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let before=hp(&w,1);command(&w,&rng,5);let _=w.step_ticks(&rng,55);
    assert!(hp(&w,1)<before-20.,"hold did not acquire and attack a visible enemy: {} -> {}",before,hp(&w,1));
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    s[STRIDE+UNIT_X]=1000.;w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let _=w.step_ticks(&rng,30);
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(s[UNIT_X],-200.);assert_eq!(s[TARGET],-1.);
    action_command(&w,&rng,[2,0,0,0,70,64]);let _=w.step_ticks(&rng,10);
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[HOLD],0.);assert!(s[UNIT_X]>-200.);
}

#[test]
fn moving_hero_cannot_pass_through_units_and_can_leave_an_overlap(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    action_command(&w,&rng,[2,0,0,0,128,64]);
    for _ in 0..110 {
        let _=w.step_ticks(&rng,1);let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        let dx=s[UNIT_X]-s[STRIDE+UNIT_X];let dy=s[UNIT_Y]-s[STRIDE+UNIT_Y];
        assert!(dx*dx+dy*dy>=48.*48.-0.01,"hero hulls overlapped");
    }
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert!(s[UNIT_X]>200.,"local steering did not pass the blocker: {},{}",s[UNIT_X],s[UNIT_Y]);
    s[STRIDE+UNIT_X]=s[UNIT_X];s[STRIDE+UNIT_Y]=s[UNIT_Y];
    s[ORDER]=2.;s[DEST_X]=s[UNIT_X]+512.;s[DEST_Y]=s[UNIT_Y];
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let _=w.step_ticks(&rng,20);
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert!((s[UNIT_X]-s[STRIDE+UNIT_X]).abs()>48.,"overlapping spawn permanently trapped hero");
}

#[test]
fn fear_slows_actual_movement_and_the_reported_move_speed(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();let p=STRIDE;
    s[p+FACING]=0.;s[p+FEAR_END]=100.;s[p+SLOW_PERCENT]=-0.2;s[p+FEAR_X]=-200.;s[p+FEAR_Y]=0.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let _=w.step_ticks(&rng,1);
    let speed=w.engine.prototypes[1][UNIT_MOVE_SPEED]*0.8;
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert!((s[p+UNIT_X]-(200.+speed/30.)).abs()<0.001);
    let observed=w.observe(initial_previous(2,&d)).decode().heroes.into_data().try_to_vec::<f32>().unwrap();
    assert!((observed[4*UNIT_COUNT+UNIT_MOVE_SPEED]-speed/600.).abs()<0.001);
}

#[test]
fn policy_trace_two_frames_keep_fp16_observations_and_f32_aux_separate(){
    let _g=GPU.lock().unwrap();let d=device();let cache=crate::policy_trace::PolicyTrace::new(2,&d);
    let obs=Tensor::<2>::ones([1,crate::observe::WIDTH],(&d,burn::tensor::DType::F16))*0.5;
    let extra=Tensor::<2>::ones([1,crate::policy_trace::AUX],&d)*7.;cache.record(0,obs,extra);
    cache.record(1,Tensor::<2>::zeros([1,crate::observe::WIDTH],(&d,burn::tensor::DType::F16)),Tensor::<2>::ones([1,crate::policy_trace::AUX],&d)*-3.);
    assert_eq!(cache.observations.dtype,burn::tensor::DType::F16);
    let x=tensor::<2>(cache.observations).into_data().try_to_vec::<burn::tensor::f16>().unwrap();
    assert!(x[..crate::observe::WIDTH].iter().all(|v|v.to_f32()==0.5));assert!(x[crate::observe::WIDTH..].iter().all(|v|v.to_f32()==0.));
    let y=tensor::<2>(cache.aux).into_data().try_to_vec::<f32>().unwrap();assert!(y[..crate::policy_trace::AUX].iter().all(|v|*v==7.));assert!(y[crate::policy_trace::AUX..].iter().all(|v|*v== -3.));
}

#[test]
fn selection_preserves_transposed_logits_for_argmax_and_sampling(){
    let _g=GPU.lock().unwrap();let d=device();
    let strided=Tensor::<2>::from_data(TensorData::new(vec![1.,20.,30.,2.,3.,4.],[3,2]),&d).transpose();
    let greedy=Sampler::new(2,411,&d).with_argmax(true);
    assert_eq!(greedy.draw(strided).into_data().try_to_vec::<i32>().unwrap(),vec![1,0]);
    let mut backing=vec![-80f32;3*64];for a in 0..64 {backing[(a%3)*64+a]=80.;}
    let logits=Tensor::<2>::from_data(TensorData::new(backing,[3,64]),&d).transpose();
    let expected:Vec<i32>=(0..64).map(|a|(a%3)as i32).collect();
    assert_eq!(Sampler::new(64,99,&d).with_argmax(true).draw(logits.clone()).into_data().try_to_vec::<i32>().unwrap(),expected);
    assert_eq!(Sampler::new(64,99,&d).draw(logits).into_data().try_to_vec::<i32>().unwrap(),expected);
}

#[test]
fn argmax_selection_masks_ties_and_does_not_consume_sampling_rng(){
    let _g=GPU.lock().unwrap();let d=device();
    let logits=Tensor::<2>::from_data(TensorData::new(vec![9.,4.,4.,-5.,-3.,-4.],[2,3]),&d);
    let mask=Tensor::<2>::from_data(TensorData::new(vec![0.,1.,1.,1.,1.,1.],[2,3]),&d).greater_elem(0.5);
    let greedy=Sampler::new(2,8123,&d).with_argmax(true);
    let out=crate::sampling::distribution(logits.clone(),mask.clone(),None,&greedy).choice.into_data().try_to_vec::<i32>().unwrap();
    assert_eq!(out,vec![1,1]);
    let after=greedy.with_argmax(false);let fresh=Sampler::new(2,8123,&d);
    let one=crate::sampling::distribution(logits.clone(),mask.clone(),None,&after).choice.into_data().try_to_vec::<i32>().unwrap();
    let two=crate::sampling::distribution(logits,mask,None,&fresh).choice.into_data().try_to_vec::<i32>().unwrap();assert_eq!(one,two);
}

#[test]
fn evaluation_prefix_stops_at_arrival_or_deadline_without_faking_a_terminal(){
    let _g=GPU.lock().unwrap();let d=device();let mut w=World::load(&engine(),1,&d).unwrap();
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    for side in 0..2 {state[side*STRIDE+UNIT_X]=0.;state[side*STRIDE+UNIT_Y]=0.;}
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state.clone(),[UNITS*STRIDE]),&d));
    let mut meta=vec![0f32;META];let signals=Tensor::<2>::zeros([2,crate::experience::SIGNALS],&d);
    let result=primitive(Tensor::<2>::zeros([1,crate::evaluation::FIELDS],&d));
    for second in [0,9,10] {
        meta[0]=second as f32*30.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta.clone(),[META]),&d));
        crate::evaluation::record_prefix(&w,signals.clone(),&result,90.,10.);
        let row=tensor::<2>(result.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(row[0],if second==10{1.}else{0.});assert_eq!(row[34],0.);
    }
    let saved=tensor::<2>(result.clone()).into_data().try_to_vec::<f32>().unwrap();meta[0]=3000.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta.clone(),[META]),&d));
    crate::evaluation::record_prefix(&w,signals.clone(),&result,90.,10.);assert_eq!(saved,tensor::<2>(result).into_data().try_to_vec::<f32>().unwrap());
    for side in 0..2 {state[side*STRIDE+UNIT_X]=6000.;state[side*STRIDE+UNIT_Y]=6000.;}
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));meta[0]=2700.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
    let deadline=primitive(Tensor::<2>::zeros([1,crate::evaluation::FIELDS],&d));crate::evaluation::record_prefix(&w,signals,&deadline,90.,10.);
    let row=tensor::<2>(deadline).into_data().try_to_vec::<f32>().unwrap();assert_eq!(row[0],1.);assert_eq!((row[3]*30.).round(),2700.);assert_eq!(row[18],0.);assert_eq!(row[26],0.);assert_eq!(row[34],0.);
    let mut positions=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    for side in 0..2 {positions[side*STRIDE+UNIT_X]=0.;positions[side*STRIDE+UNIT_Y]=0.;}
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(positions,[UNITS*STRIDE]),&d));
    let pregame=primitive(Tensor::<2>::zeros([1,crate::evaluation::FIELDS],&d));
    for tick in [-900.,-600.] {let mut values=vec![0f32;META];values[0]=tick;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(values,[META]),&d));crate::evaluation::record_prefix(&w,Tensor::zeros([2,crate::experience::SIGNALS],&d),&pregame,90.,10.);}
    let row=tensor::<2>(pregame).into_data().try_to_vec::<f32>().unwrap();assert_eq!(row[0],1.);assert_eq!((row[3]*30.).round(),-600.);
}

#[test]
fn evaluation_behavior_counters_keep_pregame_mid_and_require_unblocked_movement(){
    let _g=GPU.lock().unwrap();let d=device();let mut w=World::load(&engine(),1,&d).unwrap();
    let mut meta=vec![0f32;META];meta[0]=-1800.;
    let mut state=vec![0f32;UNITS*STRIDE];state[ACTIVE]=1.;state[UNIT_HEALTH]=582.;
    let result=primitive(Tensor::<2>::zeros([1,crate::evaluation::FIELDS],&d));
    let signals=Tensor::<2>::zeros([2,crate::experience::SIGNALS],&d);
    w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta.clone(),[META]),&d));
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state.clone(),[UNITS*STRIDE]),&d));
    crate::evaluation::record(&w,signals.clone(),&result);
    let first=tensor::<2>(result.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert!((first[18]-w.engine.pre_game_seconds-1.+60.).abs()<0.001);
    state[UNIT_X]=3000.;state[DEST_X]=4000.;state[ORDER]=1.;
    for i in 1..=100 {
        meta[0]=-1800.+i as f32*4.;
        // Casting blocks the first20 samples. The following80 stationary
        // movement samples must count, without counting the cast as stuck.
        state[CAST_END]=if i<=20 {1000.}else{0.};
        w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta.clone(),[META]),&d));
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(state.clone(),[UNITS*STRIDE]),&d));
        crate::evaluation::record(&w,signals.clone(),&result);
    }
    let row=tensor::<2>(result).into_data().try_to_vec::<f32>().unwrap();
    assert!((row[20]-80./7.5).abs()<0.001);assert!((row[21]-80./7.5).abs()<0.001);
    assert_eq!(row[18],first[18]);assert_eq!(row[28],0.);
}

#[test]
fn evaluation_keeps_every_first_terminal_including_draws_and_late_slots(){
    let _g=GPU.lock().unwrap();let d=device();let mut w=World::load(&engine(),4,&d).unwrap();
    let mut meta=vec![0f32;4*META];let mut state=vec![0f32;4*UNITS*STRIDE];
    let mut signals=vec![0f32;8*crate::experience::SIGNALS];
    for m in 0..4 {
        meta[m*META]=3000.+m as f32*30.;meta[m*META+17]=[-1.,0.,1.,1.][m];
        for side in 0..2 {state[(m*UNITS+side)*STRIDE+UNIT_LAST_HITS]=(10*m+side)as f32;}
        signals[(m*2+m%2)*crate::experience::SIGNALS+11]=1.;
        if m<3 {signals[m*2*crate::experience::SIGNALS+9]=1.;}
    }
    w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[4*META]),&d));
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[4*UNITS*STRIDE]),&d));
    let result=primitive(Tensor::<2>::zeros([4,crate::evaluation::FIELDS],&d));
    crate::evaluation::record(&w,Tensor::from_data(TensorData::new(signals.clone(),[8,crate::experience::SIGNALS]),&d),&result);
    let first=tensor::<2>(result.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(first[3*crate::evaluation::FIELDS],0.);
    // Completed slots now represent later episodes; their saved results must survive.
    w.meta=primitive(Tensor::<1>::zeros([4*META],&d));
    signals[6*crate::experience::SIGNALS+9]=1.;
    crate::evaluation::record(&w,Tensor::from_data(TensorData::new(signals,[8,crate::experience::SIGNALS]),&d),&result);
    let last=tensor::<2>(result).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(&last[..3*crate::evaluation::FIELDS],&first[..3*crate::evaluation::FIELDS]);
    for m in 0..4 {let row=&last[m*crate::evaluation::FIELDS..(m+1)*crate::evaluation::FIELDS];
        assert_eq!(row[0],1.);assert_eq!(row[2],(m%2)as f32);assert_eq!(row[4],(10*m)as f32);assert_eq!(row[5],(10*m+1)as f32);
    }
    assert_eq!(last[crate::evaluation::FIELDS+1],0.);assert_eq!(last[3*crate::evaluation::FIELDS+1],0.);
}

#[test]
fn frozen_opponent_metrics_count_only_one_terminal_and_the_correct_learner_side(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();
    let mut meta=vec![0f32;META];meta[0]=3000.;meta[17]=-1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
    let mut signals=vec![0f32;2*crate::experience::SIGNALS];signals[9]=1.;signals[crate::experience::SIGNALS+9]=1.;signals[crate::experience::SIGNALS+11]=1.;
    crate::selfplay::record_outcomes(&w,Tensor::from_data(TensorData::new(signals.clone(),[2,crate::experience::SIGNALS]),&d));
    signals[9]=0.;signals[crate::experience::SIGNALS+9]=0.;
    crate::selfplay::record_outcomes(&w,Tensor::from_data(TensorData::new(signals,[2,crate::experience::SIGNALS]),&d));
    let values=tensor::<1>(w.meta.clone()).slice([18..32]).into_data().try_to_vec::<f32>().unwrap();let report=crate::selfplay::outcome_report(values);
    assert_eq!(report["games"],1.);assert_eq!(report["wins"],1.);assert_eq!(report["learner_dire_games"],1.);assert_eq!(report["learner_dire_wins"],1.);assert!((report["mean_game_seconds"].as_f64().unwrap()-100.).abs()<0.00002);
}

#[test]
fn a_later_overkill_projectile_cannot_steal_last_hit_or_turn_it_into_deny(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();let p=6*STRIDE;
    s[p+UNIT_HEALTH]=1.;s[p+UNIT_HEALTH_MAX]=550.;s[p+UNIT_X]=0.;s[p+UNIT_Y]=0.;s[p+ACTIVE]=1.;s[p+TEAM]=1.;s[p+KIND]=2.;s[p+ATTACK_READY]=1000.;s[p+TARGET]=-1.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let mut shots=vec![0f32;SHOTS*SHOT_FIELDS];
    for source in 0..2 {let k=source*SHOT_FIELDS;shots[k]=1.;shots[k+3]=6.;shots[k+5]=1200.;shots[k+6]=50.;shots[k+7]=source as f32;shots[k+8]=source as f32;}
    w.shots=primitive(Tensor::<1>::from_data(TensorData::new(shots,[SHOTS*SHOT_FIELDS]),&d));let _=w.step_ticks(&rng,1);
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[UNIT_LAST_HITS],1.);assert_eq!(s[STRIDE+UNIT_DENIES],0.);
}

#[test]
fn all_towers_are_resident_background_towers_attack_and_do_not_fill_creep_slots(){
    let _g=GPU.lock().unwrap();let d=device();let w=World::load(&engine(),1,&d).unwrap();
    let initial=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(initial.chunks_exact(STRIDE).filter(|s|s[ACTIVE]>0. && s[KIND]==1.).count(),24);
    for side in 0..2 {for local in 0..9 {let p=(BACKGROUND_START+side*9+local)*STRIDE;
        assert_eq!(initial[p+TEAM],side as f32);assert_eq!(initial[p+STRUCTURE_PARENT],if local<6 {if local%3==0 {-1.}else{(BACKGROUND_START+side*9+local-1) as f32}}else if local==6 {if side==0 {3.}else{5.}}else{-2.});
    }}
    let mut w=arena();let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();let tower=BACKGROUND_START*STRIDE;
    s[tower..tower+STRIDE].copy_from_slice(&initial[tower..tower+STRIDE]);
    s[tower+UNIT_X]=0.;s[tower+UNIT_Y]=0.;s[tower+UNIT_Z]=0.;
    s[UNIT_X]=-4000.;s[UNIT_Y]=0.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let obs=w.observe(initial_previous(2,&d));let decoded=obs.decode();
    assert_eq!(decoded.creep_valid.int().sum().into_scalar::<i32>(),0);
    assert_ne!(decoded.hero_valid.int().into_data().try_to_vec::<i32>().unwrap()[1],0,"background tower must share vision with its hero");
    let rng=Sampler::new(1,7,&d);let _=w.step_ticks(&rng,65);assert!(hp(&w,1)<500.,"background tower did not damage invading hero");
}

#[test]
fn tango_walks_to_tree_and_cut_opens_visibility(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    w.tree_count=1;w.trees=primitive(Tensor::<1>::from_data([0.,0.,0.],&d));w.tree_alive=primitive(Tensor::<1>::ones([1],&d));
    w.tree_map=primitive(Tensor::<1>::ones([512*512],&d));
    // This artificial flat arena has no real-map navigation detours.
    w.navigation=primitive(Tensor::<1>::full([crate::navigation::GOALS*crate::navigation::GOALS*crate::navigation::WORDS_PER_GOAL],f32::from_bits(0x88888888),&d));
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();state[ITEM_START+6]=1.;state[SLOT_START+6]=1.;
    // Observe a front-facing fog cell, with the enemy behind the tree. A tree
    // centre queried from its shadow side need not itself be visible in Dota.
    state[UNIT_X]=1000.;state[UNIT_Y]=1000.;state[FACING]=-2.3561945;
    state[STRIDE+UNIT_X]=-200.;state[STRIDE+UNIT_Y]=-200.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
    let mut meta=vec![0.;META];meta[0]=1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
    assert_eq!(w.observe(initial_previous(2,&d)).decode().tree_valid.int().sum().into_scalar::<i32>(),1);
    assert_eq!(w.observe(initial_previous(2,&d)).decode().hero_valid.int().into_data().try_to_vec::<i32>().unwrap()[1],0);
    action_command(&w,&rng,[10,0,6,(2+crate::observe::CREEPS+4) as i32,64,64]);let _=w.step_ticks(&rng,140);
    assert_eq!(tensor::<1>(w.tree_alive.clone()).into_scalar::<f32>(),0.);
    assert_ne!(w.observe(initial_previous(2,&d)).decode().hero_valid.int().into_data().try_to_vec::<i32>().unwrap()[1],0);
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[ITEM_START+6],0.);assert!(s[TANGO_END]>0.);
}

#[test]
fn teleport_uses_home_reference_clamps_destination_and_keeps_cooldown(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();state[ITEM_START+3]=2.;state[SLOT_START+3]=15.;state[ITEM_READY+3]=0.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
    action_command(&w,&rng,[10,0,3,(crate::observe::TARGETS-1) as i32,128,128]);let _=w.step_ticks(&rng,110);
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();let home=&w.engine.bases[0];
    let distance=((s[UNIT_X]-home.x).powi(2)+(s[UNIT_Y]-home.y).powi(2)).sqrt();
    assert!(distance<=800.5 && distance>=69.5,"TP destination outside home bounds: {distance}");
    assert_eq!(s[ITEM_START+3],1.);assert_eq!(s[TP_END],0.);assert!(s[ITEM_READY+3]>100.);
}

#[test]
fn recycled_target_cannot_receive_a_queued_old_attack(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();let p=6*STRIDE;
    state[p+UNIT_X]=400.;state[p+UNIT_HEALTH]=550.;state[p+UNIT_HEALTH_MAX]=550.;state[p+ACTIVE]=1.;state[p+KIND]=2.;state[p+TEAM]=1.;state[p+GENERATION]=11.;state[p+TARGET]=-1.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
    let mut meta=vec![0f32;META];meta[0]=1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
    action_command(&w,&rng,[3,0,0,2,64,64]);
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();state[p+GENERATION]=12.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));let _=w.step_ticks(&rng,28);
    assert_eq!(hp(&w,6),550.);
}

#[test]
fn creep_kill_updates_level_and_stats_before_next_observation(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();let p=6*STRIDE;
    state[XP]=239.;state[p+UNIT_X]=0.;state[p+UNIT_HEALTH]=1.;state[p+UNIT_HEALTH_MAX]=550.;state[p+ACTIVE]=1.;state[p+KIND]=2.;state[p+TEAM]=1.;state[p+TARGET]=-1.;
    state[p+DAMAGE]=2.;state[p+CREDIT]=0.;state[p+CREDIT_TEAM]=0.;state[p+XP_BOUNTY]=57.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
    let mut meta=vec![0f32;META];meta[0]=1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));let _=w.step_ticks(&rng,1);
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(s[UNIT_LEVEL],2.);assert_eq!(s[UNIT_XP_TO_NEXT],344.);assert_eq!(s[UNIT_SOULS],1.);
    assert_eq!(s[UNIT_HEALTH_MAX],626.); // Native level2,Wraith Band,whole total strength.
}

#[test]
fn large_snapshot_copy_crosses_vulkan_dispatch_axis_limit(){
    let _g=GPU.lock().unwrap();let d=device();let words=65536*256;
    let input=primitive(Tensor::<1>::full([words],3.25,&d));
    let output=crate::artifacts::copy(&input);
    assert_eq!((tensor::<1>(output)-3.25).abs().max().into_scalar::<f32>(),0.);
}

#[test]
fn experience_is_zeroed_fp16_and_preserves_rollout_write_read(){
    let _g=GPU.lock().unwrap();let d=device();let e=crate::experience::Experience::new(2,&d);
    assert_eq!(e.observations[0].dtype,burn::tensor::DType::F16);
    assert_eq!(tensor::<3>(e.observations[0].clone()).dims(),[2,128,crate::observe::WIDTH]);
    let read=||tensor::<3>(e.observations[0].clone());
    assert_eq!(read().cast(burn::tensor::DType::F32).abs().max().into_scalar::<f32>(),0.);
    let observed=crate::observe::Observed{packed:Tensor::ones([2,crate::observe::WIDTH],(&d,burn::tensor::DType::F16)),
        refs:e.observations[0].clone(),reset:Tensor::zeros([2],&d)};
    e.store(127,&observed,Tensor::zeros([2,crate::experience::SIGNALS],&d));
    let tail=read().slice([0..2,127..128,0..crate::observe::WIDTH]).cast(burn::tensor::DType::F32);
    assert_eq!(tail.clone().min().into_scalar::<f32>(),1.);assert_eq!(tail.max().into_scalar::<f32>(),1.);
    assert_eq!(read().slice([0..2,0..127,0..crate::observe::WIDTH]).cast(burn::tensor::DType::F32).max().into_scalar::<f32>(),0.);
}

#[test]
fn fused_cell_forward_and_both_derivatives(){let _g=GPU.lock().unwrap();crate::checks::cell_gradient().unwrap();}

#[test]
fn negative_dota_time_does_not_enable_inactive_fear_or_cooldowns(){
    let _g=GPU.lock().unwrap();let d=device();let w=World::load(&engine(),1,&d).unwrap();
    let b=w.observe(initial_previous(2,&d)).decode();
    let mask=b.action_mask.int().into_data().try_to_vec::<i32>().unwrap();
    assert!(mask[1]!=0 && mask[8]!=0,"pregame movement and learned raze must be available");
    let heroes=b.heroes.into_data().try_to_vec::<f32>().unwrap();assert_eq!(heroes[UNIT_FEAR],0.);
}

#[test]
fn gae_stops_at_terminal_and_bootstraps_rollout_boundary(){
    let _g=GPU.lock().unwrap();let d=device();let mut experience=crate::experience::Experience::new(2,&d);
    let mut signals=vec![0f32;2*128*crate::experience::SIGNALS];let stride=crate::experience::SIGNALS;
    signals[8]=1.;signals[stride+8]=2.;signals[stride+9]=1.;
    signals[(128+127)*stride+8]=3.;
    experience.signals=primitive(Tensor::<3>::from_data(TensorData::new(signals,[2,128,stride]),&d));
    let (adv,ret)=experience.gae(Tensor::from_data([0.,5.],&d));
    let a=adv.into_data().try_to_vec::<f32>().unwrap();let r=ret.into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(a[0],3.);assert_eq!(a[1],2.);assert_eq!(a[2],0.);assert_eq!(a[255],8.);assert_eq!(a,r);
}

#[test]
fn absent_target_never_becomes_unit_zero_on_gpu(){
    let _g=GPU.lock().unwrap();let w=arena();let rng=Sampler::new(1,7,&w.device);
    for _ in 0..20 {command(&w,&rng,0);}
    assert!((hp(&w,0)-582.).abs()<0.001 && (hp(&w,1)-582.).abs()<0.001);
    let meta=tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(meta[8],0.,"idle heroes released a tracking projectile");
}

#[test]
fn cancel_before_release_prevents_hit_but_after_release_keeps_projectile(){
    let _g=GPU.lock().unwrap();
    let first=arena();let rng=Sampler::new(1,7,&first.device);
    command(&first,&rng,3);command(&first,&rng,0);command(&first,&rng,4);
    for _ in 0..10 {command(&first,&rng,0);}
    assert!((hp(&first,1)-582.).abs()<0.001,"cancelled attack dealt damage");
    let second=arena();let rng=Sampler::new(1,7,&second.device);
    command(&second,&rng,3);for _ in 0..4 {command(&second,&rng,0);}command(&second,&rng,4);
    for _ in 0..10 {command(&second,&rng,0);}
    assert!(hp(&second,1)<562.,"released projectile was cancelled");
}

#[test]
fn reissuing_same_raze_does_not_restart_cast(){
    let _g=GPU.lock().unwrap();let w=arena();let rng=Sampler::new(1,7,&w.device);
    command(&w,&rng,7);command(&w,&rng,7);command(&w,&rng,7);
    let _=w.step_ticks(&rng,16);
    assert!(hp(&w,1)<540.,"reissued raze never completed its original cast");
}

#[test]
fn reissuing_same_attack_target_preserves_windup(){
    let _g=GPU.lock().unwrap();let w=arena();let rng=Sampler::new(1,7,&w.device);
    command(&w,&rng,3);command(&w,&rng,3);for _ in 0..3 {command(&w,&rng,0);}
    let active=tensor::<1>(w.shots.clone()).reshape([SHOTS,SHOT_FIELDS]).slice([0..SHOTS,0..1]).sum().into_scalar::<f32>();
    assert!(active>0.,"a repeated order restarted attack windup");
}

#[test]
fn simultaneous_death_is_draw_and_reset_restores_both_heroes(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    for u in 0..2 {state[u*STRIDE+UNIT_HEALTH]=1.;state[u*STRIDE+DAMAGE]=10.;state[u*STRIDE+CREDIT]=(1-u) as f32;}
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
    let obs=w.observe(initial_previous(2,&d));let rng=Sampler::new(1,7,&d);
    let (reward,done)=w.advance(Tensor::<2,Int>::zeros([2,6],&d),&obs.refs,&rng);
    assert_eq!(done.into_data().try_to_vec::<f32>().unwrap(),vec![1.,1.]);
    assert!(reward.into_data().try_to_vec::<f32>().unwrap().iter().all(|r|r.abs()<1e-6));
    let next=w.observe(initial_previous(2,&d));
    assert_eq!(next.reset.into_data().try_to_vec::<f32>().unwrap(),vec![1.,1.]);
    assert!(hp(&w,0)>500. && hp(&w,1)>500.);
}

#[test]
fn pregame_death_respawns_after_twelve_seconds_and_is_remembered_at_time_zero(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    s[UNIT_HEALTH]=1.;s[UNIT_MANA]=7.;s[DAMAGE]=10.;s[CREDIT]=1.;s[UNIT_SOULS]=4.;s[UNIT_GOLD]=421.;
    s[CAST_END]=1000.;s[CAST]=3.;s[FEAR_END]=1000.;s[TANGO_END]=1000.;s[COOLDOWNS+2]=200.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let mut meta=vec![0.;META];meta[0]=-900.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
    let (_,done)=w.step_ticks(&rng,1);assert_eq!(done.sum().into_scalar::<f32>(),0.);
    let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(s[UNIT_HEALTH],0.);assert_eq!(s[ACTIVE],0.);assert_eq!(s[DEATHS],1.);assert_eq!(s[GENERATION],1.);
    assert_eq!(s[UNIT_GOLD],398.);assert_eq!(s[UNIT_SOULS],3.);assert_eq!(s[CAST_END],0.);assert_eq!(s[FEAR_END],0.);assert_eq!(s[TANGO_END],0.);
    let obs=w.observe(initial_previous(2,&d));assert_eq!(obs.reset.clone().into_data().try_to_vec::<f32>().unwrap(),vec![1.,0.]);
    let b=obs.decode();let mask=b.action_mask.int().into_data().try_to_vec::<i32>().unwrap();assert_eq!(&mask[..11],&[1,0,0,0,0,0,0,0,0,0,0]);
    assert_eq!(&b.buy_mask.int().into_data().try_to_vec::<i32>().unwrap()[..7],&[1,0,0,0,0,0,0]);
    assert_eq!(b.hero_valid.int().into_data().try_to_vec::<i32>().unwrap()[3],0,"enemy cannot observe a dead hero");
    let _=w.step_ticks(&rng,359);assert_eq!(hp(&w,0),0.);
    let _=w.step_ticks(&rng,1);let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(s[ACTIVE],1.);assert_eq!(s[UNIT_HEALTH],582.);assert_eq!(s[UNIT_MANA],291.);assert_eq!(s[COOLDOWNS+2],200.);
    assert_eq!(s[UNIT_X],w.engine.hero[33]);assert_eq!(s[UNIT_Y],w.engine.hero[34]);assert!((s[FACING]-std::f32::consts::FRAC_PI_4).abs()<0.0001);
    let obs=w.observe(Tensor::ones([2,PREVIOUS],&d));assert_eq!(obs.reset.clone().into_data().try_to_vec::<f32>().unwrap(),vec![1.,0.]);
    assert_eq!(obs.decode().previous.slice([0..1]).into_data().try_to_vec::<f32>().unwrap(),initial_previous(1,&d).into_data().try_to_vec::<f32>().unwrap());
    let _=w.step_ticks(&rng,1);assert_eq!(w.observe(initial_previous(2,&d)).reset.sum().into_scalar::<f32>(),0.);
    let mut meta=tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap();meta[0]=-1.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
    let (_,done)=w.step_ticks(&rng,2);assert_eq!(done.sum().into_scalar::<f32>(),2.);assert_eq!(hp(&w,0),582.);
    assert_eq!(tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap()[17],-1.,"a pregame death still loses after respawn");
}

#[test]
fn level_one_kill_economy_matches_both_native_sides_and_protects_reliable_gold(){
    let _g=GPU.lock().unwrap();
    for victim in 0..2 {for case in 0..4 {
        let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);let p=victim*STRIDE;let q=(1-victim)*STRIDE;
        let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        s[p+UNIT_HEALTH]=1.;s[p+DAMAGE]=10.;s[p+CREDIT]=if case==0 {(1-victim)as f32}else{if victim==0{4.}else{2.}};
        if case==2 {s[p+UNIT_GOLD]=421.;s[p+UNIT_SOULS]=4.;s[p+XP]=100.;s[q+UNIT_GOLD]=80.;s[q+DEATHS]=1.;s[q+UNIT_X]=7000.;s[q+UNIT_Y]=7000.;}
        if case==3 {s[p+RELIABLE_GOLD]=90.;}
        w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
        let mut meta=vec![0.;META];meta[0]=-900.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));let _=w.step_ticks(&rng,1);
        let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(s[p+UNIT_GOLD],if case==2 {398.}else if case==3 {90.}else{80.});
        assert_eq!(s[q+UNIT_GOLD],if case==0 {421.}else if case==2 {213.}else{297.});
        assert_eq!(s[q+XP],if case==2 {0.}else{100.});assert_eq!(s[q+UNIT_SOULS],if case==0 {4.}else{0.});
    }}
}

#[test]
fn dead_hero_cannot_gain_nearby_creep_xp_or_be_revived_by_item_stats(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    s[UNIT_HEALTH]=0.;s[ACTIVE]=0.;s[DEATHS]=1.;s[ITEM_START+2]=1.;s[RESPAWN_AT]=1000.;s[XP]=239.;
    let p=6*STRIDE;s[p+UNIT_X]=0.;s[p+UNIT_HEALTH]=1.;s[p+UNIT_HEALTH_MAX]=550.;s[p+ACTIVE]=1.;s[p+KIND]=2.;s[p+TEAM]=1.;s[p+TARGET]=-1.;s[p+DAMAGE]=2.;s[p+CREDIT]=-1.;s[p+XP_BOUNTY]=57.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let mut meta=vec![0.;META];meta[0]=-900.;w.meta=primitive(Tensor::<1>::from_data(TensorData::new(meta,[META]),&d));
    let _=w.step_ticks(&rng,1);let s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(s[UNIT_HEALTH],0.);assert_eq!(s[ACTIVE],0.);assert_eq!(s[XP],239.);assert_eq!(s[UNIT_LEVEL],1.);assert!(s[UNIT_HEALTH_MAX]>582.);
}

#[test]
fn hidden_enemy_resources_and_position_do_not_enter_observations(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    state[STRIDE+UNIT_X]=7000.;state[STRIDE+UNIT_Y]=7000.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state.clone(),[UNITS*STRIDE]),&d));
    let first=w.observe(initial_previous(2,&d)).packed.slice([0..1]).into_data().try_to_vec::<f32>().unwrap();
    state[STRIDE+UNIT_X]=6000.;state[STRIDE+UNIT_GOLD]=9999.;state[STRIDE+UNIT_SOULS]=19.;state[STRIDE+UNIT_HEALTH]=1.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
    let second=w.observe(initial_previous(2,&d)).packed.slice([0..1]).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(first,second);
}

#[test]
fn categorical_samples_only_legal_actions_with_reproducible_per_actor_rng(){
    let _g=GPU.lock().unwrap();let d=device();let a=Sampler::new(64,17,&d);let b=Sampler::new(64,17,&d);
    let logits=Tensor::<2>::zeros([64,11],&d).slice_assign([0..64,0..10],Tensor::full([64,10],-1e30,&d));
    for _ in 0..4 {
        let x=a.draw(logits.clone()).into_data().try_to_vec::<i32>().unwrap();let y=b.draw(logits.clone()).into_data().try_to_vec::<i32>().unwrap();
        assert_eq!(x,y);assert!(x.iter().all(|x|*x==10));
    }
}

#[test]
fn scripted_teacher_lh_deny_competition_and_delay_work_on_both_sides(){
    let _g=GPU.lock().unwrap();
    let output=std::env::temp_dir().join(format!("sf-scripted-teacher-{}",std::process::id()));
    let result=crate::scripted_teacher::check(&engine(),&output).unwrap();
    assert_eq!(result["passed"],true);
    assert_eq!(result["cases"],16);
}

#[test]
fn ggd_relations_bind_attackers_and_projectiles_without_changing_the_world(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    for (u,kind,side,x) in [(6usize,2usize,1.,0.),(7,2,0.,-80.),(8,3,0.,-300.)] {
        let p=u*STRIDE;s[p..p+STRIDE].fill(0.);s[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[kind+4]);
        s[p+ACTIVE]=1.;s[p+KIND]=kind as f32;s[p+TEAM]=side;s[p+GENERATION]=1.;s[p+UNIT_X]=x;s[p+UNIT_Y]=0.;s[p+TARGET]=-1.;
    }
    for u in [7usize,8]{s[u*STRIDE+TARGET]=6.;s[u*STRIDE+TARGET_GENERATION]=1.;}
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let mut shots=vec![0.;SHOTS*SHOT_FIELDS];shots[..10].copy_from_slice(&[1.,-150.,0.,6.,1.,900.,99999.,8.,0.,3.]);
    w.shots=primitive(Tensor::<1>::from_data(TensorData::new(shots.clone(),[SHOTS*SHOT_FIELDS]),&d));
    let observed=w.observe(initial_previous(2,&d));let before=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    let memory=crate::combat_observe::CombatObserver::new(&w);let extended=memory.enrich(&w,&observed);
    let actual=extended.packed.clone().into_data().try_to_vec::<f32>().unwrap();
    let at=crate::observe::COMBAT_UNIT_OFFSET+2*2*COMBAT_UNIT_COUNT;
    assert_eq!(actual[at+COMBAT_UNIT_INCOMING_ATTACKERS]*16.,2.);
    assert_eq!(actual[at+COMBAT_UNIT_INCOMING_PROJECTILES]*16.,1.);
    assert!(actual[at+COMBAT_UNIT_INCOMING_ATTACK_DAMAGE]>0.);
    assert_eq!(actual[at+COMBAT_UNIT_COUNT+COMBAT_UNIT_INCOMING_PROJECTILE_DAMAGE],1.);
    let target=crate::observe::COMBAT_UNIT_OFFSET+3*2*COMBAT_UNIT_COUNT;
    assert!((actual[target+COMBAT_UNIT_ATTACK_TARGET_DX]*2048.-200.).abs()<0.001);
    assert!((actual[crate::observe::COMBAT_PROJECTILE_OFFSET+COMBAT_PROJECTILE_CASTER_DAMAGE]*200.-23.).abs()<0.001);
    assert_eq!(before,tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap());
    shots[6]=-99999.;w.shots=primitive(Tensor::<1>::from_data(TensorData::new(shots,[SHOTS*SHOT_FIELDS]),&d));
    assert_eq!(actual,memory.enrich(&w,&observed).packed.into_data().try_to_vec::<f32>().unwrap(),"rolled projectile damage must not be an input");
    let policy=crate::model::Policy::new_ggd(&d);let b=extended.decode_bucket(16,32,16);let encoded=policy.encode(&b);
    assert_eq!(policy.hero.first.weight.val().dims(),[118,128]);assert_eq!(policy.unit.first.weight.val().dims(),[118,64]);assert_eq!(policy.projectile.first.weight.val().dims(),[28,32]);
    assert_eq!(encoded.fused.dims(),[2,256]);assert!(encoded.fused.into_data().try_to_vec::<f32>().unwrap().iter().all(|v|v.is_finite()));
    assert!(!crate::model::Policy::new(&d).is_ggd());
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();s[8*STRIDE+ACTIVE]=0.;s[8*STRIDE+UNIT_HEALTH]=0.;
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));let obs=w.observe(initial_previous(2,&d));
    let cached=memory.enrich(&w,&obs).packed.into_data().try_to_vec::<f32>().unwrap();
    let fresh=crate::combat_observe::CombatObserver::new(&w).enrich(&w,&obs).packed.into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(cached[at+COMBAT_UNIT_COUNT+COMBAT_UNIT_INCOMING_PROJECTILE_DAMAGE],1.);
    assert_eq!(fresh[at+COMBAT_UNIT_COUNT+COMBAT_UNIT_INCOMING_PROJECTILE_DAMAGE],0.);
}

#[test]
fn ggd_metrics_count_actual_cast_hits_death_and_freeze_at_terminal(){
    let _g=GPU.lock().unwrap();let mut w=arena();let d=w.device.clone();let rng=Sampler::new(1,20261103,&d);
    let mut state=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    state[UNIT_LEVEL]=7.;state[UNIT_MANA]=1000.;state[STRIDE+UNIT_HEALTH]=20.;
    state[STRIDE+UNIT_X]=state[UNIT_X]+w.engine.spells[2][0][3];
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(state,[UNITS*STRIDE]),&d));
    let teacher=crate::scripted_teacher::Teacher::new(&w);let metrics=crate::ggd_metrics::Metrics::new(&w);metrics.sample(&w,&teacher);
    for step in 0..20 {
        let obs=w.observe(initial_previous(2,&d));let actions=Tensor::<2,Int>::from_data([[if step==0{8}else{0},0,0,0,64,64],[0,0,0,0,64,64]],&d);
        metrics.commands(&w,&teacher,actions.clone(),&obs.refs);let _=w.advance_teacher(actions,&obs.refs,&rng,&teacher,false);teacher.record(&w,600.);metrics.sample(&w,&teacher);
    }
    let games=metrics.games(&teacher.values(),w.engine.pre_game_seconds,0,Some("mirror"));let radiant=&games[0]["sides"][0];let dire=&games[0]["sides"][1];
    assert_eq!(radiant["actions"][8],1.);assert_eq!(radiant["spell_completions"][2],1.);assert_eq!(radiant["raze_hero_hits"],1.);assert_eq!(radiant["razes_hitting_hero"],1.);
    assert_eq!(dire["deaths"],1.);assert!(dire["first_death_game_seconds"].as_f64().unwrap()>0.);assert_eq!(games[0]["native_terminal"],true);
    let before=tensor::<1>(metrics.buffer.clone()).into_data().try_to_vec::<f32>().unwrap();metrics.sample(&w,&teacher);
    assert_eq!(before,tensor::<1>(metrics.buffer.clone()).into_data().try_to_vec::<f32>().unwrap());
    assert_eq!(crate::ggd_metrics::summary(&games)["sides"].as_array().unwrap().len(),2);
}




#[path="teacher_v10_tests.rs"] mod teacher_v10_tests;
