use super::*;
use crate::teacher_v10::{Controller,Version};

fn decide_case(case:usize)->(Vec<i32>,World,crate::scripted_teacher::Teacher){
    let mut w=arena();let d=w.device.clone();
    let mut s=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
    let now=w.engine.pre_game_seconds;
    if matches!(case,1|2|3|4|5) {s[UNIT_MANA]=0.;}
    if case==2 || case==6 || case==7 {
        let p=6*STRIDE;
        s[p..p+UNIT_COUNT].copy_from_slice(&w.engine.prototypes[6]);
        s[p+ACTIVE]=1.;s[p+GENERATION]=1.;s[p+TEAM]=1.;s[p+KIND]=2.;s[p+UNIT_X]=50.;s[p+UNIT_Y]=0.;s[p+UNIT_Z]=0.;
        s[p+UNIT_HEALTH]=if case==2{500.}else{30.};s[p+TARGET]=-1.;s[p+WINDUP_TARGET]=-1.;s[p+LANE_POSITION]=-1.;
        if case==7 {s[WINDUP]=now+0.4;s[WINDUP_TARGET]=6.;s[WINDUP_GENERATION]=1.;s[ORDER]=3.;s[TARGET]=6.;s[TARGET_GENERATION]=1.;}
    }
    if case==3 {
        s[UNIT_HEALTH]=s[UNIT_HEALTH_MAX]*0.6;
        let mut events=vec![0.;2*crate::teacher_combat::FIELDS];events[crate::teacher_combat::LAST_HERO_DAMAGE]=now+1.;
        w.teacher_combat=primitive(Tensor::<1>::from_data(TensorData::new(events,[2*crate::teacher_combat::FIELDS]),&d));
    }
    if case==4 {let p=4*STRIDE;s[p+ACTIVE]=1.;s[p+UNIT_X]=100.;s[p+UNIT_Y]=0.;s[p+TARGET]=0.;}
    if case==5 {s[UNIT_HEALTH]=s[UNIT_HEALTH_MAX]*0.2;}
    // Rule 1 may spend the last raze, whereas rule 2 must keep one in reserve.
    if case==6 {s[UNIT_MANA]=w.engine.spells[0][0][1];}
    if case==8 {s[UNIT_MANA]=w.engine.spells[0][0][1];}
    if case==9 {s[CAST]=2.;s[CAST_END]=now+0.3;}
    w.state=primitive(Tensor::<1>::from_data(TensorData::new(s,[UNITS*STRIDE]),&d));
    let t=crate::scripted_teacher::Teacher::new(&w);let controller=Controller::new(&w,Version::V10,Version::V9,false);
    let obs=w.observe(initial_previous(2,&d));let actions=controller.actions(&t,&w,&obs).into_data().try_to_vec::<i32>().unwrap();
    (actions,w,t)
}

#[test]
fn v10_rules_cast_reserve_farm_lock_aggro_trade_tower_and_hysteresis(){
    let _g=GPU.lock().unwrap();crate::teacher_v10::check_pin().unwrap();
    for case in 0..10 {let (a,w,_)=decide_case(case);
        match case {
            0|6=>assert!((6..=8).contains(&a[0]),"case {case}: {a:?}"),
            1=>{assert_eq!(a[0],3,"{a:?}");assert_eq!(a[3],1);},
            2=>assert!(a[0]!=3 || a[3]!=1,"creep aggro guard: {a:?}"),
            3|4=>{assert_eq!(a[0],1,"case {case}: {a:?}");assert!(a[4]<64,"must retreat away: {a:?}");},
            5=>assert!(!(6..=8).contains(&a[0]) && (a[0]!=3 || a[3]!=1),"v9 retreat: {a:?}"),
            7=>assert!(!(6..=8).contains(&a[0]),"useful farm windup interrupted: {a:?}"),
            8=>assert!(!(6..=8).contains(&a[0]),"mana reserve: {a:?}"),
            9=>assert_eq!(a[0],0,"casting must finish: {a:?}"),
            _=>unreachable!(),
        }
        let v9=crate::scripted_teacher::Teacher::new(&w);let obs=w.observe(initial_previous(2,&w.device));
        let expected=v9.actions(&w,&obs).into_data().try_to_vec::<i32>().unwrap();
        assert_eq!(&a[6..],&expected[6..],"v9 side changed in case {case}");
    }
}

#[test]
fn v10_dagger_override_clears_speculative_attack_and_equal_is_noop(){
    let _g=GPU.lock().unwrap();let (_,w,t)=decide_case(1);let d=w.device.clone();
    let controller=Controller::new(&w,Version::V10,Version::V10,false);let obs=w.observe(initial_previous(2,&d));
    let before=t.command_snapshot();let labels=controller.actions(&t,&w,&obs);
    let memory=tensor::<1>(t.memory.clone()).into_data().try_to_vec::<f32>().unwrap();
    t.reconcile_commands(&w,&obs,before,labels.clone(),labels.clone());
    assert_eq!(tensor::<1>(t.memory.clone()).into_data().try_to_vec::<f32>().unwrap(),memory);
    let before=t.command_snapshot();let labels=controller.actions(&t,&w,&obs);
    let stop=Tensor::<2,Int>::from_data([[4,0,0,0,64,64];2],&d);
    t.reconcile_commands(&w,&obs,before,labels,stop);
    let memory=tensor::<1>(t.memory.clone()).into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(memory[1],-1.);assert_eq!(memory[12],4.);
}

#[test]
fn v10_combat_counters_count_execution_and_actual_damage(){
    let _g=GPU.lock().unwrap();let (_,w,t)=decide_case(2);let d=w.device.clone();let rng=Sampler::new(1,7,&d);
    let obs=w.observe(initial_previous(2,&d));let attack=Tensor::<2,Int>::from_data([[3,0,0,1,64,64],[0,0,0,0,64,64]],&d);
    let _=w.advance_teacher(attack,&obs.refs,&rng,&t,false);
    assert!(tensor::<1>(w.teacher_combat.clone()).into_data().try_to_vec::<f32>().unwrap().iter().all(|v|*v==0.),"delayed command cannot count immediately");
    for _ in 0..12 {let _=w.advance_teacher(Tensor::<2,Int>::from_data([[0,0,0,0,64,64];2],&d),&obs.refs,&rng,&t,false);}
    let events=tensor::<1>(w.teacher_combat.clone()).into_data().try_to_vec::<f32>().unwrap();
    use crate::teacher_combat::*;
    assert_eq!(events[HERO_ORDERS],1.);assert_eq!(events[AGGRO_ORDERS],1.);
    assert!(events[HERO_ATTACKS]>=1.,"{events:?}");assert!(events[FIELDS+HERO_DAMAGE]>0.,"{events:?}");
}
