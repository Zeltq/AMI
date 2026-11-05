//! Independent arena lifecycle. Production physics/observation code is reused.
use burn::tensor::{Tensor,TensorData,Bool};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use crate::{world::*,runtime::{primitive,tensor,argument}};

#[cube(launch)]
fn reset_state(state:&mut [f32],meta:&mut [f32],prototypes:&[f32],mask:&[f32],matches:usize,pregame:f32){
    let m=ABSOLUTE_POS;if m<matches && mask[m]>0.0f32 {
        let generation=meta[m*META+1]+1.0f32;
        initialize(state,prototypes,m);
        for f in 0..META {meta[m*META+f]=0.0f32;}
        meta[m*META]=-pregame*30.0f32;meta[m*META+1]=generation;
    }
}
#[cube(launch)]
fn fill_selected(data:&mut [f32],mask:&[f32],words:usize,width:usize,value:f32){
    let i=ABSOLUTE_POS;if i<words && mask[i/width]>0.0f32 {data[i]=value;}
}
#[cube(launch)]
fn hold(audit:&mut [f32],matches:usize){let m=ABSOLUTE_POS;if m<matches{audit[m*crate::scripted_teacher::AUDIT_FIELDS]=1.0f32;}}

pub fn hold_all(t:&crate::scripted_teacher::Teacher,matches:usize){
    hold::launch(&t.audit.client,CubeCount::Static(matches.div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&t.audit),matches);
}
fn fill(data:&CubeTensor,mask:&CubeTensor,matches:usize,value:f32){
    let words=data.meta.shape().iter().product::<usize>();assert_eq!(words%matches,0);
    fill_selected::launch(&data.client,CubeCount::Static(words.div_ceil(256)as u32,1,1),CubeDim::new_1d(256),argument(data),argument(mask),words,words/matches,value);
}
/// A routed teacher has independent speculative caches but shares terminal audit.
pub fn reset_teacher_cache(t:&crate::scripted_teacher::Teacher,selected:&[bool],d:&burn::tensor::Device){
    let mask=primitive(Tensor::<1>::from_data(TensorData::new(selected.iter().map(|x|if *x{1f32}else{0.}).collect::<Vec<_>>(),[selected.len()]),d));
    for data in [&t.memory,&t.forecasts]{fill(data,&mask,selected.len(),0.);}
}
/// Reset only selected arenas. Teacher and observer private caches invalidate
/// themselves when the native episode generation changes; no teacher logic changes.
pub fn reset(w:&World,t:&crate::scripted_teacher::Teacher,metrics:&crate::ggd_metrics::Metrics,selected:&[bool])->Tensor<1,Bool>{
    assert_eq!(selected.len(),w.matches);let mask=primitive(Tensor::<1>::from_data(TensorData::new(selected.iter().map(|x|if *x{1f32}else{0.}).collect::<Vec<_>>(),[w.matches]),&w.device));
    reset_state::launch(&w.state.client,CubeCount::Static(w.matches.div_ceil(32)as u32,1,1),CubeDim::new_1d(32),argument(&w.state),argument(&w.meta),argument(&w.prototypes),argument(&mask),w.matches,w.engine.pre_game_seconds);
    for data in [&w.shots,&w.shot_hits,&w.commands,&w.teacher_combat,&w.chio_events,&w.collision_grid,&w.rewards,&w.terminals,&t.memory,&t.forecasts,&t.audit,&metrics.buffer]{fill(data,&mask,w.matches,0.);}
    fill(&w.tree_alive,&mask,w.matches,1.);
    metrics.sample(w,t);
    tensor::<1>(mask).greater_elem(0.).unsqueeze_dim::<2>(1).expand([w.matches,2]).reshape([w.matches*2])
}
pub fn learner_rows(sides:&[usize],d:&burn::tensor::Device)->Tensor<1,Bool>{
    Tensor::<1>::from_data(TensorData::new((0..sides.len()*2).map(|a|if a%2==sides[a/2]{1f32}else{0.}).collect::<Vec<_>>(),[sides.len()*2]),d).greater_elem(0.)
}
pub fn delays(matches:usize,blocks:usize,seed:u32)->Vec<usize>{
    let mut values:Vec<_>=(0..matches).map(|i|i%(blocks+1)).collect();let mut rng=seed|1;
    for i in (1..matches).rev(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;values.swap(i,rng as usize%(i+1));}values
}
/// Counts only learner decisions eligible for PPO, including all pregame decisions.
pub fn phase_counts(seconds:Tensor<1>,valid:Tensor<1>)->Tensor<1>{
    let bins=vec![seconds.clone().lower_elem(0.),seconds.clone().greater_equal_elem(0.).bool_and(seconds.clone().lower_elem(180.)),seconds.clone().greater_equal_elem(180.).bool_and(seconds.clone().lower_elem(360.)),seconds.greater_equal_elem(360.)];
    Tensor::stack::<2>(bins.into_iter().map(|x|(x.float()*valid.clone()).sum()).collect(),1).reshape([4])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn p08_phase_bins_include_boundaries_and_exclude_invalid_decisions(){
        let d=crate::runtime::device();let result=phase_counts(Tensor::from_data([-1f32,0.,179.9,180.,359.9,360.,600.],&d),Tensor::from_data([1f32,1.,1.,1.,1.,1.,0.],&d));
        assert_eq!(result.into_data().try_to_vec::<f32>().unwrap(),vec![1.,2.,2.,1.]);
    }
    #[test]
    fn p08_offsets_cover_the_full_episode_and_repeat_for_the_same_seed(){
        let a=delays(192,40,20261103);assert_eq!(a,delays(192,40,20261103));assert_ne!(a,delays(192,40,20261104));
        for b in 0..=40 {assert!((4..=5).contains(&a.iter().filter(|x|**x==b).count()));}
    }
    #[test]
    fn p08_reset_changes_only_the_selected_arena_and_clears_its_queue(){
        let d=crate::runtime::device();let engine=std::path::Path::new("C:/AMI/runs/gpu/dota-calibration-003/engine.json");
        let w=World::load(engine,2,&d).unwrap();let t=crate::scripted_teacher::Teacher::new(&w);let metrics=crate::ggd_metrics::Metrics::new(&w);
        let obs=w.observe(crate::observe::initial_previous(4,&d));let _=t.actions(&w,&obs);
        let before=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();
        let all=primitive(Tensor::<1>::from_data([1f32,1.],&d));fill(&w.commands,&all,2,9.);fill(&t.audit,&all,2,7.);
        reset(&w,&t,&metrics,&[false,true]);
        let after=tensor::<1>(w.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(&before[..UNITS*STRIDE],&after[..UNITS*STRIDE]);
        let fresh=World::load(engine,2,&d).unwrap();let expected=tensor::<1>(fresh.state.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(&expected[UNITS*STRIDE..],&after[UNITS*STRIDE..]);
        let queue=tensor::<1>(w.commands.clone()).into_data().try_to_vec::<f32>().unwrap();assert!(queue[..COMMAND_WORDS].iter().all(|x|*x==9.));assert!(queue[COMMAND_WORDS..].iter().all(|x|*x==0.));
        let audit=t.values();assert!(audit[..crate::scripted_teacher::AUDIT_FIELDS].iter().all(|x|*x==7.));assert!(audit[crate::scripted_teacher::AUDIT_FIELDS..].iter().all(|x|*x==0.));
        let meta=tensor::<1>(w.meta.clone()).into_data().try_to_vec::<f32>().unwrap();assert_eq!(meta[1],0.);assert_eq!(meta[META+1],1.);assert_eq!(meta[META+6],0.);
    }
}
