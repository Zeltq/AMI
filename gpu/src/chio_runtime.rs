//! Shared observe/execute/reward boundary for every CHiO controller and mode.
use burn::tensor::{Tensor,Int,DType,Bool,TensorData};
use burn_cubecl::tensor::CubeTensor;
use crate::{world::*,observe::{self,Observed},chio_observe as co,chio_actions as ca,chio_reward as reward,
    scripted_teacher::Teacher,teacher_v10::{Controller,Version},sampling::Sampler};

pub struct Frame {
    pub base:Observed,pub observation:Observed,pub masks:ca::Masks,pub labels:Tensor<2,Int>,pub valid:Tensor<1>,
    raw:Tensor<2,Int>,teacher_before:(CubeTensor,CubeTensor),teacher_labels:Tensor<2,Int>,
    routed:Option<((CubeTensor,CubeTensor),Tensor<2,Int>,Tensor<2,Int>)>,
}
pub struct Transition {pub before:CubeTensor,pub after:CubeTensor,pub reward:reward::Reward,pub done:Tensor<1>}
pub struct Runtime {
    pub world:World,pub diagnostic_teacher_purchases:bool,pub teacher:Teacher,pub metrics:crate::ggd_metrics::Metrics,pub extra:crate::chio_metrics::Metrics,
    controller:Controller,translator:crate::chio_teacher::Translator,economy:crate::chio_economy::Economy,
    history:co::Observer,combat:crate::combat_observe::CombatObserver,enemy:crate::enemy_observe::Observer,
    v9:Option<(Teacher,crate::chio_teacher::Translator)>,v9_rows:Option<Tensor<2,Bool>>,
    previous:Tensor<2,Int>,legacy_previous:Tensor<2>,legacy_right:bool,swap:bool,zero_sum:bool,
}
impl Runtime {
    pub fn new(w:World,left:Version,right:Version,swap:bool,legacy_right:bool,zero_sum:bool)->anyhow::Result<Self>{
        w.validate_profile("chio-v1")?;anyhow::ensure!(!legacy_right||right==Version::V9,"original opponent mode is defined for v9");
        let teacher=Teacher::new(&w);let metrics=crate::ggd_metrics::Metrics::new(&w);metrics.sample(&w,&teacher);
        let result=Self{controller:Controller::new(&w,left,right,swap),translator:crate::chio_teacher::Translator::new(&w),economy:crate::chio_economy::Economy::new(&w),
            history:co::Observer::new(&w),combat:crate::combat_observe::CombatObserver::new(&w),enemy:crate::enemy_observe::Observer::new(&w),
            v9:None,v9_rows:None,previous:co::initial_previous(w.matches*2,&w.device),legacy_previous:observe::initial_previous(w.matches*2,&w.device),extra:crate::chio_metrics::Metrics::new(&w),
            world:w,diagnostic_teacher_purchases:false,teacher,metrics,legacy_right,swap,zero_sum};Ok(result)
    }
    /// Opponent routing only; each v9 uses the unchanged teacher and CHiO translator.
    pub fn route_v9(&mut self,selected:&[bool]) {
        assert_eq!(selected.len(),self.world.matches*2);
        if self.v9.is_none() {let mut t=Teacher::new(&self.world);t.audit=self.teacher.audit.clone();
            self.v9=Some((t,crate::chio_teacher::Translator::new(&self.world)));}
        self.v9_rows=Some(Tensor::<1,Int>::from_data(TensorData::new(selected.iter().map(|x|i32::from(*x)).collect(),[selected.len()]),&self.world.device).greater_elem(0).unsqueeze_dim(1));
    }
    pub fn observe(&self)->Frame {
        let w=&self.world;let base=w.observe(self.legacy_previous.clone());let valid=crate::chio_collect::eligibility(w,&self.teacher);let masks=ca::observe(w,&base);
        let teacher_before=self.teacher.command_snapshot();let raw=self.controller.actions(&self.teacher,w,&base);let labels=self.translator.labels_with_intent(w,&base,&masks,raw.clone(),&self.controller.intents);
        let teacher_labels=labels.clone();
        let routed=self.v9.as_ref().map(|(t,translator)|{let before=t.command_snapshot();let raw=t.actions(w,&base);
            let labels=translator.labels(w,&base,&masks,raw.clone());(before,raw,labels)});
        let labels=if let Some((_,_,v9))=&routed {labels.mask_where(self.v9_rows.as_ref().unwrap().clone(),v9.clone())}else{labels};
        let v4=self.enemy.extend(w,crate::timing_observe::extend(w,self.combat.enrich(w,&base),&self.teacher,true,false));
        let mut observation=self.history.extend(w,&base,v4,self.previous.clone(),&masks);observation.packed=observation.packed.cast(DType::F16);
        Frame{base,observation,masks,labels,valid,raw,teacher_before,teacher_labels,routed}
    }
    pub fn advance(&mut self,frame:&Frame,chosen:Tensor<2,Int>,rng:&Sampler,deadline:f32)->Transition {
        let w=&self.world;let buys=if self.diagnostic_teacher_purchases {frame.raw.clone().slice([0..w.matches*2,1..2]).float().squeeze_dim(1)}else{self.economy.choose(w,&frame.base)};let mut executed=ca::enqueue(w,&frame.base,&frame.masks,chosen.clone(),buys.clone());
        if self.legacy_right {executed=ca::original_v9_right(w,&frame.base,frame.raw.clone(),buys,executed,self.swap);}
        self.translator.reconcile(w,frame.teacher_labels.clone(),chosen.clone());
        if let (Some((t,translator)),Some((before,raw,labels)))=(&self.v9,&frame.routed) {
            translator.reconcile(w,labels.clone(),chosen.clone());
            t.reconcile_commands(w,&frame.base,(before.0.clone(),before.1.clone()),raw.clone(),executed.clone());
        }
        self.teacher.reconcile_commands(w,&frame.base,(frame.teacher_before.0.clone(),frame.teacher_before.1.clone()),frame.raw.clone(),executed.clone());
        self.metrics.commands(w,&self.teacher,executed.clone(),&frame.base.refs);self.legacy_previous=frame.base.previous(executed,&w.bases,&w.state);self.previous=chosen;
        let before=reward::snapshot(w);let (_,done)=w.step_ticks_audit(rng,w.engine.decision_ticks,true,Some(&self.teacher));self.teacher.record(w,deadline);self.metrics.sample(w,&self.teacher);
        let after=reward::snapshot(w);let reward=reward::transition(&before,&after,self.zero_sum,&w.device);self.extra.sample(w,&after,&reward,frame.valid.clone());
        Transition{before,after,reward,done}
    }
    pub fn reset(&mut self,selected:&[bool]) {
        let mask=crate::ggd_arenas::reset(&self.world,&self.teacher,&self.metrics,selected).unsqueeze_dim::<2>(1);
        self.previous=self.previous.clone().mask_where(mask.clone(),co::initial_previous(self.world.matches*2,&self.world.device));
        self.legacy_previous=self.legacy_previous.clone().mask_where(mask,observe::initial_previous(self.world.matches*2,&self.world.device));
        self.extra.reset(&self.world,selected);
        if let Some((t,_))=&self.v9 {crate::ggd_arenas::reset_teacher_cache(t,selected,&self.world.device);}
    }
    pub fn games(&self,offset:usize)->anyhow::Result<Vec<serde_json::Value>>{
        let audit=self.teacher.values();anyhow::ensure!(audit.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).all(|v|v[39]==0.),"CHiO simulation/label error");
        let mut games=self.metrics.games(&audit,self.world.engine.pre_game_seconds,offset,None);crate::demo_telemetry::enrich(&self.world,&mut games)?;self.extra.enrich(&mut games)?;Ok(games)
    }
}
