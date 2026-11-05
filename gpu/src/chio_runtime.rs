//! Shared observe/execute/reward boundary for every CHiO controller and mode.
use burn::tensor::{Tensor,Int,DType};
use burn_cubecl::tensor::CubeTensor;
use crate::{world::*,observe::{self,Observed},chio_observe as co,chio_actions as ca,chio_reward as reward,
    scripted_teacher::Teacher,teacher_v10::{Controller,Version},sampling::Sampler};

pub struct Frame {
    pub base:Observed,pub observation:Observed,pub masks:ca::Masks,pub labels:Tensor<2,Int>,pub valid:Tensor<1>,
    raw:Tensor<2,Int>,teacher_before:(CubeTensor,CubeTensor),
}
pub struct Transition {pub before:CubeTensor,pub after:CubeTensor,pub reward:reward::Reward,pub done:Tensor<1>}
pub struct Runtime {
    pub world:World,pub teacher:Teacher,pub metrics:crate::ggd_metrics::Metrics,pub extra:crate::chio_metrics::Metrics,
    controller:Controller,translator:crate::chio_teacher::Translator,economy:crate::chio_economy::Economy,
    history:co::Observer,combat:crate::combat_observe::CombatObserver,enemy:crate::enemy_observe::Observer,
    previous:Tensor<2,Int>,legacy_previous:Tensor<2>,legacy_right:bool,swap:bool,zero_sum:bool,
}
impl Runtime {
    pub fn new(w:World,left:Version,right:Version,swap:bool,legacy_right:bool,zero_sum:bool)->anyhow::Result<Self>{
        w.validate_profile("chio-v1")?;anyhow::ensure!(!legacy_right||right==Version::V9,"original opponent mode is defined for v9");
        let teacher=Teacher::new(&w);let metrics=crate::ggd_metrics::Metrics::new(&w);metrics.sample(&w,&teacher);
        let result=Self{controller:Controller::new(&w,left,right,swap),translator:crate::chio_teacher::Translator::new(&w),economy:crate::chio_economy::Economy::new(&w),
            history:co::Observer::new(&w),combat:crate::combat_observe::CombatObserver::new(&w),enemy:crate::enemy_observe::Observer::new(&w),
            previous:co::initial_previous(w.matches*2,&w.device),legacy_previous:observe::initial_previous(w.matches*2,&w.device),extra:crate::chio_metrics::Metrics::new(&w),
            world:w,teacher,metrics,legacy_right,swap,zero_sum};Ok(result)
    }
    pub fn observe(&self)->Frame {
        let w=&self.world;let base=w.observe(self.legacy_previous.clone());let valid=crate::chio_collect::eligibility(w,&self.teacher);let masks=ca::observe(w,&base);
        let teacher_before=self.teacher.command_snapshot();let raw=self.controller.actions(&self.teacher,w,&base);let labels=self.translator.labels(w,&base,&masks,raw.clone());
        let v4=self.enemy.extend(w,crate::timing_observe::extend(w,self.combat.enrich(w,&base),&self.teacher,true,false));
        let mut observation=self.history.extend(w,&base,v4,self.previous.clone(),&masks);observation.packed=observation.packed.cast(DType::F16);
        Frame{base,observation,masks,labels,valid,raw,teacher_before}
    }
    pub fn advance(&mut self,frame:&Frame,chosen:Tensor<2,Int>,rng:&Sampler,deadline:f32)->Transition {
        let w=&self.world;let buys=self.economy.choose(w,&frame.base);let mut executed=ca::enqueue(w,&frame.base,&frame.masks,chosen.clone(),buys.clone());
        if self.legacy_right {executed=ca::original_v9_right(w,&frame.base,frame.raw.clone(),buys,executed,self.swap);}
        self.teacher.reconcile_commands(w,&frame.base,(frame.teacher_before.0.clone(),frame.teacher_before.1.clone()),frame.raw.clone(),executed.clone());
        self.metrics.commands(w,&self.teacher,executed.clone(),&frame.base.refs);self.legacy_previous=frame.base.previous(executed,&w.bases,&w.state);self.previous=chosen;
        let before=reward::snapshot(w);let (_,done)=w.step_ticks_audit(rng,w.engine.decision_ticks,true,Some(&self.teacher));self.teacher.record(w,deadline);self.metrics.sample(w,&self.teacher);
        let after=reward::snapshot(w);let reward=reward::transition(&before,&after,self.zero_sum,&w.device);self.extra.sample(w,&after,&reward,frame.valid.clone());
        Transition{before,after,reward,done}
    }
    pub fn games(&self,offset:usize)->anyhow::Result<Vec<serde_json::Value>>{
        let audit=self.teacher.values();anyhow::ensure!(audit.chunks_exact(crate::scripted_teacher::AUDIT_FIELDS).all(|v|v[39]==0.),"CHiO simulation/label error");
        let mut games=self.metrics.games(&audit,self.world.engine.pre_game_seconds,offset,None);crate::demo_telemetry::enrich(&self.world,&mut games)?;self.extra.enrich(&mut games)?;Ok(games)
    }
}
