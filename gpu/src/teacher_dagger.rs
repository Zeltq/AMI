//! Reconcile speculative teacher bookkeeping after a DAgger command override.
//! `decide` itself stays unchanged. Observation-derived memory is retained.
use super::*;

#[cube(launch)]
fn reconcile(state:&mut [f32],meta:&[f32],hero:&[f32],refs:&[f32],audit:&[f32],
    before:&[f32],old_forecasts:&[f32],labels:&[f32],executed:&[f32],memory:&mut [f32],
    forecasts:&mut [f32],actors:usize,pregame:f32){
    let a=ABSOLUTE_POS;
    if a<actors && audit[a/2*AUDIT_FIELDS]==0.0f32 {
        let mut same=true;for i in 0..6 {if labels[a*6+i]!=executed[a*6+i]{same=false;}}
        // Exact no-op for teacher commands, including beta=1 parity.
        if !same {
            let m=a/2;let side=a%2;let base=m*UNITS*STRIDE;let p=base+side*STRIDE;let mem=a*MEMORY_FIELDS;
            let time=(meta[m*META]+pregame*30.0f32)/30.0f32;
            let same_episode=before[mem]==memory[mem];
            for i in 1..7 {memory[mem+i]=if same_episode{before[mem+i]}else{0.0f32};}
            for i in 9..15 {memory[mem+i]=if same_episode{before[mem+i]}else{0.0f32};}
            if !same_episode {memory[mem+1]=-1.0f32;memory[mem+10]=-1.0f32;}
            let action=executed[a*6]as usize;let target=executed[a*6+3]as usize;
            let id=refs[a*TARGETS+target];
            if action==3 && id>=0.0f32 && id<UNITS as f32 {
                let u=id as usize;let q=base+u*STRIDE;let f=(a*UNITS+u)*FORECAST_FIELDS;
                let mut hit=own_hit(state,hero,meta,base,p,q,m,side,time);
                if state[p+WINDUP]>time && state[p+WINDUP_TARGET]==id && state[p+WINDUP_GENERATION]==state[q+GENERATION] {hit=state[p+WINDUP]+flight(state,hero,p,q);}
                else if memory[mem+1]==id && memory[mem+2]==state[q+GENERATION] && time<memory[mem+3] {hit=memory[mem+5];}
                memory[mem+1]=id;memory[mem+2]=state[q+GENERATION];memory[mem+3]=hit-flight(state,hero,p,q)+1.0f32/30.0f32;
                memory[mem+4]=time;memory[mem+5]=hit;
                memory[mem+6]=if forecasts[f]==state[q+GENERATION] && forecasts[f+1]==time{forecasts[f+3]}else{state[q+UNIT_HEALTH]};
            }else if action!=0 {memory[mem+1]=-1.0f32;}
            // CONTINUE preserves the actual pending order, unless its target died
            // or its protected release window expired.
            if action==0 && memory[mem+1]>=0.0f32 {
                let u=memory[mem+1]as usize;
                if u<UNITS {let q=base+u*STRIDE;
                    if state[q+ACTIVE]==0.0f32 || state[q+UNIT_HEALTH]<=0.0f32 || state[q+GENERATION]!=memory[mem+2] ||
                        (time>=memory[mem+3] && !(state[p+WINDUP]>time && state[p+WINDUP_TARGET]==u as f32)) {memory[mem+1]=-1.0f32;}
                }
            }
            if action==10 && (executed[a*6+2]==5.0f32 || executed[a*6+2]==6.0f32) {memory[mem+14]=time+0.4f32;}
            if action!=0 && state[p+WINDUP]>time && (action!=3 || id!=state[p+WINDUP_TARGET]) {
                memory[mem+9]+=1.0f32;memory[mem+10]=state[p+WINDUP_TARGET];memory[mem+11]=time;
            }
            memory[mem+12]=action as f32;memory[mem+13]+=if action>0{1.0f32}else{0.0f32};
            // Forecast 9 is diagnostic competition history, not an input to
            // decide. Recompute it against the actual committed target too.
            for u in 6..CREEP_END {let f=(a*UNITS+u)*FORECAST_FIELDS;
                forecasts[f+9]=if same_episode && old_forecasts[f]==forecasts[f]{old_forecasts[f+9]}else{0.0f32};
                if forecasts[f+1]==time && forecasts[f+6]>0.0f32 && memory[mem+1]>=0.0f32 && memory[mem+1]!=u as f32 {forecasts[f+9]=time;}
            }
        }
    }
}

impl Teacher {
    pub fn command_snapshot(&self)->(CubeTensor,CubeTensor){
        (primitive(tensor::<1>(self.memory.clone())+0.0f32),primitive(tensor::<1>(self.forecasts.clone())+0.0f32))
    }
    pub fn reconcile_commands(&self,w:&World,obs:&Observed,before:(CubeTensor,CubeTensor),labels:Tensor<2,Int>,executed:Tensor<2,Int>){
        let labels=primitive(labels.float());let executed=primitive(executed.float());
        reconcile::launch(&w.state.client,CubeCount::Static((w.matches*2).div_ceil(32)as u32,1,1),CubeDim::new_1d(32),
            argument(&w.state),argument(&w.meta),argument(&w.hero),argument(&obs.refs),argument(&self.audit),argument(&before.0),argument(&before.1),
            argument(&labels),argument(&executed),argument(&self.memory),argument(&self.forecasts),w.matches*2,w.engine.pre_game_seconds);
    }
}
