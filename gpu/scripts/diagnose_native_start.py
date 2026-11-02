"""Compare frozen action distributions on the same recorded native trajectory."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.model import Policy,distribution
from sf_duel.native_data import examples
from sf_duel.observations import batch
from sf_duel.spec import UNIT_FIELDS

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--baseline',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--team',type=int,choices=(2,3),default=2);p.add_argument('--zero-animation-ablation',action='store_true');p.add_argument('--sim-reset-pregame',type=float);p.add_argument('--until',type=float,default=1e9);a=p.parse_args()
 if a.zero_animation_ablation and a.sim_reset_pregame is not None:p.error('Change only one diagnostic factor at a time')
 torch.set_num_threads(2);models={};identities={}
 for role,file in [('baseline',a.baseline),('learner',a.checkpoint)]:
  path=Path(file);saved=torch.load(path,map_location='cpu',weights_only=False)
  assert saved['gpu_pipeline']['recurrent_reset_interval']==128 and saved['gpu_pipeline']['observation_storage']=='f16'
  model=Policy();model.load_state_dict(saved['state_dict'],strict=True);model.eval();models[role]=model;identities[role]=hashlib.sha256(path.read_bytes()).hexdigest()
 if a.zero_animation_ablation:models['learner_zero_animation']=models['learner'];identities['learner_zero_animation']=identities['learner']
 if a.sim_reset_pregame is not None:models['learner_sim_reset']=models['learner'];identities['learner_sim_reset']=identities['learner']
 rows=[];segment=[];phase_state=None;phase_block=None;first_time=None;first_tick=None;phase_checks=0;clock_residuals=[]
 def emit():
  nonlocal phase_state,phase_block,first_time,first_tick,phase_checks
  if not segment:return
  assert (segment[0]['sequence']-1)%128==0
  if first_time is None:first_time=segment[0]['encoded'].snapshot['time'];first_tick=segment[0]['observed_tick']
  origin_tick=first_tick+(-a.sim_reset_pregame-first_time)*30 if a.sim_reset_pregame is not None else first_tick
  block_for=lambda row:math.floor((row['observed_tick']-origin_tick)/(4*128))
  b=batch([row['encoded']for row in segment],'cpu');b={k:v.half().float()if v.is_floating_point()else v for k,v in b.items()};outputs={}
  with torch.inference_mode():
   for role,model in models.items():
    inputs=b
    if role=='learner_zero_animation':
     inputs=dict(b);index=list(UNIT_FIELDS).index('attack_cycle');assert index==42
     for group in ('heroes','creeps','towers'):inputs[group]=b[group].clone();inputs[group][...,index]=0
    fused,_=model.encode_entities(inputs)
    if role=='learner_sim_reset':
     hs=[];start=0
     while start<len(segment):
      block=block_for(segment[start])
      if block!=phase_block:phase_state=model.initial_state(1);phase_block=block
      end=start+1
      while end<len(segment)and block_for(segment[end])==block:end+=1
      part,phase_state=model.memory(fused[start:end].unsqueeze(0),phase_state);hs.append(part);start=end
     h=torch.cat(hs,dim=1)
    else:h,_=model.memory(fused.unsqueeze(0),model.initial_state(1))
    probs=distribution(model.action_head(h[0]),inputs['action_mask']).probs.numpy()
    outputs[role]=probs
  for i,row in enumerate(segment):
   time=row['encoded'].snapshot['time']
   # Use the authoritative observation tick, not DotaTime: the latter has
   # an approximately one-physical-tick phase change at game start.
   assert (row['observed_tick']-first_tick)%4==0
   assert (row['observed_tick']-first_tick)//(4*128)==(row['sequence']-1)//128
   clock_residuals.append(time-(first_time+(row['observed_tick']-first_tick)/30))
   phase_checks+=1
   rows.append({'sequence':row['sequence'],'time':time,'recorded_action':row['action']['action'],
    'movement_legal':row['encoded'].arrays['action_mask'][1:3].tolist(),
    **{role:{'continue':float(probs[i,0]),'move':float(probs[i,1]+probs[i,2])}for role,probs in outputs.items()}})
  print(f"Compared through sequence{segment[-1]['sequence']} / time{rows[-1]['time']:.1f}",flush=True)
 for row in examples(a.run):
  if row['team']!=a.team:continue
  if row['encoded'].snapshot['time']>a.until:break
  segment.append(row)
  if len(segment)==128:emit();segment=[]
 emit()
 samples=[]
 for time in (t for t in (0,10,30,60,120,300,390,410,450)if t<=rows[-1]['time']):
  row=min(rows,key=lambda row:abs(row['time']-time));samples.append(row)
 stationary=[row for row in rows if 0<=row['time']<400]
 report={'run':str(Path(a.run).resolve()),'team':a.team,'weights_sha256':identities,'samples':samples,'observations':len(rows),
  'pre400_mean_probabilities':{role:{key:sum(row[role][key]for row in stationary)/len(stationary)for key in ('continue','move')}for role in models},
  'movement_mask_legal_fraction':sum(all(row['movement_legal'])for row in stationary)/len(stationary),
  'method':'Both frozen models receive the same recorded observations, FP16 rounding, previous actual actions and128-step recurrent resets. Each model replays its own hidden state. No weight update or sampled action. This is an off-policy diagnostic; it does not simulate the counterfactual game.',
  'zero_animation_ablation':'Additional learner copy receives attack_cycle value0 for heroes/creeps/towers; known flags and all other inputs unchanged. Only an offline hypothesis test.'if a.zero_animation_ablation else None,
  'reset_phase_ablation':{'sim_pregame_seconds':a.sim_reset_pregame,'original_phase_reconstructed_checks':phase_checks,'DotaTime_minus_linear_tick_clock_minmax':[min(clock_residuals),max(clock_residuals)],'method':'Additional learner uses observed_tick periodic resets with a constant origin shifted to the simulator initial pregame time. Inputs including DotaTime remain unchanged. Original native tick-to-sequence reset formula checked exactly.'}if a.sim_reset_pregame is not None else None,
  'all_steps':rows}
 Path(a.output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
