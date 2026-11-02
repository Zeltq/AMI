"""Check native observations and conditional action labels on ROCm, no training."""
import argparse,json,sys
from pathlib import Path
import torch
from sf_duel.native_data import examples,padded_label,decode_action,action_label
from sf_duel.observations import batch
from sf_duel.model import Policy

def main():
 p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=Path(a.run)
 torch.set_num_threads(2);torch.manual_seed(7);model=Policy().to('cuda').eval()
 samples=[];sequence=[]
 for row in examples(run):
  if row['encoded'].snapshot['time']<0:continue
  if row['sequence']%19==0:samples.append(row)
  if row['team']==2 and len(sequence)<16:sequence.append(row)
  if len(samples)>=64 and len(sequence)>=16:break
 assert len(samples)>=16
 checked=0
 with torch.inference_mode():
  for start in range(0,len(samples),32):
   chunk=samples[start:start+32];b=batch([r['encoded'] for r in chunk],'cuda');pc=b['creeps'].shape[1];pt=b['trees'].shape[1]
   labels=[padded_label(r['action'],r['encoded'],pc,pt) for r in chunk]
   teacher={k:torch.tensor([r[k] for r in labels],device='cuda') for k in ('action','buy','item','target','x','y')}
   _,logp,_,value,_=model.decide(b,actions=teacher)
   assert torch.isfinite(logp).all() and torch.isfinite(value).all()
   choices,_,_,_,_=model.decide(b)
   choices={k:v.cpu().tolist() for k,v in choices.items()}
   for i,r in enumerate(chunk):
    picked={k:v[i] for k,v in choices.items()};d=decode_action(picked,r['encoded'],pc,pt,r['sequence'],r['observed_tick'])
    reconstructed=padded_label(action_label(d,r['encoded']),r['encoded'],pc,pt)
    assert all(reconstructed[k]==picked[k] for k in picked)
    checked+=1
  memory=model.initial_state(1)
  for r in sequence:
   b=batch([r['encoded']],'cuda');_,_,_,value,memory=model.decide(b,memory)
   assert torch.isfinite(value).all() and all(torch.isfinite(x).all() for x in memory)
 torch.cuda.synchronize()
 report={'passed':True,'device':torch.cuda.get_device_name(),'parameters':sum(p.numel() for p in model.parameters()),'random_decisions_roundtripped':checked,'recurrent_steps':len(sequence),'teacher_conditional_log_probs_finite':True,'weights_updated':False,'training_started':False}
 (run/'native-gpu-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
