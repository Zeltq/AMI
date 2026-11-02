"""Inference only: real recorded observations, random weights, persistent GPU LSTM."""
import json,time,sys
from pathlib import Path
import numpy as np
import torch
from sf_duel.demonstrations import teacher_records
from sf_duel.observations import encode,batch
from sf_duel.model import Policy
root=Path('/mnt/c/AMI');torch.set_num_threads(2);torch.manual_seed(1)
model=Policy().to('cuda').eval();samples=[]
for row in teacher_records(root/'runs/go-benchmark-10-001/match-0002'):
 s=row['observation']
 if 90<s['time']<300 and s['seq']%31==0:samples.append(s)
 if len(samples)>=80:break
assert len(samples)>=40
report={'device':torch.cuda.get_device_name(),'parameters':sum(p.numel() for p in model.parameters()),'random_weights':True,'training_started':False,'scope':'WSL local; excludes native protobuf conversion, batch queue, Windows transport and engine execution','batches':[]}
def percentiles(values):return {k:round(float(np.percentile(values,p)),3) for k,p in [('p50_ms',50),('p95_ms',95),('p99_ms',99),('max_ms',100)]}
with torch.inference_mode():
 for n in (1,10,20,40):
  rows=[samples[i%len(samples)] for i in range(n)]
  prepared=[encode(s) for s in rows];b=batch(prepared,'cuda');state=model.initial_state(n)
  for _ in range(30):*_,state=model.decide(b,state)
  torch.cuda.synchronize();gpu=[];local=[]
  for i in range(100):
   started=time.perf_counter();*_,state=model.decide(b,state);torch.cuda.synchronize();gpu.append((time.perf_counter()-started)*1000)
  for i in range(100):
   started=time.perf_counter();encoded=[encode(s) for s in rows];inputs=batch(encoded,'cuda')
   chosen,_,_,_,state=model.decide(inputs,state)
   output={k:v.cpu().tolist() for k,v in chosen.items()}
   torch.cuda.synchronize();local.append((time.perf_counter()-started)*1000)
  result={'actors':n,'iterations':100,'gpu_resident_input_decision':percentiles(gpu),'encode_transfer_decide_return':percentiles(local)}
  report['batches'].append(result);print(json.dumps(result),flush=True)
(root/'runs/inference-latency-20261030.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
