"""CPU gradient and replay-boundary verification for masked base navigation."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from p15_model_fixture import Policy,batch
from train_ggd import recurrent
from chio_mid_phase import MidPhase
OUT=Path('/mnt/c/AMI/runs/p16/mid-mask-check');OUT.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(20261103);model=Policy();b,labels=batch(model);reset=np.zeros((1,128),np.float32);reset[0,4]=1
b['heroes'].requires_grad_(True)
h,keys=recurrent(model,b,1,reset);loss=model.value_head(h[4:]).square().mean();loss.backward()
pre=float(b['heroes'].grad[:4].abs().max());post=float(b['heroes'].grad[4:].abs().max());assert pre==0 and post>0
changed={k:v.detach().clone() for k,v in b.items()};changed['heroes'][:4]=1000
with torch.no_grad():
    baseline,_=recurrent(model,{k:v.detach() for k,v in b.items()},1,reset);other,_=recurrent(model,changed,1,reset)
error=float((baseline[4:]-other[4:]).abs().max());assert error==0
manifest=json.loads(Path('/mnt/c/AMI/runs/p16/teacher-1024/dataset.json').read_text());phase=MidPhase(manifest)
counts={'radiant':0,'dire':0};actors={'radiant':0,'dire':0}
for actor,start in phase.start.items():
    side='radiant' if actor%2==0 else 'dire';actors[side]+=1;counts[side]+=start or 0
result=dict(passed=True,pre_input_gradient_max=pre,post_input_gradient_max=post,prefix_perturbation_post_hidden_max_error=error,teacher_premid_decisions=counts,actors=actors,source_matches=manifest['completed_matches'])
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
