"""Export a 128-step real CHiO demonstration fixture after imitation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from ggd_dataset import read_block,decode
from p15_model_fixture import Policy,HEADS,export_weights,write_bundle
from train_ggd import recurrent
p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
torch.set_num_threads(2);manifest=json.loads((a.dataset/'dataset.json').read_text());best=None;score=-1
for entry in manifest['files']:
    h,o,s=read_block(a.dataset/entry['file'],entry['sha256'])
    for i,actor in enumerate(h['actors']):
        if (s[i,:,11]>.5).all():
            count=int(((s[i,:,0]>=6)&(s[i,:,0]<=9)).sum())
            if count>score:best=(entry,h,o[i:i+1].copy(),s[i:i+1].copy(),actor);score=count
if best is None:raise ValueError('No complete 128-step sequence')
entry,header,obs,signals,actor=best;b=decode(header,obs,signals);labels=torch.from_numpy(b.pop('labels'));b.pop('loss_valid');b={k:torch.from_numpy(v) for k,v in b.items()}
model=Policy();saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['state_dict']);model.eval();resets=signals[:,:,10]
with torch.no_grad():
    h,keys=recurrent(model,b,1,resets);value=model.value_head(h).squeeze(-1)
    _,logp,_,_,_,logits=model.heads(b,h,keys,value,actions={name:labels[:,i] for i,name in enumerate(HEADS)},return_logits=True)
a.output.mkdir(parents=True,exist_ok=False);export_weights(model,a.output/'weights')
arrays={k:v.numpy() for k,v in b.items()};arrays.update(labels=labels.numpy(),resets=resets.reshape(-1),expected_h=h.numpy(),expected_value=value.numpy(),expected_logp=logp.numpy())
arrays.update({f'expected_{name}_logits':value.numpy() for name,value in zip(HEADS,logits)})
write_bundle(a.output/'fixture',arrays,source='real FP16 CHiO demonstration, decoded honest model input')
result={'file':entry['file'],'sha256':entry['sha256'],'actor':actor,'steps':128,'cast_decisions':score,'checkpoint':str(a.checkpoint)}
(a.output/'provenance.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result))
