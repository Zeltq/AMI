"""Offline PyTorch bridge. JSON describes binary tensors; never used in the hot loop."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from sf_duel.model import Policy

def rust_name(name):
    for group in ('hero','unit','tree','projectile','skill','item'):
        if name.startswith(group+'.'):
            return name.replace(group+'.0.',group+'.first.').replace(group+'.2.',group+'.second.')
    for source,target in {'terrain.0':'terrain_first','terrain.2':'terrain_second','terrain.6':'terrain_linear',
        'combine.0':'combine','value_head.0':'value_first','value_head.2':'value_second',
        'memory.weight_ih_l0':'memory.input.weight','memory.weight_hh_l0':'memory.hidden.weight',
        'memory.bias_ih_l0':'memory.input.bias','memory.bias_hh_l0':'memory.hidden.bias'}.items():
        if name==source or name.startswith(source+'.'): return name.replace(source,target,1)
    return name

def write_bundle(path,arrays,**metadata):
    path.mkdir(parents=True,exist_ok=True)
    entries={}
    for i,(name,value) in enumerate(arrays.items()):
        value=np.asarray(value,dtype='<f4',order='C')
        file=f'{i:03d}.f32';value.tofile(path/file)
        entries[name]={'shape':list(value.shape),'file':file,'sha256':hashlib.sha256((path/file).read_bytes()).hexdigest()}
    (path/'manifest.json').write_text(json.dumps({'format':1,'tensors':entries,**metadata},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--fixture');p.add_argument('--steps',type=int,default=160);p.add_argument('--zero-previous',action='store_true');a=p.parse_args()
    torch.set_num_threads(2)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    state=checkpoint.get('model',checkpoint.get('state_dict',checkpoint))
    model=Policy(combat_inputs=state['hero.0.weight'].shape[1]>2*48,timing_inputs=state['hero.0.weight'].shape[1]>2*(48+11),enemy_inputs=state['hero.0.weight'].shape[1]>140);model.load_state_dict(state);model.eval()
    arrays={}
    for name,v in state.items():
        value=v.detach().float().numpy()
        if value.ndim==2 and not name.endswith('_embed.weight'): value=value.T
        arrays[rust_name(name)]=value
    output=Path(a.output)
    metadata={'zero_previous':bool(a.zero_previous or checkpoint.get('config',{}).get('zero_previous',False))}
    if model.combat_inputs:
        config=checkpoint.get('config',{})
        metadata.update(family='GGD',model_name=config.get('model_name'),observation_profile='ggd-v4' if model.enemy_inputs else 'ggd-v3' if model.timing_inputs else 'ggd-v2',timing_honest=config.get('timing_honest',True),timing_oracle=config.get('timing_oracle',False),teacher_sha256=config.get('teacher_policy_sha256'),provenance='Fresh GGD policy trained by imitation in the GPU simulator')
    write_bundle(output/'weights',arrays,source=str(a.checkpoint),source_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),recurrent_reset=128,**metadata)
    if a.fixture:
        data=np.load(a.fixture);count=min(a.steps,len(data['labels']))
        fixture_arrays={k:np.asarray(data[k][:count],dtype=np.bool_ if data[k].dtype==np.bool_ else np.float32) for k in data.files if k not in ('labels','loss_valid')}
        if model.timing_inputs:
            from sf_duel.timing_observations import ablate
            fixture_arrays=ablate(fixture_arrays,metadata['timing_honest'],metadata['timing_oracle'])
        batch={k:torch.from_numpy(v) for k,v in fixture_arrays.items()}
        if metadata['zero_previous']:batch['previous']=torch.zeros_like(batch['previous'])
        labels=torch.from_numpy(data['labels'][:count].astype(np.int64))
        arrays={k:v.numpy() for k,v in batch.items()};arrays['labels']=labels.numpy()
        with torch.no_grad():
            fused,keys=model.encode_entities(batch);hs=[];cs=[];memory=None
            for i in range(count):
                if i%128==0: memory=None
                _,memory=model.memory(fused[i:i+1,None,:],memory)
                hs.append(memory[0][0]);cs.append(memory[1][0])
            h=torch.cat(hs);c=torch.cat(cs)
            value=model.value_head(h).squeeze(-1)
            _,logp,_,_,_=model.heads(batch,h,keys,value,None,actions={k:labels[:,i] for i,k in enumerate(('action','buy','item','target','x','y'))})
        arrays.update(expected_fused=fused.numpy(),expected_keys=keys.numpy(),expected_h=h.numpy(),expected_c=c.numpy(),expected_value=value.numpy(),expected_logp=logp.numpy())
        write_bundle(output/'fixture',arrays,source=str(a.fixture),recurrent_reset=128)
    print(json.dumps({'output':str(output),'parameters':sum(v.numel() for v in state.values()),'fixture_steps':count if a.fixture else 0}))

if __name__=='__main__':main()
