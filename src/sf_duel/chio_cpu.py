"""Strict CPU loader for the versioned CHiO Burn tensor bundle."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from .chio_model import Policy


def burn_name(name):
    for group in ('hero','unit','tree','projectile','skill','item'):
        if name.startswith(group+'.'):
            return name.replace(group+'.0.',group+'.first.').replace(group+'.2.',group+'.second.')
    for source,target in {'terrain.0':'terrain_first','terrain.2':'terrain_second','terrain.6':'terrain_linear',
        'combine.0':'combine','value_head.0':'value_first','value_head.2':'value_second',
        'memory.weight_ih_l0':'memory.input.weight','memory.weight_hh_l0':'memory.hidden.weight',
        'memory.bias_ih_l0':'memory.input.bias','memory.bias_hh_l0':'memory.hidden.bias'}.items():
        if name==source or name.startswith(source+'.'):return name.replace(source,target,1)
    return name


def load_bundle(root):
    root=Path(root).resolve()
    saved=torch.load(root,map_location='cpu',weights_only=True) if root.is_file() else None
    manifest=saved['config'] if saved is not None else json.loads((root/'manifest.json').read_text())
    required=dict(family='CHiO',observation_profile='chio-v1',action_types=13,directions=36,
                  decision_ticks=6,offset_ticks=6,recurrent_reset=128,timing_honest=True,timing_oracle=False)
    for key,value in required.items():
        if manifest.get(key)!=value:raise ValueError('CHiO contract mismatch: '+key)
    torch.set_num_threads(1)
    model=Policy().cpu().eval();state={}
    if saved is not None:
        model.load_state_dict(saved['state_dict'],strict=True)
        if any(not torch.isfinite(x).all() for x in model.state_dict().values()):raise ValueError('Nonfinite CPU weights')
        return model,manifest
    for name,expected in model.state_dict().items():
        entry=manifest['tensors'][burn_name(name)];path=(root/entry['file']).resolve()
        if path.parent!=root:raise ValueError('Tensor path escapes bundle')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry.get('sha256'):raise ValueError('Tensor checksum: '+name)
        value=np.frombuffer(raw,dtype='<f4').reshape(entry['shape'])
        if value.ndim==2 and not name.endswith('_embed.weight'):value=value.T
        if value.shape!=tuple(expected.shape) or not np.isfinite(value).all():raise ValueError('Invalid tensor: '+name)
        state[name]=torch.from_numpy(value.copy())
    if len(state)!=len(manifest['tensors']):raise ValueError('Unexpected tensors')
    model.load_state_dict(state,strict=True)
    return model,manifest


class Session:
    def __init__(self,model):
        self.model=model;self.state=None;self.decisions=0;self.episode=None
        self.last_tick=None;self.resets=0

    @torch.inference_mode()
    def decide(self,b,tick,episode,death=False,handoff=False):
        if self.episode!=episode or death:
            self.state=None;self.decisions=0;self.last_tick=None;self.episode=episode
        if self.last_tick is not None and tick<=self.last_tick:raise ValueError('Nonmonotonic observation tick')
        reset=self.decisions%128==0 or handoff
        if reset:self.state=None;self.resets+=1
        b={k:(v.cpu().half().float() if v.is_floating_point() else v.cpu()) for k,v in b.items()}
        if any(v.is_floating_point() and not torch.isfinite(v).all() for v in b.values()):raise ValueError('Nonfinite observation')
        chosen,_,_,value,self.state=self.model.decide(b,self.state,deterministic=False)
        self.decisions+=1;self.last_tick=tick
        return {k:int(v.item()) for k,v in chosen.items()},float(value.item()),reset
