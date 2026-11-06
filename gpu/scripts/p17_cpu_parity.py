"""Replay existing 128-step Burn parity inputs on CPU, without starting Burn/GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.chio_cpu import load_bundle
from sf_duel.chio_model import HEADS


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    # This is the actual Burn export from the prior parity, not its original Python weights.
    model,manifest=load_bundle(a.reference/'roundtrip-weights')
    fm=json.loads((a.reference/'fixture/manifest.json').read_text());arrays={}
    for name,e in fm['tensors'].items():
        raw=(a.reference/'fixture'/e['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=e['sha256']:raise ValueError('Fixture checksum')
        arrays[name]=torch.from_numpy(np.frombuffer(raw,dtype='<f4').reshape(e['shape']).copy())
    b={k:v for k,v in arrays.items() if not k.startswith('expected_') and k not in ('labels','resets')}
    for k in b:
        if k.endswith(('_valid','_mask')) or k=='creep_ally':b[k]=b[k].bool()
    errors={};state=None;hs=[]
    with torch.inference_mode():
        fused,keys=model.encode_entities(b)
        for i in range(128):
            if i%128==0 or arrays['resets'][i]>.5:state=None
            _,state=model.memory(fused[i:i+1,None,:],state);hs.append(state[0][0])
        h=torch.cat(hs);value=model.value_head(h).squeeze(-1)
        _,logp,_,_,_,logits=model.heads(b,h,keys,value,actions={n:arrays['labels'][:,i].long() for i,n in enumerate(HEADS)},return_logits=True)
        for name,actual in [('h',h),('value',value),('logp',logp),*[(n+'_logits',v) for n,v in zip(HEADS,logits)]]:
            errors[name]=float((actual-arrays['expected_'+name]).abs().max())
    prior=json.loads((a.reference/'result.json').read_text())
    result=dict(device='cpu',steps=128,tensors=len(manifest['tensors']),errors=errors,passed=max(errors.values())<=1e-4,
                reference=str(a.reference),prior_burn_result=prior,
                limitation='CPU replay of a saved Burn export and shared fixture; no new Burn execution (GPU prohibited).')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
