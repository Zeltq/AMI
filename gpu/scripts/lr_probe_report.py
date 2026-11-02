"""Offline, fixed-history decisions after the one-rollout LR experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from sf_duel.model import Policy
from sf_duel.native_data import examples
from policy_history_audit import LAYOUT,canonical_native,decode_packed,tensor_batch,ACTION_COLUMNS


def replay(model,arrays,resets):
    state=model.initial_state(1);logits=[];actions=[]
    with torch.inference_mode():
        for start in range(0,len(resets),32):
            b=tensor_batch(arrays,start,start+32);fused,keys=model.encode_entities(b);hs=[]
            for i in range(len(fused)):
                if resets[start+i]:state=model.initial_state(1)
                _,state=model.memory(fused[i:i+1,None],state);hs.append(state[0][0,0])
            h=torch.stack(hs);logits.append(model.action_head(h))
            chosen,*_=model.heads(b,h,keys,model.value_head(h).squeeze(-1),None,deterministic=True,need_entropy=False)
            actions.append(np.stack([chosen[k].numpy()for k in ACTION_COLUMNS],1))
    return torch.cat(logits).masked_fill(~torch.as_tensor(arrays['action_mask'].astype(bool)),-1e30),np.concatenate(actions)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--baseline',required=True)
    p.add_argument('--capture',required=True);p.add_argument('--native-run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    root=Path(a.run);capture=Path(a.capture);meta=json.loads((capture/'policy-trace.json').read_text())
    for file,key in [('policy-inputs.f16','inputs_sha256'),('policy-aux.f32','aux_sha256')]:
        if hashlib.sha256((capture/file).read_bytes()).hexdigest()!=meta[key]:raise ValueError('Changed capture')
    arrays=decode_packed(np.fromfile(capture/'policy-inputs.f16','<f2').reshape(meta['observation_shape']))
    aux=np.fromfile(capture/'policy-aux.f32','<f4').reshape(meta['aux_shape'])
    datasets=[('gpu_idle',arrays,aux[:,meta['columns']['reset'][0]]>.5,aux[:,meta['columns']['game_tick'][0]]/30)]
    rows=[]
    for row in examples(a.native_run):
        if row['team']!=2:continue
        if row['encoded'].snapshot['time']>30:break
        rows.append(row)
    if not rows:raise ValueError('Empty native idle history')
    native=[canonical_native(row['encoded'].arrays)for row in rows]
    datasets.append(('native_failed_pilot_idle',{k:np.stack([x[k]for x in native])for k in LAYOUT},
                     np.array([(x['sequence']-1)%128==0 for x in rows]),np.array([x['encoded'].snapshot['time']for x in rows])))
    paths={'baseline':Path(a.baseline),**{label:root/label/'diagnostic.pt'for label in ['lr-1e-4','lr-2e-5']}}
    result={'scope':'Offline recorded observations and previous actions; each model replays its own recurrent state under the same recorded reset flags. No closed-loop or strength claim.',
            'weights':{k:hashlib.sha256(v.read_bytes()).hexdigest()for k,v in paths.items()},'datasets':{},'training_report':str(root/'report.json')}
    for label,arrays,resets,times in datasets:
        reference=None;report={}
        for name,path in paths.items():
            model=Policy().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'],strict=True)
            logits,actions=replay(model,arrays,resets)
            if reference is None:reference=logits,actions
            old,old_actions=reference;kl=(old.softmax(-1)*(old.log_softmax(-1)-logits.log_softmax(-1))).sum(-1).numpy();post=times>=0
            changed=actions[:,0]!=old_actions[:,0];moves=np.isin(actions[:,0],[1,2]);indices=np.flatnonzero(changed)
            spatial=post&np.isin(actions[:,0],[1,2])&np.isin(old_actions[:,0],[1,2])
            report[name]={'mean_categorical_KL':float(kl.mean()),'postzero_mean_KL':float(kl[post].mean()),'max_KL':float(kl.max()),
                          'primary_changed':int(changed.sum()),'postzero_primary_changed':int(changed[post].sum()),
                          'postzero_moves':int(moves[post].sum()),'first_changed_time':float(times[indices[0]])if len(indices)else None,
                          'changed_heads':{key:int((actions[:,i]!=old_actions[:,i]).sum())for i,key in enumerate(ACTION_COLUMNS)},
                          'move_coordinates_changed_on_common_moves':int((spatial&np.any(actions[:,4:]!=old_actions[:,4:],axis=1)).sum())}
        result['datasets'][label]={'frames':len(times),'postzero_frames':int((times>=0).sum()),'results':report}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
