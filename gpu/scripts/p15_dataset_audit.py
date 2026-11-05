"""Check every CHiO label, cast restriction, FP16 block and offline reward."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from ggd_dataset import read_block,decode

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.chio_reward import audit_record


def audit(root):
    manifest=json.loads((root/'dataset.json').read_text())
    if manifest['observation_profile']!='chio-v1':raise ValueError('CHiO dataset required')
    frames=0;valid_labels=0;cast_frames=0;cast_labels=0;finite=0;blocks=0
    for entry in manifest['files']:
        header,compact,signals=read_block(root/entry['file'],entry['sha256'])
        if header['version']!=5 or compact.dtype!=np.float16:raise ValueError('CHiO FP16 format required')
        b=decode(header,compact,signals);labels=b['labels'];valid=b['loss_valid'];count=int(valid.sum())
        if count!=entry['valid_frames']:raise ValueError('valid frame count mismatch')
        ok=(labels[:,0]>=0)&(labels[:,0]<13)&(labels[:,1]>=0)&(labels[:,1]<36)
        indices=np.arange(len(labels));ok&=b['action_mask'][indices,labels[:,0]]
        for kind,column,name in [(2,2,'enemy_target_mask'),(3,3,'ally_target_mask')]:
            selected=valid&(labels[:,0]==kind)
            if np.any(labels[selected,column]>=b[name].shape[1]) or np.any(labels[selected,column]<0):raise ValueError('target index outside bucket')
            ok[selected]&=b[name][indices[selected],labels[selected,column]]
        cast=b['hero_combat'][:,0,-50+20]>.5;skills=b['hero_combat'][:,0,-50+21:-50+25].argmax(axis=1)+6
        cast_ok=(labels[:,0]==1)|(labels[:,0]==skills)
        cast_frames+=int((valid&cast).sum());cast_labels+=int((valid&cast&cast_ok).sum())
        frames+=count;valid_labels+=int((valid&ok).sum());finite+=int((valid&np.isfinite(signals.reshape(-1,13)).all(axis=1)).sum());blocks+=1
    rewards=[audit_record(path) for path in sorted(root.glob('chio-reward-*.f32.gz'))]
    reward_frames=sum(r['checked'] for r in rewards)
    return dict(frames=frames,valid_labels=valid_labels,cast_frames=cast_frames,valid_cast_labels=cast_labels,finite_signals=finite,blocks=blocks,
                reward_frames=reward_frames,reward_max_error=max((r['max_error'] for r in rewards),default=None),
                passed=frames>0 and valid_labels==frames and cast_labels==cast_frames and finite==frames and reward_frames==frames and all(r['passed'] for r in rewards))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=audit(a.dataset);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
