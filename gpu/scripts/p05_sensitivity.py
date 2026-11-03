"""CPU diagnostic: action-distribution sensitivity to GPU's zero animation fields.

Each counterfactual shares the real trajectory, issued previous command and
BASELINE recurrent state. This is local input sensitivity, never a match result.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import records,NativeAdapter,action_label
from sf_duel.observations import encode,batch
from sf_duel.combat_observations import CombatMemory
from sf_duel.model import Policy
from sf_duel.spec import UNIT_FIELDS


def run(checkpoint,cohort,limit):
    torch.set_num_threads(1)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=Policy(combat_inputs=True).eval();model.load_state_dict(saved['state_dict'],strict=True)
    matches=json.loads((cohort/'matches.json').read_text());selected=[]
    for team in (2,3):selected.append(next(m for m in matches if m['model_team']==team))
    results=[];field=list(UNIT_FIELDS).index('attack_cycle')
    for match in selected:
        path=Path(match['run'])/'native-decisions.pb.gz';team=match['model_team'];supplements={}
        for r in records(path):
            if r.HasField('model_supplement') and r.model_supplement.team==team:supplements[r.model_supplement.tick]=r.model_supplement
        adapter=NativeAdapter();combat=CombatMemory();previous=None;state=model.initial_state(1);count=0
        sums={}
        with torch.inference_mode():
            for r in records(path):
                if not r.HasField('decision') or r.decision.team!=team:continue
                d=r.decision
                if d.sequence!=count+1:raise ValueError('Non-contiguous sequence')
                if count%128==0:state=model.initial_state(1)
                s=adapter.snapshot(r,supplements[d.observed_tick]);e=encode(s,previous,combat_inputs=True,combat_memory=combat)
                b=batch([e]*5,'cpu');b={k:v.half().float() if v.is_floating_point() else v for k,v in b.items()}
                groups=('heroes','creeps','towers')
                for i,group in enumerate(groups,1):b[group][i,:,field]=0
                for group in groups:b[group][4,:,field]=0
                memory=tuple(v.expand(-1,5,-1).contiguous() for v in state)
                h,_,_,next_state=model(b,memory)
                probabilities=model.action_head(h).masked_fill(~b['action_mask'],torch.finfo(torch.float32).min).softmax(-1)
                if not torch.isfinite(probabilities).all():raise ValueError('Nonfinite probability')
                baseline=probabilities[0];period='pregame' if s['time']<0 else 'after_zero'
                for i,name in enumerate((*groups,'all_cycles'),1):
                    key=period+'/'+name;row=sums.setdefault(key,dict(decisions=0,tv_sum=0.,max_tv=0.,tv_over_one_percent=0,argmax_changed=0))
                    tv=float((probabilities[i]-baseline).abs().sum()/2)
                    row['decisions']+=1;row['tv_sum']+=tv;row['max_tv']=max(row['max_tv'],tv)
                    row['tv_over_one_percent']+=int(tv>.01);row['argmax_changed']+=int(probabilities[i].argmax()!=baseline.argmax())
                state=tuple(v[:,0:1].contiguous() for v in next_state)
                previous=action_label(d,e);count+=1
                if count>=limit:break
        for row in sums.values():row['mean_tv']=row['tv_sum']/row['decisions']
        results.append(dict(index=match['index'],team=team,decisions=count,last_time=s['time'],sensitivity=sums))
    return dict(games=results,device='cpu',method=__doc__,checkpoint=str(checkpoint),max_decisions=limit)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--max-decisions',type=int,default=1200);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    started=time.monotonic();result=run(a.checkpoint,a.cohort,a.max_decisions);result['wall_minutes']=(time.monotonic()-started)/60
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
