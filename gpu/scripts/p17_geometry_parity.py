"""Reconstruct reproducible geometry from saved simulator inputs; CPU only."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from sf_duel.chio_observations import Observer
from sf_duel.spec import UNIT_FIELDS,TOWERS
from fixtures import snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.fixture/'manifest.json').read_text());b={n:np.fromfile(a.fixture/e['file'],dtype='<f4').reshape(e['shape']) for n,e in m['tensors'].items()}
    checked=0;maximum=0.;failures=[]
    def unit(values,identity,team,kind):
        row={name:float(values[i])*scale for i,(name,scale) in enumerate(UNIT_FIELDS.items()) if values[i+48]>.5}
        row.update(id=identity,team=team,type=kind,visible=True)
        row['facing']=math.degrees(math.atan2(row.get('facing_sin',0),row.get('facing_cos',1)))
        row['hull']=24 if kind=='hero' else 144 if kind=='tower' else 8 if row.get('creep_ranged') else 16
        return row
    for t in range(len(b['heroes'])):
        s=snapshot();s.update(tick=t*6,match='geometry-'+str(t));s['self']=unit(b['heroes'][t,0],1,2,'hero')
        s['enemy']=unit(b['heroes'][t,1],2,3,'hero') if b['hero_valid'][t,1] else {}
        s['creeps']=[];creep_indices=[]
        for i,row in enumerate(b['creeps'][t]):
            if b['creep_valid'][t,i]:s['creeps'].append(unit(row,3+i,2 if b['creep_ally'][t,i] else 3,'creep'));creep_indices.append(i)
        s['towers']=[]
        for i,row in enumerate(b['towers'][t]):
            u=unit(row,100+i,2 if i<2 else 3,'tower');u['name']=TOWERS[i];u['visible']=bool(row[48+7]);s['towers'].append(u)
        # This fixture checks only the five geometry fields, not full world dynamics.
        s['legal']['actions']=[True,True,True,False,True,True,False,False,False,False,False]
        actual=Observer().encode(s)
        for group,indices,width in [('hero_combat',[0,1],25),('creep_combat',creep_indices,17),('tower_combat',[0,1,2,3],17)]:
            for j,original in enumerate(indices):
                expected=b[group][t,original,-2*width:];computed=actual[group][0,j,-2*width:].numpy()
                for field in range(12,17):
                    if expected[width+field]<.5:continue
                    checked+=1;error=abs(float(expected[field])-float(computed[field]));maximum=max(maximum,error)
                    # Saved observation values are FP16; compare at that boundary.
                    rounded=float(np.float16(computed[field]));difference=abs(float(expected[field])-rounded)
                    if difference>1e-3 or computed[width+field]!=1:failures.append(dict(step=t,group=group,slot=original,field=field,error=difference))
    result=dict(checked=checked,passed=checked-len(failures),max_unrounded_error=maximum,failures=failures,
                source=str(a.fixture),scope='five geometry fields per observed unit; reconstructed saved simulator state, no GPU and no policy evaluation')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='failures'}))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
