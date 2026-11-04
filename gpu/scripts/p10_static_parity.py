"""Compare measured native ggd-v4 static fields with the simulator encoder dump."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import records,NativeAdapter
from sf_duel.observations import encode
from sf_duel.combat_observations import CombatMemory
from sf_duel.spec import UNIT_FIELDS
from ggd_dataset import layout

STATS=('health_max','mana_max','armor','attack_damage','attack_interval','attack_point','health_regen','mana_regen','move_speed','attack_range')
TOWER_STATS=('health_max','armor','attack_damage','attack_interval','attack_point','health_regen','attack_range')


def main():
    p=argparse.ArgumentParser();p.add_argument('cohort',type=Path);p.add_argument('initial',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    raw=np.fromfile(a.initial,dtype='<f4').reshape(2,37324);groups,_=layout()
    rows=[];coverage=[]
    for match in json.loads((a.cohort/'matches.json').read_text()):
        team=match['model_team'];run=Path(match['run']);supp={}
        for r in records(run/'native-decisions.pb.gz'):
            if r.HasField('model_supplement') and r.model_supplement.team==team:supp[r.model_supplement.tick]=r.model_supplement
        adapter=NativeAdapter();memory=CombatMemory();selected=False;decisions=0;v4_rows=0
        for r in records(run/'native-decisions.pb.gz'):
            if not r.HasField('decision') or r.decision.team!=team:continue
            m=supp[r.decision.observed_tick]
            s=adapter.snapshot(r,m,observation_profile='ggd-v4')
            arrays=encode(s,combat_inputs=True,combat_memory=memory,observation_profile='ggd-v4').arrays
            decisions+=1;v4_rows+=sum(bool(u.v4_json) for u in m.units)
            if selected or r.decision.observed_dota_time<0:continue
            selected=True
            for group,names in [('heroes',STATS),('towers',TOWER_STATS)]:
                offset,shape,_=groups[group];gpu=raw[team-2,offset:offset+int(np.prod(shape))].reshape(shape).astype(np.float16)
                dota=arrays[group].astype(np.float16)
                for slot in ([0] if group=='heroes' else range(4)):
                    for name in names:
                        i=list(UNIT_FIELDS).index(name);known=bool(gpu[slot,i+len(UNIT_FIELDS)] and dota[slot,i+len(UNIT_FIELDS)])
                        rows.append(dict(team=team,group=group,slot=slot,field=name,known=known,
                                         dota=float(dota[slot,i]),gpu=float(gpu[slot,i]),equal=bool(known and dota[slot,i]==gpu[slot,i]),time=s['time']))
        coverage.append(dict(team=team,decisions=decisions,v4_rows=v4_rows,postzero_selected=selected))
    result=dict(scope='First native postzero observation per side, FP16 static values. Other dynamic vector fields are not claimed equal.',
                coverage=coverage,groups={g:dict(equal=sum(r['equal'] for r in rows if r['group']==g),known=sum(r['known'] for r in rows if r['group']==g),total=sum(r['group']==g for r in rows)) for g in ('heroes','towers')},rows=rows)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result['groups']))


if __name__=='__main__':main()
