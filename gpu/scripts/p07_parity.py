"""Record the exact Lua v4 output and public input for a GPU event parity run."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.enemy_observations import FIELDS,CastMemory
from sf_duel.observations import features
from p07_summary import records

def main():
    p=argparse.ArgumentParser();p.add_argument('log');p.add_argument('fixture');p.add_argument('report');a=p.parse_args()
    rows=[];memory=CastMemory();maximum=0.;fields=0;known_errors=0
    names=['nevermore_shadowraze1','nevermore_shadowraze2','nevermore_shadowraze3','nevermore_requiem']
    for tick in records(a.log,'SF_P07_API|'):
        if tick['team']!=2:continue
        units=[];expected=[];compare=True
        for own in (True,False):
            u=next((u for u in tick['units'] if u['own']==own),{})
            visible=u.get('visible',False)
            spell=names.index(u['active']['value'])+1 if visible and u['active'].get('value') in names and u['IsCastingAbility'].get('value') else 0
            levels=[x['level'].get('value') for x in u.get('abilities',[])]
            stacks=u.get('stacks',{}).get('value',0);remaining=u.get('remaining',{}).get('value',0)
            raw=u.get('v4') or {};want=features(raw,FIELDS)
            got=features(memory.observe(own,tick['game_time'],spell,visible,levels,stacks,remaining),FIELDS)
            maximum=max(maximum,float(np.abs(got-want).max()));fields+=len(want)
            known_errors+=int(np.count_nonzero(got[len(FIELDS):]!=want[len(FIELDS):]))
            units.append(dict(visible=visible,spell=spell,stacks=stacks,remaining=remaining))
            expected.extend(want.tolist())
            if visible and levels!=[4,4,4,1]:compare=False
        rows.append(dict(time=tick['game_time'],units=units,expected=expected,compare=compare))
    Path(a.fixture).write_text(json.dumps(rows),encoding='utf-8')
    result=dict(frames=len(rows),fields=fields,known_mismatches=known_errors,max_abs_error=maximum,tolerance=1e-5,passed=maximum<=1e-5 and known_errors==0,
                source=a.log,scope='Production Lua adapter versus Python CastMemory on measured visible events, fog and debuffs')
    Path(a.report).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
    if not result['passed']:raise RuntimeError('Lua/Python public spell parity failed')
if __name__=='__main__':main()
