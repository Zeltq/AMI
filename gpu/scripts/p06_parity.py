"""Compare the new static/0:00 fields and export a validation-only IL fixture."""
import argparse
import json
from pathlib import Path
import numpy as np
from ggd_dataset import read_block,decode
from sf_duel.observations import encode
from sf_duel.timing_observations import TIMING_FIELDS
from sf_duel.spec import UNIT_FIELDS,COMBAT_UNIT_FIELDS


def main():
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('dota_log',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.dataset/'dataset.json').read_text());sim={};fixture=None;extension=[]
    for entry in manifest['files']:
        h,o,s=read_block(a.dataset/entry['file'],entry['sha256'])
        if fixture is None:
            for i,actor in enumerate(h['actors']):
                if actor//2%8==6 and np.any((s[i,:,0]==3)&(s[i,:,11]>.5)):
                    fixture=decode(h,o[i:i+1],s[i:i+1]);np.savez_compressed(a.output/'validation.npz',**fixture)
                    (a.output/'fixture-source.json').write_text(json.dumps(dict(file=entry['file'],actor=actor,split='validation'),indent=2));break
        for i,actor in enumerate(h['actors']):
            if actor//2!=0:continue
            arr=decode(h,o[i:i+1],s[i:i+1])
            time=arr['global_features'][:,0]*600
            for phase,select in [('static',time<-89),('zero',(time>=0)&(time<.5))]:
                for t in np.flatnonzero(select & arr['loss_valid']):sim.setdefault((phase,actor%2),{k:v[t] for k,v in arr.items()})
            extension.append(arr['creep_combat'][...,2*len(COMBAT_UNIT_FIELDS):])
    native={}
    with a.dota_log.open(encoding='utf-8',errors='replace') as f:
        for line in f:
            if 'SF_P05_PHASE|' not in line:continue
            r=json.loads(line.split('SF_P05_PHASE|',1)[1]);s=r['snapshot'];t=s['time']
            phase='static' if t<-89 else 'zero' if 0<=t<.5 else None
            if phase:native.setdefault((phase,s['team']-2),encode(s,r['previous'],observation_profile='ggd-v3').arrays)
    fields=list(TIMING_FIELDS);at=2*len(COMBAT_UNIT_FIELDS);rows=[]
    for key,n in native.items():
        if key not in sim:continue
        g=sim[key]
        # Compare own SF, named tower slots and the first eight fresh creeps
        # sorted by side/type; do not compare units from different trajectories.
        pairs=[('heroes','hero_combat',0,0)]
        pairs.extend(('towers','tower_combat',i,i) for i in range(4))
        def creep_key(row):
            return tuple(row[list(UNIT_FIELDS).index(k)] for k in ('team_relative','creep_melee','creep_ranged'))
        ni=sorted(range(len(n['creeps'])),key=lambda i:creep_key(n['creeps'][i]))
        gi=sorted(np.flatnonzero(g['creep_valid']),key=lambda i:creep_key(g['creeps'][i]))
        if len(ni)==len(gi):pairs.extend(('creeps','creep_combat',i,j) for i,j in zip(ni,gi))
        for group,extra,i,j in pairs:
            for index,field in enumerate(fields[:3]):
                nv=n[extra][i,at+index];gv=g[extra][j,at+index]
                nk=n[extra][i,at+len(fields)+index];gk=g[extra][j,at+len(fields)+index]
                rows.append(dict(phase=key[0],side=key[1],group=group,slot=int(i),field=field,dota=float(nv),sim=float(gv),dota_known=int(nk),sim_known=int(gk),equal=bool(np.float16(nv)==np.float16(gv) and nk==gk)))
            c=list(UNIT_FIELDS).index('attack_cycle')
            nv,nk=n[group][i,[c,len(UNIT_FIELDS)+c]];gv,gk=g[group][j,[c,len(UNIT_FIELDS)+c]]
            rows.append(dict(phase=key[0],side=key[1],group=group,slot=int(i),field='attack_cycle',dota=float(nv),sim=float(gv),dota_known=int(nk),sim_known=int(gk),equal=bool(np.float16(nv)==np.float16(gv) and nk==gk)))
    report=dict(pairs=len(set((r['phase'],r['side'],r['group'],r['slot']) for r in rows)),equal=sum(r['equal'] for r in rows),fields=len(rows),rows=rows,fixture_created=fixture is not None)
    (a.output/'parity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
