"""P05 fixed-phase audit of recorded P03 Lua and actual FP16 simulator inputs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from compare_initial_observation import observation_layout, padded_arrays, align_rows, group_fields, field_summary
from ggd_dataset import read_block
from sf_duel.observations import encode
from sf_duel.combat_observations import CombatMemory


def native_frames(path):
    frames={2:[],3:[]};memory={2:CombatMemory(),3:CombatMemory()}
    for line in path.read_text(errors='replace').splitlines():
        if not line.startswith('SF_P03_OBSERVATION|'):continue
        value=json.loads(line.split('|',1)[1]);s=value['snapshot'];team=s['team']
        arrays=encode(s,value.get('previous'),combat_inputs=True,combat_memory=memory[team]).arrays
        frames[team].append((s,arrays))
    return frames


def gpu_frames(root, until=1.4):
    layout,width=observation_layout('ggd-v2');frames={2:[],3:[]}
    manifest=json.loads((root/'dataset.json').read_text())
    for entry in manifest['files']:
        header,compact,signals=read_block(root/entry['file'])
        for actor_index,actor in enumerate(header['actors']):
            if actor not in (0,1):continue
            team=actor+2
            for step in range(header['horizon']):
                if signals[actor_index,step,11]<.5:continue
                flat=np.zeros(width,np.float32);flat[header['columns']]=compact[actor_index,step]
                groups={};offset=0
                for key,shape in layout.items():
                    size=int(np.prod(shape));groups[key]=flat[offset:offset+size].reshape(shape);offset+=size
                t=float(groups['global_features'][0]*600)
                if not frames[team] or -.15<=t<=until+.15:frames[team].append((t,groups))
        if all(frames[t] and frames[t][-1][0]>=until for t in (2,3)):break
    return frames


def decoded_rows(arrays,key):
    fields=group_fields(key);names=list(fields);n=len(names)
    rows=arrays[key].reshape(-1,2*n)
    valid={'heroes':'hero_valid','creeps':'creep_valid','trees':'tree_valid','projectiles':'projectile_valid'}.get(key)
    return [{name:float(row[i]*fields[name]) if row[i+n]>.5 else None for i,name in enumerate(names)}
            for j,row in enumerate(rows) if not valid or arrays[valid][j]]


def audit(lua,gpu):
    native=native_frames(lua);sim=gpu_frames(gpu);layout,_=observation_layout('ggd-v2')
    pairs=[]
    for team in (2,3):
        for index,(snapshot,arrays) in enumerate(native[team]):
            t,actual=sim[team][0] if index==0 else min(sim[team][1:],key=lambda row:abs(row[0]-snapshot['time']))
            expected=padded_arrays(arrays,layout)
            raw={key:dict(dota=decoded_rows(expected,key),gpu=decoded_rows(actual,key))
                 for key in ('heroes','creeps','towers','items','global_features')}
            alignment=align_rows(expected,actual)
            differences={};mask_counts={}
            for key in layout:
                fields=group_fields(key)
                if not fields:continue
                entries=field_summary(np.abs(actual[key]-expected[key]),fields,actual[key],expected[key])
                differences[key]=entries
                n=len(fields)
                mask_counts[key]=dict(dota_known=int((expected[key][...,n:]>.5).sum()),
                                      gpu_known=int((actual[key][...,n:]>.5).sum()),
                                      mismatched=int(((actual[key][...,n:]>.5)!=(expected[key][...,n:]>.5)).sum()),
                                      slots=int(actual[key][...,n:].size))
            pairs.append(dict(team=team,index=index,phase='initial' if index==0 else 'first_wave',
                              dota_time=snapshot['time'],gpu_time=t,alignment=alignment,
                              raw=raw,differences=differences,mask_counts=mask_counts))
    unique=lambda selected,flag: sorted({key+'.'+e['field'] for p in selected for key,entries in p['differences'].items() for e in entries if e[flag]})
    return dict(method='First frame and nearest recorded game time after 0:00; raw rows retained before spatial assignment. Values require both known bits. Initial frame deliberately retains purchase-transfer transient.',
                denominator_group_fields=sum(len(group_fields(k) or {}) for k in layout),
                threshold=.01,all_value_fields=unique(pairs,'above_one_percent'),
                postzero_value_fields=unique([p for p in pairs if p['index']>0],'above_one_percent'),
                all_known_fields=unique(pairs,'known_mismatches'),pairs=pairs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--lua-log',type=Path,required=True)
    p.add_argument('--gpu-dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.lua_log,a.gpu_dataset)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'}))


if __name__=='__main__':main()
