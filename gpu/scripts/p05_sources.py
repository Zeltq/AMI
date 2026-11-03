"""Compare Lua getters and protobuf fields on the SAME recorded P03 tick."""
import argparse
import gzip
import json
import math
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import UNIT_MAP
from sf_duel.protos.worldstate_pb2 import CMsgBotWorldState
from sf_duel.spec import UNIT_FIELDS


def audit(root):
    samples={2:{},3:{}}
    for line in (root/'dota.log').read_text(errors='replace').splitlines():
        if line.startswith('SF_P03_OBSERVATION|'):
            s=json.loads(line.split('|',1)[1])['snapshot'];samples[s['team']][round(s['time']*30)]=s
    checks={};frames=[]
    for team in (2,3):
        seen=set()
        with gzip.open(root/f'world-{team}.pb.gz','rb') as f:
            while header:=f.read(4):
                size=struct.unpack('<I',header)[0];payload=f.read(size)
                if len(payload)!=size:raise ValueError('Truncated world packet')
                state=CMsgBotWorldState.FromString(payload);tick=round(state.dota_time*30)
                if tick not in samples[team] or tick in seen:continue
                seen.add(tick);s=samples[team][tick];pairs=[]
                for group,rows in [('heroes',[s['self'],s.get('enemy')]),('creeps',s['creeps']),('towers',s['towers'])]:
                    for index,row in enumerate(rows):
                        if not row or not row.get('visible'):continue
                        candidates=[u for u in state.units if u.name==row['name'] and u.team_id==row['team']]
                        if not candidates:continue
                        u=min(candidates,key=lambda u:math.hypot(u.location.x-row['x'],u.location.y-row['y']))
                        if math.hypot(u.location.x-row['x'],u.location.y-row['y'])>.05:continue
                        pairs.append((group,index))
                        for field,source in UNIT_MAP.items():
                            key=group+'.'+field
                            check=checks.setdefault(key,dict(compared=0,above_one_percent=0,missing_lua=0,missing_proto=0,max_normalized_error=0,examples=[]))
                            if field not in row:check['missing_lua']+=1;continue
                            if not u.HasField(source):check['missing_proto']+=1;continue
                            delta=abs(row[field]-getattr(u,source))/UNIT_FIELDS.get(field,360.)
                            check['compared']+=1;check['above_one_percent']+=int(delta>.01)
                            check['max_normalized_error']=max(check['max_normalized_error'],delta)
                            if delta>.01 and len(check['examples'])<4:check['examples'].append(dict(team=team,time=s['time'],index=index,lua=row[field],protobuf=getattr(u,source)))
                frames.append(dict(team=team,time=s['time'],matched_entities=len(pairs)))
    return dict(scope='Same tick and same visible entity; direct UNIT_MAP fields only, no invented supplement',frames=frames,checks=checks)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.run);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result['checks'].items() if v['above_one_percent'] or v['missing_proto'] or v['missing_lua']},indent=2))
