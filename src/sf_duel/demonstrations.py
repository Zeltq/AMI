"""Validate causal teacher frames against the existing policy action contract."""
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
from .observations import encode,visible_units
from .spec import Act, blank_action
from .telemetry import unpack_snapshot


def teacher_records(run):
    """Yield the same causal records from either verified on-disk format."""
    run=Path(run);result=json.loads((run/'result.json').read_text())
    name=result['data'].get('file','frames.jsonl.gz')
    previous={}
    with gzip.open(run/name,'rt',encoding='utf-8') as source:
        for line in source:
            row=json.loads(line)
            if result.get('format_version',1)==2:
                s=unpack_snapshot(row);team=s['team']
                action,ref=demonstration_label(s,previous.get(team))
                row={'observation':s,'action':action,'target_ref':ref,
                     'previous':previous.get(team,blank_action())}
                previous[team]=action
            yield row


def demonstration_label(snapshot, previous=None, full_observation_check=False):
    data=snapshot['demonstration'];command=data['command']
    if snapshot.get('errors'):
        raise ValueError(f"Bot API errors: {snapshot['errors']}")
    if command['seq']!=snapshot['seq'] or command['match']!=snapshot['match']:
        raise ValueError('Decision identity mismatch')
    if abs(command['observed_time']-snapshot['time'])>.00001:
        raise ValueError('Observation is not from the decision tick')
    if abs(data['decision_tick']/30-snapshot['time'])>.002:
        raise ValueError('Decision tick and game time disagree')
    if data['decision_tick']%data['period_ticks']:
        raise ValueError('Decision outside the common clock')
    if not data['execution'].get('accepted') or data['execution'].get('seq')!=snapshot['seq']:
        raise ValueError('Executor did not accept the exact decision')
    action=blank_action()
    for key in ('action','buy','item','x','y'):action[key]=int(command[key])
    if not snapshot['legal']['actions'][action['action']] or not snapshot['legal']['buys'][action['buy']]:
        raise ValueError('Illegal action or purchase label')
    if not 0<=action['x']<=128 or not 0<=action['y']<=128:
        raise ValueError('Coordinates outside model action space')
    if data['issued_hero_orders']!=(0 if action['action']==Act.CONTINUE else 1):
        raise ValueError('Unexpected number of executed hero orders')
    if action['action']==Act.ITEM and not snapshot['legal']['items'][action['item']]:
        raise ValueError('Item unavailable in pre-action observation')
    receipts=data['purchase_receipts']
    if receipts and (not action['buy'] or any(row['result']!=-1 for row in receipts)):
        raise ValueError('Purchase receipt does not match a successful buy label')
    if action['buy'] and not receipts:
        raise ValueError('Purchase label has no engine receipt')
    if full_observation_check:encode(snapshot,previous)
    own,enemy,units=visible_units(snapshot)
    refs=[own,enemy]+units+list(snapshot['towers'])+list(snapshot['trees'])+[snapshot['base']]
    targeted=action['action']==Act.ATTACK or (action['action']==Act.ITEM and action['item'] in (3,6))
    ref=None
    if targeted:
        candidates=[(i,r) for i,r in enumerate(refs) if r.get('id')==command.get('target_id')
                    and ((r.get('type')=='tree')==(command.get('target_type')=='tree'))]
        if len(candidates)!=1:raise ValueError('Target absent or ambiguous in pre-action observation')
        action['target'],ref=candidates[0]
        legal=(bool(action['target']>0 and ref.get('attackable')) if action['action']==Act.ATTACK
               else (ref.get('type')=='tree' if action['item']==6 else bool(ref.get('tp'))))
        if not legal:raise ValueError('Target is masked out')
        action['target_offset']=[(ref['x']-snapshot['self']['x'])/2048,(ref['y']-snapshot['self']['y'])/2048]
    if action['action']!=Act.ITEM:action['item']=0
    spatial=action['action'] in (Act.MOVE,Act.MOVE_DIRECT) or (action['action']==Act.ITEM and action['item']==3)
    if not spatial:action['x']=action['y']=64
    return action, ({k:ref[k] for k in ('id','type','name') if k in ref} if ref else None)


class TeacherWriter:
    """Stream both perspectives; reject incomplete sequences instead of hiding gaps."""
    def __init__(self,path,hz):
        self.path=Path(path);self.hz=hz;self.previous={};self.last={}
        self.counts=Counter();self.actions=Counter();self.purchases=Counter()
        self.file=gzip.open(self.path,'wt',encoding='utf-8',compresslevel=1)

    def append(self,s):
        team=s['team'];data=s['demonstration'];period=data['period_ticks']
        if abs(30/period-self.hz)>.00001:raise ValueError('Teacher and requested policy frequency differ')
        if team in self.last:
            seq,tick=self.last[team]
            if s['seq']!=seq+1 or data['decision_tick']!=tick+period:
                raise ValueError(f'Missing or duplicate decision for team {team}')
        roster=s['probe']['allied_heroes']
        if len(roster)!=1 or roster[0]['name']!='npc_dota_hero_nevermore':
            raise ValueError('Not a solo SF duel')
        label,ref=demonstration_label(s,self.previous.get(team))
        record={'observation':s,'action':label,'target_ref':ref,'previous':self.previous.get(team,blank_action())}
        self.file.write(json.dumps(record,separators=(',',':'))+'\n')
        self.previous[team]=label;self.last[team]=(s['seq'],data['decision_tick'])
        self.counts[str(team)]+=1;self.actions[str(label['action'])]+=1;self.purchases[str(label['buy'])]+=1

    def close(self,terminal_time=None):
        self.file.close()
        if terminal_time is not None and any(tick/30>=terminal_time-.001 for _,tick in self.last.values()):
            temporary=self.path.with_name(self.path.name+'.trimmed')
            self.counts.clear();self.actions.clear();self.purchases.clear()
            try:
                with gzip.open(self.path,'rt',encoding='utf-8') as source, gzip.open(temporary,'wt',encoding='utf-8',compresslevel=1) as target:
                    for line in source:
                        row=json.loads(line)
                        if row['observation']['time']>=terminal_time-.001:continue
                        target.write(line)
                        self.counts[str(row['observation']['team'])]+=1
                        self.actions[str(row['action']['action'])]+=1
                        self.purchases[str(row['action']['buy'])]+=1
                os.replace(temporary,self.path)
            finally:temporary.unlink(missing_ok=True)
        with self.path.open('rb') as source:digest=hashlib.file_digest(source,'sha256').hexdigest()
        return {'perspectives':dict(self.counts),'actions':dict(self.actions),'purchases':dict(self.purchases),
                'frames':sum(self.counts.values()),'sha256':digest,'bytes':self.path.stat().st_size}
