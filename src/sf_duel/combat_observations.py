"""GGD combat inputs from visible identities and causal last-seen information."""
import math
import numpy as np
from .spec import COMBAT_UNIT_FIELDS, COMBAT_PROJECTILE_FIELDS

# Current calibrated attack/armor type table; callers can pin another table.
COMBAT=(1,.5,1,1,.5, 1,.7,1,1,.7, .75,.7,1,1,.7, .5,.35,1.5,1.5,.35, 1,2.5,1,1,1)
SPREAD=(3.,2.,2.,2.5,5.5)

def kind(u):
    if u.get('type')=='hero':return 0
    if u.get('type')=='tower' or 'tower' in u.get('name',''):return 1
    if u.get('creep_siege') or 'siege' in u.get('name',''):return 4
    if u.get('creep_ranged') or 'ranged' in u.get('name',''):return 3
    return 2

class CombatMemory:
    def __init__(self,combat=COMBAT,spread=SPREAD):
        self.combat=tuple(combat);self.spread=tuple(spread);self.match=None;self.seen={}

def encode_combat(snapshot,own,enemy,units,towers,projectiles,memory):
    if memory.match!=snapshot.get('match'):
        memory.seen.clear();memory.match=snapshot.get('match')
    entities=[own,enemy]+units+towers
    visible={u['id']:u for u in entities if u.get('id') is not None and u.get('health',0)>0}
    for identity,u in visible.items():
        if u.get('attack_damage') is not None:memory.seen[identity]=(u['attack_damage'],kind(u))
    def dealt(raw,source_kind,target):
        armor=target.get('armor')
        if armor is None:return None
        return math.ceil(raw*memory.combat[source_kind*5+kind(target)]*(1-.06*armor/(1+.06*abs(armor))))
    def caster(p):
        return memory.seen.get(p.get('caster_id'))
    complete=all(p.get('target_id') is not None for p in projectiles)
    rows=[]
    for u in entities:
        row={}
        if u.get('id') in visible:
            target=visible.get(u.get('attack_target_id'))
            if target:
                row.update(attack_target_dx=target['x']-own['x'],attack_target_dy=target['y']-own['y'],attack_target_health=target['health'])
            row['damage_spread']=u.get('damage_spread',memory.spread[kind(u)])
            attackers=[v for v in visible.values() if v.get('attack_target_id')==u['id']]
            amounts=[dealt(v.get('attack_damage',0),kind(v),u) for v in attackers]
            row['incoming_attackers']=len(attackers)
            if all(d is not None for d in amounts):
                row['incoming_attack_damage']=sum(amounts)
                if all(v.get('attack_interval') is not None for v in attackers):row['incoming_attack_dps']=sum(d/max(.1,v['attack_interval']) for d,v in zip(amounts,attackers))
            if complete:
                incoming=[p for p in projectiles if p['target_id']==u['id']]
                row['incoming_projectiles']=len(incoming)
                amounts=[dealt(*caster(p),u) if caster(p) else None for p in incoming]
                if all(d is not None for d in amounts):row['incoming_projectile_damage']=sum(amounts)
                timed=[(max(0,math.hypot(u['x']-p['x'],u['y']-p['y'])-8)/p['speed'],d) for p,d in zip(incoming,amounts) if p.get('speed',0)>0 and 'x' in p and 'y' in p]
                if timed:
                    eta,d=min(timed,key=lambda x:x[0]);row['next_projectile_eta']=eta
                    if d is not None:row['next_projectile_damage']=d
        rows.append(row)
    projectile_rows=[]
    for p in projectiles:
        row={};target=visible.get(p.get('target_id'));source=visible.get(p.get('caster_id'));cached=caster(p)
        if target:row.update(target_dx=target['x']-own['x'],target_dy=target['y']-own['y'],target_health=target['health'],target_team_relative=1 if target['team']==own['team'] else -1)
        if cached:row.update(caster_damage=cached[0],caster_kind=cached[1])
        if source:row.update(caster_dx=source['x']-own['x'],caster_dy=source['y']-own['y'])
        projectile_rows.append(row)
    def features(row,fields):
        return [float(row.get(k,0))/scale for k,scale in fields.items()]+[float(k in row) for k in fields]
    def array(records,fields):
        return np.asarray([features(r,fields) for r in records],dtype=np.float32).reshape(len(records),2*len(fields))
    return dict(hero_combat=array(rows[:2],COMBAT_UNIT_FIELDS),creep_combat=array(rows[2:2+len(units)],COMBAT_UNIT_FIELDS),tower_combat=array(rows[2+len(units):],COMBAT_UNIT_FIELDS),projectile_combat=array(projectile_rows,COMBAT_PROJECTILE_FIELDS))
