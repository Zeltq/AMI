"""CHiO extension of honest v4 Bot API snapshots; identity-based missing history."""
import math
import numpy as np
import torch
from .observations import encode
from .combat_observations import CombatMemory

ITEM_IDS=[0,1,2,5,4,6]


class Observer:
    def __init__(self):
        self.history={};self.episode=None;self.tick=None;self.enemy_identity=None

    def encode(self,s,previous=None,pending_cast=None):
        tick=s['tick'];episode=s['match']
        if episode!=self.episode:self.history={};self.tick=None;self.episode=episode;self.enemy_identity=None
        if self.tick is not None and tick<=self.tick:raise ValueError('Repeated observation')
        delta=1 if self.tick is None else max(1,(tick-self.tick)//6)
        self.tick=tick
        for history in self.history.values():
            history[:]=([None]*min(12,delta)+history)[:12]
        # The legacy collector supplies v4 fields. CombatMemory is intentionally
        # frame-local here: hidden projectile casters must not reuse old damage.
        encoded=encode(s,combat_memory=CombatMemory(),observation_profile='ggd-v4')
        own=s['self'];enemy=s.get('enemy') or {};units=s['creeps'];towers=s['towers']
        if enemy.get('id') is not None:self.enemy_identity=(enemy['id'],enemy.get('generation',0))
        elif self.enemy_identity is not None:
            enemy=dict(id=self.enemy_identity[0],generation=self.enemy_identity[1],visible=False)
        def key(u):return (u.get('id'),u.get('generation',0))
        def seen(u):return bool(u) and u.get('visible',False) and u.get('health',0)>0
        def zone(source,target,reach,radius=250):
            angle=math.radians(source['facing']);x=source['x']+reach*math.cos(angle);y=source['y']+reach*math.sin(angle)
            return float((target['x']-x)**2+(target['y']-y)**2<=radius**2)
        def extension(u,hero):
            width=25 if hero else 17;v=np.zeros(width,np.float32);known=np.zeros(width,np.float32)
            if u.get('id') is not None:
                h=self.history.setdefault(key(u),[None]*12)
                if seen(u) and u.get('health_max',0)>0:h[0]=min(1,max(0,u['health']/u['health_max']))
                for i,x in enumerate(h):
                    if x is not None:v[i]=x;known[i]=1
            if seen(u):
                for i,r in enumerate((200,450,700)):
                    rule=(s.get('raze_rules') or [dict(range=x,radius=250) for x in (200,450,700)])[i]
                    if rule.get('range',0)>0 and rule.get('radius',0)>0:
                        v[12+i]=zone(own,u,rule['range'],rule['radius']);known[12+i]=1
                dx=u['x']-own['x'];dy=u['y']-own['y'];distance=math.hypot(dx,dy)
                if own.get('hull') is not None and u.get('hull') is not None:
                    v[15]=distance<=own['attack_range']+own['hull']+u['hull'];known[15]=1
                v[16]=((math.atan2(dy,dx)-math.radians(own['facing'])+math.pi)%(2*math.pi)-math.pi)/math.pi if distance else 0
                known[16]=1
            if hero and u is own:
                if seen(enemy):
                    for i,r in enumerate((200,450,700)):v[17+i]=zone(enemy,own,r);known[17+i]=1
                known[20:25]=1
                if pending_cast is not None:v[20]=1;v[21+pending_cast-6]=1
            return np.concatenate((v,known))
        b=encoded.arrays
        for records,name,hero in [([own,enemy],'hero_combat',True),(units,'creep_combat',False),(towers,'tower_combat',False)]:
            width=50 if hero else 34
            extra=np.asarray([extension(u,hero) for u in records],np.float32).reshape(len(records),width)
            b[name]=np.concatenate((b[name],extra),-1)
        for name in ('heroes','creeps','towers'):b[name][:,[38,86]]=0
        for name in ('items','enemy_items'):b[name]=b[name][ITEM_IDS]
        previous=previous or {'action':1,'direction':0};kind=previous['action']
        pv=np.zeros(55,np.float32);pv[kind]=1
        if kind==0:pv[13+previous['direction']]=1
        pv[49+(kind-1 if 2<=kind<=5 else 5 if kind==12 else 0)]=1;b['previous']=pv
        enemy_mask=np.asarray([seen(u) and u['team']!=s['team'] for u in units],bool)
        ally_mask=np.asarray([seen(u) and u['team']==s['team'] and u['health']<=u['health_max']*.5 for u in units],bool)
        mask=np.zeros(13,bool);mask[:2]=True;mask[2]=enemy_mask.any();mask[3]=ally_mask.any();mask[4]=seen(enemy)
        tower_index=2 if s['team']==2 else 0;mask[5]=seen(towers[tower_index])
        for i,skill in enumerate(s['skills']):mask[6+i]=bool(skill.get('castable')) and skill.get('level',0)>0
        for kind,item in ((10,3),(11,4),(12,5)):
            row=s['items'][ITEM_IDS[item]];mask[kind]=bool(row.get('castable')) and 0<=row.get('slot',-1)<=5
        mask[10]&=not own.get('flask',False);mask[12]&=bool(s['trees'])
        if pending_cast is not None:mask[:]=False;mask[1]=mask[pending_cast]=True
        b['action_mask']=mask;b['enemy_target_mask']=enemy_mask;b['ally_target_mask']=ally_mask
        # Pad empty sets only. Attention indices retain the snapshot's identity order.
        for group,keys in [('creeps',('creeps','creep_combat','creep_valid','creep_ally','enemy_target_mask','ally_target_mask')),
                           ('trees',('trees','tree_valid')),('projectiles',('projectiles','projectile_combat','projectile_valid'))]:
            if len(b[group])==0:
                for name in keys:b[name]=np.zeros((1,*b[name].shape[1:]),dtype=b[name].dtype)
        return {name:torch.from_numpy(value[None]) for name,value in b.items()}


def command(s,action,sequence):
    kind=action['action'];own=s['self']
    d=dict(action=kind,sequence=sequence,observed_tick=s['tick'],execute_tick=s['tick']+7,player=s['player'],team=s['team'])
    if 6<=kind<=9:d['continuation']=s.get('pending_cast')==kind
    if kind==0:
        a=action['direction']*math.pi/18;d.update(direction=action['direction'],
            x=max(-8192,min(8192,own['x']+2000*math.cos(a))),y=max(-8192,min(8192,own['y']+2000*math.sin(a))))
    elif kind in (2,3):d['target']=s['creeps'][action['enemy_target' if kind==2 else 'ally_target']]['id']
    elif kind==4:d['target']=s['enemy']['id']
    elif kind==5:d['target']=s['towers'][2 if s['team']==2 else 0]['id']
    elif kind==12:d['target']=min(s['trees'],key=lambda t:((t['x']-own['x'])**2+(t['y']-own['y'])**2,t['id']))['id']
    return d
