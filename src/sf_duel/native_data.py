"""Native protobuf -> the existing policy contract. No teacher decisions here."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import struct
import numpy as np
from .protos.duel_pb2 import Record
from .spec import *
from .observations import encode, target_ref

ITEM_IDS=dict(zip(ITEMS,(75,29,73,46,216,39,44,38)))
SPELL_IDS=dict(zip(SPELLS,(5059,5060,5061,5064)))
UNIT_MAP={'health':'health','health_max':'health_max','mana':'mana','mana_max':'mana_max',
 'health_regen':'health_regen','mana_regen':'mana_regen','move_speed':'current_movement_speed',
 'facing':'facing','attack_damage':'attack_damage','attack_range':'attack_range',
 'attack_point':'attack_anim_point','projectile_speed':'attack_projectile_speed','armor':'armor',
 'magic_resist':'magic_resist','level':'level','attack_cycle':'anim_cycle'}

def records(path):
 with gzip.open(path,'rb') as f:
  while header:=f.read(4):
   if len(header)!=4:raise ValueError('Truncated native header')
   size=struct.unpack('<I',header)[0]
   if not 0<size<32*1024*1024:raise ValueError('Invalid native record size')
   data=f.read(size)
   if len(data)!=size:raise ValueError('Truncated native payload')
   yield Record.FromString(data)

def lane_creep(u):return u.name.startswith('npc_dota_creep_') or u.name in ('npc_dota_goodguys_siege','npc_dota_badguys_siege')
def position(p):return dict(x=float(p.x),y=float(p.y),z=float(p.z))
def modifier(u,name):return next((m for m in u.modifiers if m.name==name),None)

class NativeAdapter:
 def __init__(self):self.tower_positions={};self.tower_handles={}
 def snapshot(self,row,model):
  state=row.observation;d=row.decision;team=d.team
  if model is None:raise ValueError('Missing model supplement; recollect with native_model_observation.lua')
  if model.team!=team or model.tick!=d.observed_tick:raise ValueError('Misaligned model supplement')
  if len(model.terrain)!=192 or len(model.skills)!=4 or len(model.items)!=8 or len(model.towers)!=4:raise ValueError('Incomplete model supplement')
  own=next(u for u in state.units if u.name=='npc_dota_hero_nevermore' and u.team_id==team)
  if math.hypot(model.x-own.location.x,model.y-own.location.y)>.05:raise ValueError('Terrain origin differs from native hero position')
  extras={u.handle:u for u in model.units};extras[own.handle]=extras.get(0)
  def unit(u):
   result={'id':u.handle,'name':u.name,'team':u.team_id,'visible':True,**position(u.location)}
   for target,source in UNIT_MAP.items():
    if u.HasField(source):result[target]=getattr(u,source)
   extra=extras.get(u.handle)
   if extra is not None:result.update(attack_interval=extra.attack_interval,is_attacking=extra.attacking,is_casting_ability=extra.casting,is_channeling=extra.channeling,is_using_ability=extra.using_ability)
   else:
    for key,source in [('is_channeling','is_channeling'),('is_using_ability','is_using_ability')]:
     if u.HasField(source):result[key]=getattr(u,source)
   if u.handle==own.handle:
    # Native state is sampled after the Lua callback: use its current flags
    # rather than the earlier auxiliary flags when the engine supplies them.
    if u.HasField('action_type'):result['is_attacking']=u.action_type==4
    for key in ('is_using_ability','is_channeling'):
     if u.HasField(key):result[key]=getattr(u,key)
    phases=[a.is_in_ability_phase for a in u.abilities if a.HasField('is_in_ability_phase')]
    if phases:result['is_casting_ability']=any(phases)
   for key,name in [('fear','modifier_nevermore_requiem_fear'),('flask','modifier_flask_healing'),('clarity','modifier_clarity_potion'),('tango','modifier_tango_heal')]:result[key]=modifier(u,name) is not None
   m=modifier(u,'modifier_nevermore_shadowraze_debuff');result['raze_stacks']=m.stack_count if m else 0
   if u.handle==own.handle:
    for field,source in [('last_hits','last_hits'),('denies','denies'),('xp_to_next','xp_needed_to_level')]:
     if u.HasField(source):result[field]=getattr(u,source)
    if u.reliable_gold>=0 and u.unreliable_gold>=0:result['gold']=u.reliable_gold+u.unreliable_gold
    souls=modifier(u,'modifier_nevermore_necromastery');result['souls']=souls.stack_count if souls else 0
    fear=modifier(u,'modifier_nevermore_requiem_fear');result['fear_remaining']=fear.remaining_duration if fear else 0
    result['raze_remaining']=m.remaining_duration if m else 0
    active=next((a.ability_id for a in u.abilities if a.handle==u.active_ability_handle),None)
    result['active_spell']=next((i+1 for i,name in enumerate(SPELLS) if SPELL_IDS[name]==active),0)
   result['attack_target_id']=u.attack_target_handle
   result['attack_target_self']=u.attack_target_handle==own.handle
   result['attack_target_enemy']=any(v.handle==u.attack_target_handle and v.name=='npc_dota_hero_nevermore' and v.team_id!=team for v in state.units)
   result['attackable']=u.is_alive and not u.is_attack_immune and not u.is_invulnerable and (u.team_id!=team or (lane_creep(u) and u.health<=u.health_max*.5))
   return result
  hero=unit(own);hero['type']='hero'
  enemy_unit=next((u for u in state.units if u.name=='npc_dota_hero_nevermore' and u.team_id!=team and u.is_alive),None)
  enemy=unit(enemy_unit) if enemy_unit else None
  if enemy:enemy['type']='hero'
  creeps=[]
  for u in sorted(state.units,key=lambda u:u.handle):
   if lane_creep(u) and u.team_id in (2,3) and u.is_alive:
    v=unit(u);v.update(type='creep',creep_melee='melee' in u.name,creep_ranged='ranged' in u.name,creep_siege='siege' in u.name);creeps.append(v)
  towers=[]
  for i,name in enumerate(TOWERS):
   side=2 if i<2 else 3;p=model.towers[i]
   if p.x or p.y:self.tower_positions[name]=position(p)
   if name not in self.tower_positions:raise ValueError('Missing public tower position')
   u=next((u for u in state.units if u.name==name),None)
   if u:self.tower_handles[name]=u.handle
   v=unit(u) if u else {'id':self.tower_handles.get(name,-2-i),'name':name,'team':side,'visible':False,**self.tower_positions[name]}
   v.update(type='tower',tp=side==team and u is not None and u.is_alive)
   if side==team and u is None:v.update(visible=True,health=0,attackable=False)
   towers.append(v)
  aux=row.supplement
  if aux.team!=team or aux.tick>d.observed_tick:raise ValueError('Missing or future teacher supplement')
  trees=[{'id':v.id,'type':'tree',**position(v)} for v in aux.trees]
  projectiles=[];seen=set()
  relevant={own.handle,*[u['id'] for u in creeps]}
  if enemy_unit:relevant.add(enemy_unit.handle)
  for u in state.units:
   if u.handle not in relevant:continue
   for shot in u.incoming_tracking_projectiles:
    key=(shot.handle,u.handle)
    if key in seen:continue
    seen.add(key);caster=next((v for v in state.units if v.handle==shot.caster_handle),None)
    projectile={**position(shot.location),'speed':shot.velocity,'is_attack':shot.is_attack,'target_self':u.handle==own.handle}
    if caster is not None:projectile['caster_enemy']=caster.team_id!=team
    projectiles.append(projectile)
  skills=[]
  for name,meta in zip(SPELLS,model.skills):
   if name!=meta.name:raise ValueError('Ability slot order differs')
   a=next((a for a in own.abilities if a.ability_id==SPELL_IDS[name]),None)
   if a is None:raise ValueError('Own ability missing')
   skills.append(dict(level=a.level,cooldown=a.cooldown_remaining,mana_cost=meta.mana_cost,cast_point=meta.cast_point,channel_time=meta.channel_time,castable=a.is_fully_castable and a.level>0))
  items=[];enemy_items=[]
  for index,name in enumerate(ITEMS):
   meta=model.items[index]
   if meta.name!=name:raise ValueError('Item slot order differs')
   value=dict(inventory=0,backpack=0,stash=0,courier=0,castable=False,slot=-1)
   for a in own.items:
    if a.ability_id!=ITEM_IDS[name]:continue
    key='inventory' if a.slot<=5 or a.slot>=15 else 'backpack' if a.slot<=8 else 'stash'
    value[key]+=max(1,a.charges)
    if a.slot<=5 or a.slot==15:value.update(slot=a.slot,cooldown=a.cooldown_remaining,mana_cost=meta.mana_cost,castable=index>=3 and a.is_fully_castable)
   for u in state.units:
    if u.team_id==team and u.unit_type==11:value['courier']+=sum(max(1,a.charges) for a in u.items if a.ability_id==ITEM_IDS[name])
   if name=='item_tango' and not trees:value['castable']=False
   items.append(value)
   e={}
   if enemy_unit is not None and name!='item_tpscroll':
    e={'inventory':0,'backpack':0}
    for a in enemy_unit.items:
     if a.ability_id==ITEM_IDS[name] and a.slot<=8:e['inventory' if a.slot<=5 else 'backpack']+=max(1,a.charges)
   enemy_items.append(e)
  terrain=np.zeros((4,8,8),np.float32);terrain[:3]=np.asarray(model.terrain).reshape(3,8,8)
  for u in creeps:
   x=math.floor((u['x']-hero['x'])/256+4);y=math.floor((u['y']-hero['y'])/256+4)
   if 0<=x<8 and 0<=y<8:terrain[3,y,x]+=.25 if u['team']==team else -.25
  actions=[True,True,True,any(u.get('attackable') for u in creeps+towers+([enemy] if enemy else [])),True,model.can_hold]+[s['castable'] for s in skills]+[any(i['castable'] for i in items)]
  if hero['fear']:actions[1:]=[False]*10
  gold=hero.get('gold',0);shop={i.name:i for i in aux.shop}
  owned=lambda index:sum(items[index][k] for k in ('inventory','backpack','stash','courier'))>0
  permanent=2 if owned(2) else 1 if owned(1) else 0
  next_name=('item_boots','item_bracer')[permanent] if permanent<2 else None
  next_cost=model.next_cost if permanent==model.next_permanent else (shop[next_name].price if next_name in shop else float('inf'))
  fountain=next((u for u in state.units if u.name=='dota_fountain' and u.team_id==team),None)
  base_position=position(fountain.location) if fountain else position(model.base)
  buys=[True,bool(model.start_bought and permanent<2 and gold>=next_cost)]
  for name in BUYS[2:]:
   v=shop.get(name);buys.append(bool(model.start_bought and v and gold>=v.price and (name=='item_tpscroll' or v.stock!=0)))
  result=dict(version=SPEC_VERSION,team=team,time=state.dota_time,self=hero,creeps=creeps,towers=towers,trees=trees,projectiles=projectiles,skills=skills,items=items,enemy_items=enemy_items,terrain=terrain.tolist(),side=1 if team==2 else -1,next_permanent=permanent,
   base={'id':-1,'type':'base','team':team,'tp':True,**base_position},legal={'actions':actions,'items':[v['castable'] for v in items],'buys':buys})
  if enemy:result['enemy']=enemy
  return result

def action_label(decision,encoded):
 s=encoded.snapshot;d=decision
 label=blank_action();label.update(action=d.action)
 if d.item:
  if d.item not in ITEMS:raise ValueError('Unknown item action')
  label['item']=ITEMS.index(d.item)
 if d.buy:
  if d.buy in ('item_boots','item_bracer'):
   expected=('item_boots','item_bracer')[s['next_permanent']] if s['next_permanent']<2 else None
   if d.buy!=expected:raise ValueError('Fixed item purchase disagrees with progression')
   label['buy']=1
  elif d.buy in BUYS:label['buy']=BUYS.index(d.buy)
  else:raise ValueError('Forbidden purchase')
 ref=None
 if d.action==Act.ATTACK or (d.action==Act.ITEM and label['item']==6):
  kind='tree' if d.action==Act.ITEM else None
  candidates=[(i,r) for i,r in enumerate(encoded.refs) if r.get('id')==d.target and ((r.get('type')=='tree')==(kind=='tree'))]
  if len(candidates)!=1:raise ValueError('Native target is missing or ambiguous')
  label['target'],ref=candidates[0]
 if d.action==Act.ITEM and label['item']==3:
  refs=[(i,r) for i,r in enumerate(encoded.refs) if r.get('tp')]
  label['target'],ref=min(refs,key=lambda ir:(ir[1]['x']-d.x)**2+(ir[1]['y']-d.y)**2)
 spatial=d.action in (Act.MOVE,Act.MOVE_DIRECT) or (d.action==Act.ITEM and label['item']==3)
 if spatial:
  origin=ref if d.action==Act.ITEM else s['self']
  for axis in ('x','y'):
   delta=(getattr(d,axis)-origin[axis])/16
   rounded=math.floor(delta+64.5)
   if not 0<=rounded<=128:raise ValueError('Native coordinate outside policy action grid')
   # TP coordinates may require quantization; never pretend a different command is exact.
   if abs(delta-round(delta))>.002:raise ValueError('Native coordinate is not on the policy grid')
   label[axis]=rounded
 if ref:label['target_offset']=[(ref['x']-s['self']['x'])/2048,(ref['y']-s['self']['y'])/2048]
 return label

def padded_label(label,encoded,padded_creeps,padded_trees):
 out=dict(label);target=label.get('target',0);n=len(encoded.arrays['creeps']);t=len(encoded.arrays['trees'])
 if target>=6+n+t:out['target']=6+padded_creeps+padded_trees
 elif target>=6+n:out['target']=6+padded_creeps+target-(6+n)
 elif target>=2+n:out['target']=2+padded_creeps+target-(2+n)
 if target_ref(encoded,out['target'],padded_creeps,padded_trees)!=encoded.refs[target]:raise ValueError('Target changed during batching')
 return out


def decode_action(action,encoded,padded_creeps,padded_trees,sequence,observed_tick,offset=4):
 from .protos.duel_pb2 import Decision
 s=encoded.snapshot;a=int(action['action']);buy=int(action['buy']);item=int(action['item'])
 if not encoded.arrays['action_mask'][a] or not encoded.arrays['buy_mask'][buy]:raise ValueError('Masked policy action')
 d=Decision(sequence=sequence,team=s['team'],observed_tick=observed_tick,execute_tick=observed_tick+offset,observed_dota_time=s['time'],action=a)
 if buy:d.buy=('item_boots','item_bracer')[s['next_permanent']] if buy==1 else BUYS[buy]
 ref=None
 if a==Act.ITEM:
  if not encoded.arrays['item_mask'][item]:raise ValueError('Masked item')
  d.item=ITEMS[item]
 if a==Act.ATTACK or (a==Act.ITEM and item in (3,6)):
  if not 0<=int(action['target'])<7+padded_creeps+padded_trees:raise ValueError('Target index out of range')
  ref=target_ref(encoded,int(action['target']),padded_creeps,padded_trees)
  if ref is None:raise ValueError('Padded target chosen')
  valid=ref.get('attackable') if a==Act.ATTACK else ref.get('tp') if item==3 else ref.get('type')=='tree'
  if not valid:raise ValueError('Illegal policy target')
  d.target=max(0,ref['id'])
 if a in (Act.MOVE,Act.MOVE_DIRECT) or (a==Act.ITEM and item==3):
  origin=ref if a==Act.ITEM else s['self']
  for axis in ('x','y'):
   grid=int(action[axis])
   if not 0<=grid<=128:raise ValueError('Coordinate bin out of range')
   setattr(d,axis,origin[axis]+(grid-64)*16)
 return d

def examples(run):
 run=Path(run);result=json.loads((run/'duel-result.json').read_text());path=run/'native-decisions.pb.gz'
 if result.get('error'):raise ValueError('Failed native capture: '+result['error'])
 if result.get('clock')!='game_time':raise ValueError('Legacy native clock; recollect')
 supplements={};executions={}
 for row in records(path):
  if row.HasField('model_supplement'):
   m=row.model_supplement;key=(m.team,m.tick)
   if key in supplements:raise ValueError('Duplicate model supplement')
   supplements[key]=m
  if row.HasField('execution'):
   e=row.execution;key=(e.team,e.sequence)
   if key in executions:raise ValueError('Duplicate execution acknowledgement')
   executions[key]=e
 adapters={2:NativeAdapter(),3:NativeAdapter()};previous={};last={}
 for row in records(path):
  if not row.HasField('decision'):continue
  d=row.decision
  if d.observed_dota_time>=result['time']-.001:continue
  if d.team in last and d.observed_tick!=last[d.team]+result['period_ticks']:raise ValueError('Native sequence gap')
  last[d.team]=d.observed_tick
  m=supplements.pop((d.team,d.observed_tick),None)
  snapshot=adapters[d.team].snapshot(row,m);encoded=encode(snapshot,previous.get(d.team))
  label=action_label(d,encoded)
  if not encoded.arrays['action_mask'][label['action']] or not encoded.arrays['buy_mask'][label['buy']]:raise ValueError(f'Teacher label is masked at {d.team}/{d.sequence}: {label}')
  if d.action==Act.ITEM and not encoded.arrays['item_mask'][label['item']]:raise ValueError('Teacher item label is masked')
  e=executions.get((d.team,d.sequence))
  before_end=d.observed_dota_time+result['offset_ticks']/30<result['time']-.001
  valid=bool(before_end and e and e.accepted)
  reason='accepted' if valid else 'episode_ended' if not before_end else 'no_execution_ack' if e is None else e.reason
  yield dict(encoded=encoded,action=label,action_loss_mask=valid,reason=reason,team=d.team,sequence=d.sequence,observed_tick=d.observed_tick,execute_tick=d.execute_tick)
  previous[d.team]=label
