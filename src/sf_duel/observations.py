"""Causal, visibility-filtered snapshots -> padded batches for the policy."""
from dataclasses import dataclass
import math
import numpy as np
from .spec import *

def features(record, fields):
    values, known = [], []
    record = record or {}
    for name, scale in fields.items():
        value = record.get(name)
        valid = isinstance(value, (int, float, bool)) and math.isfinite(value)
        values.append(float(value) / scale if valid else 0.)
        known.append(float(valid))
    return np.asarray(values + known, dtype=np.float32)

def prepare_unit(record, own, origin):
    if not record or not record.get('visible', own):
        return {}
    result = dict(record)
    if not own:
        for key in ENEMY_PRIVATE:
            result.pop(key, None)
    for coord in ('x', 'y'):
        if coord in result and coord in origin:
            result['d' + coord] = result[coord] - origin[coord]
    if 'facing' in result:
        angle = math.radians(result['facing'])
        result['facing_cos'], result['facing_sin'] = math.cos(angle), math.sin(angle)
    for resource in ('health', 'mana'):
        maximum = result.get(resource + '_max', 0)
        if maximum > 0 and resource in result:
            result[resource + '_fraction'] = result[resource] / maximum
    return result

@dataclass
class Encoded:
    arrays: dict
    snapshot: dict
    refs: list
    previous: dict

def visible_units(snapshot,combat_inputs=False):
    """Canonical visible unit layout, without constructing feature tensors."""
    if snapshot.get('version') != (GGD_SPEC_VERSION if combat_inputs and snapshot.get('version')==GGD_SPEC_VERSION else SPEC_VERSION):
        raise ValueError('Observation protocol version mismatch')
    own = prepare_unit(snapshot.get('self'), True, snapshot.get('self') or {})
    if not own or own.get('health', 0) <= 0:
        raise ValueError('No living controlled hero; terminal observations must not be acted on')
    enemy = prepare_unit(snapshot.get('enemy'), False, own)
    units = [prepare_unit(u, u.get('team') == snapshot['team'], own)
             for u in snapshot.get('creeps', []) if u.get('visible') and u.get('health', 0) > 0]
    for u in units:
        # Private HERO fields must not accidentally reappear through another list.
        if u.get('type') != 'creep':
            raise ValueError('Non-lane-creep in creep observations')
        u['team_relative'] = 1 if u.get('team') == snapshot['team'] else -1
    return own,enemy,units


def encode(snapshot, previous=None,combat_inputs=False,combat_memory=None,observation_profile=None):
    if observation_profile not in (None,'ggd-v2','ggd-v3'):raise ValueError('Unknown observation profile')
    if observation_profile:combat_inputs=True
    own,enemy,units=visible_units(snapshot,combat_inputs)
    towers = snapshot.get('towers', [])
    if len(towers) != 4:
        raise ValueError('Exactly the four mid T1/T2 tower slots are required')
    tower_rows = []
    for name, tower in zip(TOWERS, towers):
        if tower.get('name') != name:
            raise ValueError('Tower slot order mismatch')
        row = prepare_unit(tower, tower.get('team') == snapshot['team'], own)
        # Fixed tower positions remain public; hidden HP/state do not.
        if not row:
            row = {k: tower[k] for k in ('x','y','z') if k in tower}
        tower_rows.append(row)
    trees = snapshot.get('trees', [])
    tree_rows = []
    for tree in trees:
        row = dict(tree)
        row.update(dx=tree['x']-own['x'], dy=tree['y']-own['y'])
        row['distance'] = math.hypot(row['dx'], row['dy'])
        tree_rows.append(row)
    projectiles = snapshot.get('projectiles', [])
    projectile_rows = []
    for p in projectiles:
        row = dict(p)
        if 'x' in p: row['dx'] = p['x']-own['x']
        if 'y' in p: row['dy'] = p['y']-own['y']
        projectile_rows.append(row)
    skills = snapshot.get('skills', [])
    items = snapshot.get('items', [])
    enemy_items=snapshot.get('enemy_items',[]) if enemy else [{} for _ in ITEMS]
    if len(skills) != len(SPELLS) or len(items) != len(ITEMS) or len(enemy_items)!=len(ITEMS):
        raise ValueError('Incomplete ability/item slots')
    # dkjson represents an empty Lua table as []; both encode an unknown row.
    enemy_items=[{k:v for k,v in (row or {}).items() if k in ('inventory','backpack')} for row in enemy_items]
    terrain = np.asarray(snapshot.get('terrain'), dtype=np.float32)
    if terrain.shape != (4,8,8) or not np.isfinite(terrain).all():
        raise ValueError(f'Expected valid 4x8x8 terrain, received {terrain.shape}')
    prev = previous or {'action': 0, 'x': 64, 'y': 64, 'item': 0, 'buy': 0}
    prev_vector = np.zeros(len(Act)+2+2+len(ITEMS)+len(BUYS), np.float32)
    prev_vector[int(prev['action'])] = 1
    offset = len(Act)
    prev_vector[offset:offset+2] = [(prev.get('x',64)-64)/64, (prev.get('y',64)-64)/64]
    prev_vector[offset+2:offset+4] = prev.get('target_offset', (0.,0.))
    prev_vector[offset+4+int(prev.get('item',0))] = 1
    prev_vector[offset+4+len(ITEMS)+int(prev.get('buy',0))] = 1
    def rows(data, fields):
        return np.stack([features(r, fields) for r in data]) if data else np.zeros((0,2*len(fields)),np.float32)
    refs = [own, enemy] + units + list(towers) + list(trees) + [snapshot['base']]
    action_mask = np.asarray(snapshot['legal']['actions'], dtype=bool)
    item_mask = np.asarray(snapshot['legal']['items'], dtype=bool)
    buy_mask = np.asarray(snapshot['legal']['buys'], dtype=bool)
    if action_mask.shape != (len(Act),) or item_mask.shape != (len(ITEMS),) or buy_mask.shape != (len(BUYS),):
        raise ValueError('Action mask shape mismatch')
    if not action_mask[0] or not buy_mask[0]:
        raise ValueError('CONTINUE and no purchase must always be valid')
    attack, tree_mask, tp_mask, self_mask = [], [], [], []
    for index, ref in enumerate(refs):
        attack.append(bool(index > 0 and ref.get('attackable',False)))
        tree_mask.append(bool(ref.get('type') == 'tree'))
        tp_mask.append(bool(ref.get('tp',False)))
        self_mask.append(index == 0)
    if action_mask[Act.ATTACK] and not any(attack):
        raise ValueError('Attack enabled without any legal target')
    if item_mask[3] and not any(tp_mask):
        raise ValueError('TP enabled without destination')
    if item_mask[6] and not any(tree_mask):
        raise ValueError('Tango enabled without tree')
    arrays = dict(heroes=rows([own,enemy],UNIT_FIELDS), creeps=rows(units,UNIT_FIELDS),
                  towers=rows(tower_rows,UNIT_FIELDS), trees=rows(tree_rows,TREE_FIELDS),
                  projectiles=rows(projectile_rows,PROJECTILE_FIELDS),
                  skills=rows(skills,SKILL_FIELDS), items=rows(items,ITEM_FIELDS),
                  enemy_items=rows(enemy_items,ITEM_FIELDS),
                  global_features=features(snapshot,GLOBAL_FIELDS), previous=prev_vector,
                  terrain=terrain, hero_valid=np.asarray([True,bool(enemy)]),
                  creep_valid=np.ones(len(units),bool), tree_valid=np.ones(len(trees),bool),
                  projectile_valid=np.ones(len(projectiles),bool),
                  creep_ally=np.asarray([u['team_relative']>0 for u in units],dtype=bool),
                  action_mask=action_mask,item_mask=item_mask,buy_mask=buy_mask,
                  attack_mask=np.asarray(attack),tree_mask=np.asarray(tree_mask),
                  tp_mask=np.asarray(tp_mask),self_mask=np.asarray(self_mask))
    if combat_inputs:
        from .combat_observations import CombatMemory,encode_combat
        arrays.update(encode_combat(snapshot,own,enemy,units,tower_rows,projectiles,combat_memory or CombatMemory()))
    if observation_profile=='ggd-v3':
        from .timing_observations import timing_rows
        for group,key,records in [('heroes','hero_combat',[own,enemy]),('creeps','creep_combat',units),('towers','tower_combat',tower_rows)]:
            arrays[key]=np.concatenate((arrays[key],timing_rows(records)),-1)
            cycle=list(UNIT_FIELDS).index('attack_cycle')
            for i,r in enumerate(records):
                if not r.get('attack_animation') or r.get('creep_siege') or 'tower2' in r.get('name',''):
                    arrays[group][i,cycle]=arrays[group][i,len(UNIT_FIELDS)+cycle]=0
    return Encoded(arrays,snapshot,refs,prev)

def numpy_batch(rows):
    result = {}
    variable = {'creeps':'creep_valid', 'trees':'tree_valid','projectiles':'projectile_valid'}
    minima={'creeps':16,'trees':32,'projectiles':16}
    maxima={}
    for key in variable:
        count=max(len(o[key]) for o in rows)
        # Buckets stabilize GPU shapes; this pads, and never truncates, objects.
        maxima[key]=max(minima[key],1<<max(0,count-1).bit_length())
    # Targets must use padded group offsets, including absent enemy/self slots.
    for key in rows[0]:
        if key in ('attack_mask','tree_mask','tp_mask','self_mask'):
            padded_rows=[]
            for o in rows:
                n,t=len(o['creeps']),len(o['trees'])
                raw=o[key]
                padded_rows.append(np.concatenate((raw[:2],raw[2:2+n],np.zeros(maxima['creeps']-n,bool),
                    raw[2+n:6+n],raw[6+n:6+n+t],np.zeros(maxima['trees']-t,bool),raw[-1:])))
            value=np.stack(padded_rows)
        elif key in ('creep_combat','projectile_combat'):
            group='creeps' if key=='creep_combat' else 'projectiles'
            value=np.stack([np.pad(o[key],((0,maxima[group]-len(o[key])),(0,0))) for o in rows])
        elif key in variable or key in variable.values() or key=='creep_ally':
            group=key if key in variable else next((g for g,v in variable.items() if v==key),'creeps')
            data=[]
            for o in rows:
                a=o[key]
                shape=(maxima[group]-len(a),)+a.shape[1:]
                data.append(np.concatenate((a,np.zeros(shape,dtype=a.dtype))))
            value=np.stack(data)
        else:
            value=np.stack([o[key] for o in rows])
        result[key]=value
    return result

def packed_tensors(arrays,device):
    """Two dtype-preserving transfers instead of one synchronous copy per field."""
    import torch
    groups={};result={}
    for key,value in arrays.items():groups.setdefault(value.dtype,[]).append((key,value))
    for entries in groups.values():
        flat=np.concatenate([v.reshape(-1) for _,v in entries])
        tensor=torch.as_tensor(flat,device=device);offset=0
        for key,value in entries:
            result[key]=tensor[offset:offset+value.size].reshape(value.shape);offset+=value.size
    return result


def batch(observations,device='cpu'):
    import torch
    arrays=numpy_batch([o.arrays for o in observations])
    if torch.device(device).type=='cuda':return packed_tensors(arrays,device)
    return {k:torch.as_tensor(v,device=device) for k,v in arrays.items()}


def warmup_batch(n,device,combat_inputs=False,timing_inputs=False):
    import torch
    sizes={'heroes':(2,2*len(UNIT_FIELDS)),'creeps':(16,2*len(UNIT_FIELDS)),
        'towers':(4,2*len(UNIT_FIELDS)),'trees':(32,2*len(TREE_FIELDS)),
        'projectiles':(16,2*len(PROJECTILE_FIELDS)),'skills':(4,2*len(SKILL_FIELDS)),
        'items':(8,2*len(ITEM_FIELDS)),'enemy_items':(8,2*len(ITEM_FIELDS)),'terrain':(4,8,8),'global_features':(2*len(GLOBAL_FIELDS),),
        'previous':(len(Act)+4+len(ITEMS)+len(BUYS),)}
    result={k:torch.zeros((n,*shape),device=device) for k,shape in sizes.items()}
    if combat_inputs:
        for key,count,fields in [('hero_combat',2,COMBAT_UNIT_FIELDS),('creep_combat',16,COMBAT_UNIT_FIELDS),('tower_combat',4,COMBAT_UNIT_FIELDS),('projectile_combat',16,COMBAT_PROJECTILE_FIELDS)]:result[key]=torch.zeros((n,count,2*len(fields)),device=device)
    if timing_inputs:
        from .timing_observations import TIMING_FIELDS
        for key in ('hero_combat','creep_combat','tower_combat'):
            old=result[key];result[key]=torch.cat((old,torch.zeros((*old.shape[:-1],2*len(TIMING_FIELDS)),device=device)),-1)
    sizes={'hero_valid':2,'creep_valid':16,'creep_ally':16,'tree_valid':32,'projectile_valid':16,
        'action_mask':len(Act),'buy_mask':len(BUYS),'item_mask':len(ITEMS),
        'attack_mask':55,'tree_mask':55,'tp_mask':55,'self_mask':55}
    for k,size in sizes.items(): result[k]=torch.zeros((n,size),device=device,dtype=torch.bool)
    for k in ('hero_valid','action_mask','buy_mask','self_mask'):result[k][:,0]=True
    return result

def target_ref(encoded, target, padded_creeps, padded_trees):
    n,t=len(encoded.arrays['creeps']),len(encoded.arrays['trees'])
    if target<2: return encoded.refs[target]
    if target<2+padded_creeps:
        return encoded.refs[target] if target<2+n else None
    if target<6+padded_creeps:
        return encoded.refs[2+n+target-(2+padded_creeps)]
    if target<6+padded_creeps+padded_trees:
        j=target-(6+padded_creeps)
        return encoded.refs[6+n+j] if j<t else None
    return encoded.refs[-1]
