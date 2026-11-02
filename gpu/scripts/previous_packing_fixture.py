"""Native-contract fixtures for the actual GPU previous-action kernel.

No policy, simulator or native server is run. Expected values come from the
production native action_label/encode path, not a copy of the GPU kernel.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from sf_duel.native_data import action_label, padded_label
from sf_duel.observations import encode
from sf_duel.spec import Act, ITEMS, SPELLS, TOWERS
from policy_history_audit import canonical_native, LAYOUT, ACTION_COLUMNS


def snapshot(side, index):
    team = side + 2
    own = dict(id=10, type='hero', team=team, visible=True, health=582,
               x=(-6700.375 if side == 0 else 6900.625) + index * .125,
               y=(-6699.3125 if side == 0 else 6650.8125) - index * .25, z=384.)
    enemy = dict(own, id=11, team=5-team, attackable=True, x=own['x']+315.3125, y=own['y']-71.625)
    creeps = [dict(own, id=21+i, type='creep', team=5-team, attackable=True,
                   x=own['x']+113.1875+i*33.125, y=own['y']+54.3125-i*65.25)
              for i in range(1+index % 3)]
    towers = [dict(own, id=31+i, type='tower', name=name, team=2+i//2,
                   x=-1500.25+i*1333.125, y=-1400.625+i*1199.25,
                   tp=(i//2 == side), attackable=(i//2 != side)) for i, name in enumerate(TOWERS)]
    trees = [dict(id=500+i, type='tree', x=own['x']+65.3125+i*19.125,
                  y=own['y']-89.1875+i*21.375, z=384.) for i in range(1+index % 2)]
    base = dict(id=-1, type='base', team=team, tp=True,
                x=(-7456. if side == 0 else 7488.), y=(-6938. if side == 0 else 6912.), z=384.)
    return dict(version=1, team=team, self=own, enemy=enemy, creeps=creeps,
                towers=towers, trees=trees, base=base, projectiles=[],
                skills=[{} for _ in SPELLS], items=[{} for _ in ITEMS],
                enemy_items=[{} for _ in ITEMS], terrain=np.zeros((4,8,8)),
                legal=dict(actions=[True]*11, items=[True]*8, buys=[True]*7))


def generate():
    cases = [('continue', Act.CONTINUE, None), ('move', Act.MOVE, None),
             ('move_direct', Act.MOVE_DIRECT, None), ('attack_hero', Act.ATTACK, 'enemy'),
             ('attack_creep', Act.ATTACK, 'creeps'), ('attack_tower', Act.ATTACK, 'towers'),
             ('stop', Act.STOP, None), ('raze_near', Act.RAZE_NEAR, None),
             ('raze_mid', Act.RAZE_MID, None), ('raze_far', Act.RAZE_FAR, None),
             ('requiem', Act.REQUIEM, None), ('tango', Act.ITEM, 'trees'),
             ('tp_base', Act.ITEM, 'base'), ('tp_tower', Act.ITEM, 'towers'),
             ('tp_base_offset', Act.ITEM, 'base'), ('mango', Act.ITEM, None)]
    packed, actions, expected, names = [], [], [], []
    for index, (name, action, target_kind) in enumerate(cases):
        for side in (0, 1):
            s = snapshot(side, index)
            e = encode(s)
            target = s.get(target_kind) if target_kind else None
            if isinstance(target, list):
                target = target[side*2 if target_kind == 'towers' and name.startswith('tp') else -1]
            item = 'item_tango' if name == 'tango' else 'item_tpscroll' if name.startswith('tp') else 'item_enchanted_mango' if name == 'mango' else ''
            origin = target if name.startswith('tp') else s['self']
            d = SimpleNamespace(action=int(action), item=item, buy='', target=target['id'] if target else 0,
                                x=origin['x']+(32 if name.endswith('offset') else 0),
                                y=origin['y']+(-48 if name.endswith('offset') else 0))
            if action in (Act.MOVE, Act.MOVE_DIRECT):
                d.x += 256; d.y -= 512
            label = action_label(d, e)
            padded = padded_label(label, e, 256, 256)
            canonical = canonical_native(e.arrays)
            packed.append(np.concatenate([canonical[key].ravel() for key in LAYOUT]).astype('<f4'))
            expected.append(encode(s, label).arrays['previous'].astype('<f2').astype('<f4'))
            actions.append([padded[key] for key in ACTION_COLUMNS])
            names.append(dict(name=name, side=side, native_label=label,
                              own_xy=[s['self']['x'],s['self']['y']],
                              native_target=label['target'], gpu_target=padded['target']))
    return dict(packed=np.array(packed), actions=np.array(actions, dtype='<f4'),
                expected=np.array(expected), bases=np.array([-7456.,-6938.,384.,7488.,6912.,384.],dtype='<f4')), names


def write(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    tensors, names = generate()
    # Production world indexing; unrelated state is deliberately zero. The
    # reference is still native action_label, which never reads this GPU array.
    state = np.zeros((len(names)//2, 256, 176), dtype='<f4')
    for actor, case in enumerate(names):
        state[actor//2, actor%2, :2] = case['own_xy']
    tensors['state'] = state.ravel()
    manifest = {'tensors': {}}
    for key, value in tensors.items():
        file = key + '.f32'; raw = value.astype('<f4').tobytes()
        (root / file).write_bytes(raw)
        manifest['tensors'][key] = dict(shape=list(value.shape), file=file, sha256=hashlib.sha256(raw).hexdigest())
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    (root / 'cases.json').write_text(json.dumps(dict(cases=names, source='production native_data.action_label + observations.encode; FP16 next observation'), indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(fixture=str(root), actors=len(names), previous_scalars=len(names)*30)))


def write_sparse(path):
    tensors, names = generate()
    cases = []
    for index, name in enumerate(names):
        indices = np.flatnonzero(tensors['packed'][index])
        cases.append(dict(name, action=tensors['actions'][index].astype(int).tolist(),
                          packed_nonzero=[[int(i),float(tensors['packed'][index,i])] for i in indices],
                          expected=tensors['expected'][index].tolist()))
    result = dict(source='previous_packing_fixture.py; native production action_label/encode',
                  schema=1, bases=tensors['bases'].tolist(), cases=cases)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, separators=(',',':'))+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('output', type=Path)
    parser.add_argument('--sparse-json', action='store_true'); args = parser.parse_args()
    if args.sparse_json: write_sparse(args.output)
    else: write(args.output)
