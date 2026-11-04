"""Event-v2 on the exact before/after native fields; no fraction differencing."""
import argparse,json
from pathlib import Path
import numpy as np
from p09b_reward import numeric_table as potential_table,targets as mc_targets
GAMMA=.99
NAMES=('outcome','lh','deny','own_hp','enemy_hp','mana')
FIELDS=('own_lh','own_deny','own_hp','own_hp_max','own_mana','own_mana_max','enemy_lh','enemy_deny','enemy_hp','enemy_hp_max','enemy_mana','enemy_mana_max','terminal','result')

def rewards(state,fields):
    ix={k:fields.index(k) for k in FIELDS}  # Reject old fraction-only datasets.
    b=lambda k:state[...,0,ix[k]]
    a=lambda k:state[...,1,ix[k]]
    loss=lambda k:np.maximum(b(k)-a(k),0)/np.maximum(b(k+'_max'),1)
    out=np.stack([10*a('result')*(a('terminal')>.5),.5*(a('own_lh')-b('own_lh')),.2*(a('own_deny')-b('own_deny')),-3*loss('own_hp'),3*loss('enemy_hp'),-.5*loss('own_mana')],axis=-1)
    return out*(b('terminal')<.5)[...,None]

def targets(root,out):return mc_targets(root,out,reward_mode='event-v2')

def numeric_table(engine):
    result=potential_table(engine)
    for row in result['rows']:
        for key in ('empty','last_hit','hero','hero_and_last_hit'):row[key]/=GAMMA
    result.update(reward_mode='event-v2',components=NAMES,assumptions='Equal levels, initial stat item, no additional bracer, zero souls/stacks; nonlethal hit; no simultaneous regen, damage or level-up. One enemy lane creep LH. Resource loss divided by pre-step maximum. Gamma applies to returns only.')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['table','targets']);p.add_argument('source');p.add_argument('output');a=p.parse_args()
    if a.mode=='table':Path(a.output).write_text(json.dumps(numeric_table(a.source),indent=2),newline='\n')
    else:targets(a.source,a.output)
