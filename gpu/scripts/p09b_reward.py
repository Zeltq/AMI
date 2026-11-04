"""Exact P09b reward arithmetic, numerical table and offline return targets."""
import argparse,json,math
from pathlib import Path
import numpy as np
GAMMA=.99
MANA_WEIGHT=.5
NAMES=('outcome','lh','deny','hp','mana')

def potentials(state,fields,mana_weight=MANA_WEIGHT):
    ix={k:i for i,k in enumerate(fields)}
    def diff(k):return state[...,ix['own_'+k]]-state[...,ix['enemy_'+k]]
    phi=np.stack([.5*diff('lh'),.2*diff('deny'),3.*diff('hp_fraction'),mana_weight*diff('mana_fraction')],axis=-1)
    return np.where((state[...,ix['terminal']]>.5)[...,None],0.,phi)

def rewards(state,fields):
    ix={k:i for i,k in enumerate(fields)}
    phi=potentials(state,fields)
    shaped=GAMMA*phi[...,1,:]-phi[...,0,:]
    terminal=(state[...,1,ix['terminal']]>.5)&(state[...,0,ix['terminal']]<.5)
    outcome=10.*state[...,1,ix['result']]*terminal
    return np.concatenate([outcome[...,None],shaped],axis=-1)

def numeric_table(engine):
    e=json.loads(Path(engine).read_text());h=e['hero'];p=e['prototypes'][0];items=e['items'];rows=[]
    for level in (1,3,5,7):
        # The simulator always includes the initial stat item, even before delivery.
        strength=h[3]*(level-1)+items[0][9];intellect=h[5]*(level-1)+items[0][11]
        hp=p[8]+(math.floor(h[0]+strength+1e-5)-h[0])*22
        mana=p[11]+(math.floor(h[2]+intellect+1e-5)-h[2])*12
        resistance=p[22]+intellect*.001
        for spell,name in enumerate(('near','mid','far')):
            skill=(level+1)//2;data=e['spells'][spell][skill-1];cost=data[1];damage=data[0]*(1-resistance)
            empty=-GAMMA*MANA_WEIGHT*cost/mana
            hero=empty+GAMMA*3*damage/hp
            rows.append(dict(level=level,spell=name,spell_level=skill,mana_cost=cost,max_mana=mana,mana_fraction=cost/mana,enemy_max_hp=hp,magic_resistance=resistance,hero_damage=damage,hp_fraction=damage/hp,empty=empty,last_hit=empty+GAMMA*.5,hero=hero,hero_and_last_hit=hero+GAMMA*.5))
    assert min(min(r['last_hit'],r['hero'],r['hero_and_last_hit']) for r in rows)>=.1
    return dict(gamma=GAMMA,mana_weight=MANA_WEIGHT,assumptions='Equal hero levels; initial stat item and no additional bracer, zero souls and zero raze stacks, first nonlethal hit, no simultaneous regen/other damage/level-up, old potential=0. Actual step adds (gamma-1)*old potential. Souls/stacks increase damage. Last hit means one enemy lane creep.',rows=rows)

def targets(root,out):
    """Reverse chronological MC returns across shard/block boundaries, split by actor."""
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'reward-metadata.json').read_text());carry={};seen=set();counts={'train':0,'validation':0,'test':0};entries=[]
    for entry in reversed(manifest['files']):
        with np.load(root/entry['file']) as z:
            state=z['state'].astype(np.float64);valid=z['valid'];actors=z['actors'];fields=z['fields'].tolist();outcome=z['match_outcome']
        phi=potentials(state,fields);r=rewards(state,fields).sum(-1);fi={k:i for i,k in enumerate(fields)}
        ret=np.zeros(valid.shape,np.float32);last_count=0
        for i,actor in enumerate(actors):
            actor=int(actor);indices=np.flatnonzero(valid[i])
            if not len(indices):continue
            if actor not in seen:
                # Pregame death may have no valid frame at the later duel terminal.
                # Collapse the no-decision tail onto its last valid transition.
                t=indices[-1]
                r[i,t]=10.*int(outcome[i,t])-float(phi[i,t,0].sum())
                carry[actor]=0.;seen.add(actor);last_count+=1
            for t in reversed(indices):
                carry[actor]=float(r[i,t])+GAMMA*carry[actor];ret[i,t]=carry[actor]
            stage='validation' if actor//2%8==6 else 'test' if actor//2%8==7 else 'train'
            counts[stage]+=len(indices)
        target=out/entry['file'];np.savez_compressed(target,returns=ret,valid=valid,post0=state[:,:,0,fi['game_seconds']]>=0,actors=actors)
        entries.append(dict(file=target.name,demonstration=entry['demonstration'],last_actor_transitions=last_count))
    report=dict(dataset=str(root),gamma=GAMMA,mana_weight=MANA_WEIGHT,actors=len(seen),frames=counts,files=list(reversed(entries)),terminal_convention='last valid decision carries terminal outcome and Phi(next)=0; no invalid padding discount; MC over complete valid actor sequence')
    (out/'targets.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['table','targets']);p.add_argument('source');p.add_argument('output');a=p.parse_args()
    if a.mode=='table':Path(a.output).write_text(json.dumps(numeric_table(a.source),indent=2),newline='\n')
    else:targets(a.source,a.output)
