"""Exact P09b reward arithmetic, numerical table and offline return targets."""
import argparse,json,math,hashlib,os,time
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
    """MC return on every native decision-clock step; validity masks only the loss."""
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    complete=out/'targets.json'
    if complete.exists():complete.unlink()
    manifest=json.loads((root/'reward-metadata.json').read_text());carry={};terminals={};actors_seen=set();counts={'train':0,'validation':0,'test':0};entries=[];started=time.monotonic()
    def progress(stage,index):
        status=dict(stage=stage,pid=os.getpid(),files=index,total_files=len(manifest['files']),frames=counts,elapsed_seconds=time.monotonic()-started)
        tmp=out/'targets-status.tmp';tmp.write_text(json.dumps(status,indent=2),newline='\n');tmp.replace(out/'targets-status.json')
    # A hero can be dead (invalid for imitation) while the pregame duel has not
    # terminated. Its actual terminal state is still present in privileged data.
    for index,entry in enumerate(manifest['files']):
        with np.load(root/entry['file']) as z:
            state=z['state'];actors=z['actors'];fields=z['fields'].tolist()
        terminal_column=fields.index('terminal')
        for i,actor in enumerate(actors):
            actor=int(actor);actors_seen.add(actor)
            if actor not in terminals:
                steps=np.flatnonzero(state[i,:,1,terminal_column]>.5)
                if len(steps):terminals[actor]=(index,int(steps[0]))
        if index%16==0:progress('locating native terminals',index+1)
    if actors_seen!=set(terminals):raise ValueError('Missing native terminal state for actors '+str(sorted(actors_seen-set(terminals))))
    internal_invalid=0;terminal_invalid=0
    for index in range(len(manifest['files'])-1,-1,-1):
        entry=manifest['files'][index]
        with np.load(root/entry['file']) as z:
            state=z['state'].astype(np.float64);valid=z['valid'];actors=z['actors'];fields=z['fields'].tolist();outcome=z['match_outcome']
        r=rewards(state,fields).sum(-1);fi={k:i for i,k in enumerate(fields)}
        ret=np.zeros(valid.shape,np.float32);last_count=0
        for i,actor in enumerate(actors):
            actor=int(actor);terminal_file,terminal_step=terminals[actor]
            if index>terminal_file:
                assert not valid[i].any(),('valid decisions after terminal',actor,entry['file'])
                continue
            limit=terminal_step+1 if index==terminal_file else valid.shape[1]
            assert not valid[i,limit:].any()
            if index==terminal_file:
                carry[actor]=0.;last_count+=1;terminal_invalid+=int(not valid[i,terminal_step])
                assert int(state[i,terminal_step,1,fi['result']])==int(outcome[i,terminal_step])
            for t in range(limit-1,-1,-1):
                carry[actor]=float(r[i,t])+GAMMA*carry[actor]
                if valid[i,t]:ret[i,t]=carry[actor]
                else:internal_invalid+=1
            stage='validation' if actor//2%8==6 else 'test' if actor//2%8==7 else 'train'
            counts[stage]+=int(valid[i].sum())
        target=out/entry['file'];temporary=target.with_name(target.name+'.tmp.npz')
        np.savez_compressed(temporary,returns=ret,valid=valid,post0=state[:,:,0,fi['game_seconds']]>=0,actors=actors);temporary.replace(target)
        entries.append(dict(file=target.name,demonstration=entry['demonstration'],native_terminal_actors=last_count))
        if index%8==0:progress('return targets',len(entries))
    report=dict(version=2,dataset=str(root),gamma=GAMMA,mana_weight=MANA_WEIGHT,actors=len(terminals),frames=counts,files=list(reversed(entries)),invalid_steps_before_or_at_terminal=internal_invalid,terminals_outside_imitation_mask=terminal_invalid,terminal_convention='actual first native terminal, Phi(next)=0; gamma at every four-tick decision step, including invalid imitation frames; invalid loss frames and post-terminal padding are not training examples')
    report['metadata_sha256']=hashlib.sha256((root/'reward-metadata.json').read_bytes()).hexdigest()
    complete.write_text(json.dumps(report,indent=2),newline='\n')
    (out/'targets-status.json').write_text(json.dumps(dict(stage='complete',files=len(entries),frames=counts,pid=None),indent=2),newline='\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['table','targets']);p.add_argument('source');p.add_argument('output');a=p.parse_args()
    if a.mode=='table':Path(a.output).write_text(json.dumps(numeric_table(a.source),indent=2),newline='\n')
    else:targets(a.source,a.output)
