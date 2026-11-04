"""Finalize per-decision reward metadata and audit the P09 demonstrations."""
import argparse,gzip,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
from ggd_dataset import read_block,decode

def rare_action_weights(counts):
    """P09 user correction: razes capped at 4, hero ATTACK fixed at 4."""
    total=int(sum(counts[:11]))
    return {str(k):min(4.,max(1.,round(.01*total/max(1,int(counts[k])),3))) for k in (6,7,8)},4.

def read_telemetry(path):
    with gzip.open(path,'rb') as f:
        header=json.loads(f.read(struct.unpack('<I',f.read(4))[0]))
        values=np.frombuffer(f.read(),dtype='<f4').reshape(header['shape'])
    return header,values

def finalize(root,output):
    root=Path(root);manifest=json.loads((root/'dataset.json').read_text())
    assert not manifest['partial'] and len(manifest['game_metrics'])==manifest['config']['games']
    games={g['game']:g for g in manifest['game_metrics']};counts=np.zeros(12,dtype=np.int64);checks=0;valid_frames=0;entries=[]
    terminal_seen=set();final_seen=set();state_counts={};fields=None;cache_block=None
    from sf_duel.spec import UNIT_FIELDS
    unit_fields=list(UNIT_FIELDS)
    for entry in manifest['files']:
        header,obs,sig=read_block(root/entry['file'],entry['sha256']);block=int(entry['file'].split('-')[1])
        if block!=cache_block:
            th,tv=read_telemetry(root/f'telemetry-{block:06}.f32.gz');cache_block=block
        ids=[th['actors'].index(a) for a in header['actors']];v=tv[ids].copy();valid=sig[:,:,11]>.5;fields=th['fields'];fi={k:i for i,k in enumerate(fields)}
        assert np.isfinite(v).all();assert np.all(v[valid][:,1,0]>=v[valid][:,0,0])
        outcome=np.empty(valid.shape,dtype=np.int8);reason=np.empty_like(outcome)
        arrays=decode(header,obs,sig)
        for i,actor in enumerate(header['actors']):
            g=games[actor//2];side=actor%2;row=g['sides'][side];before=v[i,:,0];after=v[i,:,1];mask=valid[i]
            result=int(g['result'])*(1 if side==0 else -1);outcome[i]=result;reason[i]={'death':1,'T1':2,'LH600':3}[g['outcome_reason']]
            if actor//2%8<6:
                labels=sig[i,mask,:6].astype(int);counts[:11]+=np.bincount(labels[:,0],minlength=11);counts[11]+=int(((labels[:,0]==3)&(labels[:,3]==1)).sum())
            # Both actor views must expose exactly the same privileged state.
            other=tv[th['actors'].index(actor^1)]
            assert np.array_equal(v[i,:,: ,1:11],other[:,:,11:21]);checks+=int(mask.sum())*20
            # Check visible native observation values against independently packed FP16 input.
            heroes=arrays['heroes'].reshape(len(ids),128,2,-1)[i]
            for j,prefix in enumerate(('own','enemy')):
                for field,raw in [('health_fraction','hp_fraction'),('mana_fraction','mana_fraction'),('level','level'),('last_hits','lh'),('denies','deny'),('souls','souls'),('raze_stacks','raze_stacks')]:
                    f=unit_fields.index(field);known=heroes[:,j,len(unit_fields)+f]>.5
                    selected=mask&known
                    assert np.allclose(heroes[selected,j,f],before[selected,fi[prefix+'_'+raw]]/UNIT_FIELDS[field],atol=.001), (actor,field)
                    checks+=int(selected.sum())
            for k in ('own_lh','own_deny','enemy_lh','enemy_deny','own_xp_total','enemy_xp_total'):
                assert np.all(after[mask,fi[k]]>=before[mask,fi[k]]-1e-3),(actor,k)
            terminal=mask&(after[:,fi['terminal']]>.5)
            if terminal.any():
                assert terminal.sum()==1 and actor not in terminal_seen;terminal_seen.add(actor)
                assert np.all(after[terminal,fi['result']]==result)
                assert np.all(after[terminal,fi['reason']]==reason[i,0]);checks+=2
            # Last post-state also covers actors dead before game-time zero.
            done=after[:,fi['terminal']]>.5
            if done.any() and actor not in final_seen:
                final_seen.add(actor);last=after[np.flatnonzero(done)[0]]
                for key,field in [('lh','own_lh'),('deny','own_deny'),('level','own_level'),('souls','own_souls'),('xp_in_level','own_xp_in_level'),('deaths','own_deaths')]:
                    assert abs(row[key]-last[fi[field]])<1e-3,(actor,key,row[key],last[fi[field]]);checks+=1
            for phase in (0,1):
                for key in ('own_raze_stacks','enemy_raze_stacks'):
                    state_counts[key]=state_counts.get(key,0)+int((v[i,mask,phase,fi[key]]>0).sum())
        target=root/entry['file'].replace('.ggd.gz','.reward.npz')
        np.savez_compressed(target,state=v,valid=valid,match_outcome=outcome,match_reason=reason,actors=np.array(header['actors']),fields=np.array(fields))
        entries.append({'file':target.name,'demonstration':entry['file'],'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'valid_frames':int(valid.sum())});valid_frames+=int(valid.sum())
    assert len(final_seen)==len(games)*2
    # Terminal decisions can be absent for actors already dead during pregame;
    # match_outcome/match_reason are still materialized on every valid decision.
    schema={'version':1,'fields':fields,'phases':['before','after'],'match_outcome':{'-1':'loss','0':'draw','1':'win'},'match_reason':{'1':'death','2':'T1','3':'LH600'},'files':entries,'policy_input':False}
    (root/'reward-metadata.json').write_text(json.dumps(schema,indent=2))
    manifest['reward_metadata']={'file':'reward-metadata.json','sha256':hashlib.sha256((root/'reward-metadata.json').read_bytes()).hexdigest()}
    (root/'dataset.json').write_text(json.dumps(manifest,indent=2))
    total=int(counts[:11].sum());weights,hero_weight=rare_action_weights(counts)
    if len(games)>=1024:assert all(counts[k]>0 for k in (6,7,8,11)), 'Requested teacher actions absent'
    report={'games':len(games),'valid_frames':valid_frames,'checked_values':checks,'final_actors':len(final_seen),'actors_with_valid_terminal_decision':len(terminal_seen),'positive_stack_states':state_counts,'train_decisions':total,'train_action_counts':counts[:11].tolist(),'train_hero_attack_decisions':int(counts[11]),'action_weights':weights,'hero_attack_weight':hero_weight,'weight_rule':'raze weight min(4, max(1, 0.01*N/count)), rounded to .001; all ATTACK including hero fixed at 4; counts from train only; denominator is weighted valid frames','metadata_manifest_sha256':hashlib.sha256((root/'reward-metadata.json').read_bytes()).hexdigest()}
    Path(output).write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('output');a=p.parse_args();finalize(a.dataset,a.output)
