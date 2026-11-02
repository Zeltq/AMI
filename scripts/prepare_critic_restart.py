"""Recover only MIA0 observations for value-head pretraining, never PPO."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import examples,records
from sf_duel.selfplay.quality import audit_match
from sf_duel.selfplay.rollouts import pack,load_actor
from sf_duel.selfplay.rewards import potential


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
    old=ROOT/'runs/mia-20261031';initial=ROOT/'runs/bc-continue-30m/best.pt';digest=hashlib.sha256(initial.read_bytes()).hexdigest();candidates=[]
    for source in old.glob('MIA*/cycle-*/collection*/collection.json'):
        actors={}
        for f in (source.parent/'rollouts').glob('*/actor.json'):
            meta=json.loads(f.read_text());actors[(meta['episode'],meta['team'])]=meta['policy_sha256']
        for m in json.loads(source.read_text())['matches']:
            teams=[t for t in (2,3) if actors.get((m['run'],t))==digest]
            if teams and max(v['max_lateness_ticks'] for v in m['sides'].values())<=2 and not sum(v['unacknowledged'] for v in m['sides'].values()):candidates.append((m,teams))
    state=json.loads((old/'state.json').read_text());assert hashlib.sha256(Path(state['models'][0]).read_bytes()).hexdigest()==digest
    for m in json.loads((old/'MIA1/vs-MIA0/collection.json').read_text())['matches']:
        if max(v['max_lateness_ticks'] for v in m['sides'].values())<=2 and not sum(v['unacknowledged'] for v in m['sides'].values()):
            assert m['routing'][str(5-m['model_team'])]=='0';candidates.append((m,[5-m['model_team']]))
    admissions=[];entries=[];accepted_matches=[]
    for index,(match,teams) in enumerate(candidates):
        admission=audit_match(match,output=False);admissions.append({'run':match['run'],**admission})
        (out/'admission.json').write_text(json.dumps(admissions,indent=2))
        if not admission['passed']:continue
        server={}
        for row in records(Path(match['run'])/'native-decisions.pb.gz'):
            if row.HasField('reward_frame'):server[(row.reward_frame.team,row.reward_frame.tick)]=np.asarray(row.reward_frame.values,np.float32)
        folders={t:out/f'match-{index:03d}-team-{t}' for t in teams};buffers={t:[] for t in teams}
        for folder in folders.values():folder.mkdir()
        def flush(team):
            rows=buffers[team]
            if not rows:return
            arrays=pack([r['encoded'].arrays for r in rows],[r['action'] for r in rows])
            arrays.update(sequence=np.array([r['sequence'] for r in rows],np.int64),tick=np.array([r['observed_tick'] for r in rows],np.int64),time=np.array([r['encoded'].snapshot['time'] for r in rows],np.float32))
            np.savez_compressed(folders[team]/f"{rows[0]['sequence']:08d}.npz",**arrays);buffers[team]=[]
        for row in examples(match['run']):
            if row['team'] in teams:
                buffers[row['team']].append(row)
                if len(buffers[row['team']])==128:flush(row['team'])
        for team in teams:
            flush(team);e=load_actor(folders[team],recorded_memory=False)
            common=max(min(t for s,t in server if s==side) for side in (2,3));prefix=int(np.searchsorted(e['tick'],common))
            if prefix>=len(e['tick']) or (prefix and np.any(e['time'][:prefix]>=0)):raise ValueError('Unaligned reward stream extends into gameplay')
            own=np.stack([server[(team,int(t))] for t in e['tick'][prefix:]]);enemy=np.stack([server[(5-team,int(t))] for t in e['tick'][prefix:]]);phi=potential(own,enemy)
            winner=match['outcome'].get('winner');result=0 if winner is None else 1 if winner==team else -1
            e['rewards']=np.concatenate((np.zeros(prefix,np.float32),np.diff(phi,append=0).astype(np.float32)));e['rewards'][-1]+=result;e['terminals']=np.zeros(len(e['tick']),bool);e['terminals'][-1]=True;e['loss_valid'][:prefix]=False
            for key,value in e.items():np.save(folders[team]/(key+'.npy'),value,allow_pickle=False)
            entries.append({'path':folders[team].name,'match':match['run'],'team':team,'frames':len(e['tick']),'masked_startup_frames':prefix})
        accepted_matches.append(match['run']);print(json.dumps({'prepared':len(accepted_matches),'candidates':len(candidates)}),flush=True)
    if len(accepted_matches)<8:raise RuntimeError('Too few verified archived MIA0 games')
    shuffled=sorted(accepted_matches);random.Random(20261031).shuffle(shuffled);validation=set(shuffled[:max(2,len(shuffled)//4)])
    for e in entries:e['split']='validation' if e['match'] in validation else 'train'
    meta={'policy_sha256':digest,'episodes':entries,'matches':len(accepted_matches),'gamma':1,'complete_episodes':True,'critic_pretraining_only':True,'validation_matches':len(validation)}
    (out/'dataset.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
