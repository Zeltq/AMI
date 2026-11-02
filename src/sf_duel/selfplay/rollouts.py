"""Float32, per-actor rollout shards with immutable behavior-policy identity."""
import hashlib
import json
from pathlib import Path
from queue import Queue
import threading
import numpy as np
from ..observations import numpy_batch
from ..bc_data import HEADS, sequence_batch
from .rewards import potential, transition, gae


def pack(rows, labels):
    arrays = numpy_batch(rows)
    pc, pt = arrays['creeps'].shape[1], arrays['trees'].shape[1]
    y = np.asarray([[a[k] for k in HEADS] for a in labels], np.int64)
    for i, row in enumerate(rows):
        n, trees, target = len(row['creeps']), len(row['trees']), y[i, 3]
        if target >= 6+n+trees: y[i, 3] = 6+pc+pt
        elif target >= 6+n: y[i, 3] = 6+pc+target-(6+n)
        elif target >= 2+n: y[i, 3] = 2+pc+target-(2+n)
    return {**arrays, 'labels': y, 'loss_valid': np.ones(len(rows), bool)}


class RolloutWriter:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=False)
        self.buffers = {}; self.metadata = {}; self.error = None; self.queue = Queue(128)
        self.thread = threading.Thread(target=self._work, daemon=True); self.thread.start()

    def _work(self):
        while True:
            item = self.queue.get()
            if item is None: return
            path, entries = item
            try:
                if isinstance(entries, dict):
                    path.mkdir()
                    (path/'actor.json').write_text(json.dumps(entries))
                    continue
                arrays = pack([e.pop('observation') for e in entries], [e.pop('action') for e in entries])
                arrays.update({k: np.stack([e[k] for e in entries]) for k in entries[0]})
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as f: np.savez_compressed(f, **arrays)
                temporary.replace(path)
            except BaseException as error: self.error = error

    def append(self, episode, team, policy_hash, learner, entry):
        if self.error: raise RuntimeError('Rollout writer failed') from self.error
        key = hashlib.sha256((episode+':'+str(team)).encode()).hexdigest()[:24]
        if key not in self.buffers:
            folder = self.root/key
            meta = {'episode': episode, 'team': team, 'policy_sha256': policy_hash, 'learner': learner}
            # Metadata must precede shards, but filesystem latency must not hold
            # up the first inference reply. The writer queue preserves ordering.
            self.queue.put((folder,meta),timeout=10)
            self.metadata[key] = meta; self.buffers[key] = []
        if self.metadata[key]['policy_sha256'] != policy_hash: raise ValueError('Policy changed inside episode')
        self.buffers[key].append(entry)
        if len(self.buffers[key]) >= 128: self._flush(key)

    def _flush(self, key):
        entries = self.buffers[key]
        if entries:
            self.queue.put((self.root/key/f"{int(entries[0]['sequence']):08d}.npz", entries), timeout=10)
            self.buffers[key] = []

    def close(self):
        for key in self.buffers: self._flush(key)
        self.queue.put(None); self.thread.join()
        if self.error: raise RuntimeError('Rollout writer failed') from self.error


def load_actor(folder, recorded_memory=True):
    folder = Path(folder); chunks = []
    for path in sorted(folder.glob('*.npz')):
        with np.load(path, allow_pickle=False) as z: chunks.append({k:z[k] for k in z.files})
    if not chunks: raise ValueError('Empty actor rollout')
    # sequence_batch preserves masks and remaps targets across variable shard padding.
    pc=max(c['creeps'].shape[1] for c in chunks);pt=max(c['trees'].shape[1] for c in chunks);pp=max(c['projectiles'].shape[1] for c in chunks)
    # A one-row padding template gives all chunks the same object layout.
    template={k:v[:1].copy() for k,v in chunks[0].items()}
    for group,n in [('creeps',pc),('trees',pt),('projectiles',pp)]:
        for k in [group]+({'creeps':['creep_valid','creep_ally'],'trees':['tree_valid'],'projectiles':['projectile_valid']}[group]):
            template[k]=np.zeros((1,n)+chunks[0][k].shape[2:],chunks[0][k].dtype)
    for key in ('attack_mask','tree_mask','tp_mask','self_mask'):
        template[key]=np.zeros((1,7+pc+pt),bool)
    template['self_mask'][:,0]=True
    arrays=[]; labels=[]; extras={}
    for chunk in chunks:
        n=len(chunk['labels']); model_chunk={k:v for k,v in chunk.items() if k not in EXTRA_KEYS}
        model_template={k:v for k,v in template.items() if k not in EXTRA_KEYS}
        x,y,valid=sequence_batch([model_chunk,model_template],[(0,0),(1,0)],n,0)
        arrays.append({k:v[:n] for k,v in x.items()});labels.append(y[:n])
        for key in EXTRA_KEYS:
            if key in chunk:extras.setdefault(key,[]).append(chunk[key])
    result={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    result.update({k:np.concatenate(v) for k,v in extras.items()});result['labels']=np.concatenate(labels);result['loss_valid']=np.ones(len(result['labels']),bool)
    if not np.array_equal(result['sequence'],np.arange(1,len(result['sequence'])+1)):raise ValueError('Actor sequence gap')
    if not np.all(np.diff(result['tick'])==4):raise ValueError('Actor clock gap')
    if recorded_memory and (np.any(result['h'][0]) or np.any(result['c'][0])):raise ValueError('Initial recurrent state not zero')
    return result


EXTRA_KEYS={'old_log_prob','old_value','h','c','tick','time','sequence','reward_state','rewards','terminals','advantages','returns','scheduled_tick','execution_tick','execution_known','execution_accepted'}


def prepare_rollouts(record_root, matches, output, current_hash):
    """Only genuinely completed matches enter PPO; technical cuts never become losses."""
    root=Path(record_root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    actors={}
    for folder in root.iterdir():
        if folder.is_dir():
            meta=json.loads((folder/'actor.json').read_text());actors[(meta['episode'],meta['team'])]=(folder,meta)
    manifest=[]
    for index,match in enumerate(matches):
        if match.get('error') or match.get('reason') not in ('death','tower','last_hits') or not match.get('outcome'):
            raise ValueError('Training requires a true completed match')
        pair=[actors[(match['run'],team)] for team in (2,3)];episodes=[load_actor(folder) for folder,_ in pair]
        for i,e in enumerate(episodes):
            keep=e['time'] < match['time']-.001
            episodes[i]={k:v[keep] for k,v in e.items()}
        from ..native_data import records
        server={};executions={}
        for record in records(Path(match['run'])/'native-decisions.pb.gz'):
            if record.HasField('execution'):
                ack=record.execution;executions[(ack.team,ack.sequence)]=ack
            if record.HasField('reward_frame'):
                frame=record.reward_frame;server[(frame.team,frame.tick)]=np.asarray(frame.values,np.float32)
        winner=match['outcome'].get('winner');result=0 if winner is None else 1 if winner==2 else -1
        first={team:min((tick for side,tick in server if side==team),default=None) for team in (2,3)}
        if any(tick is None for tick in first.values()):raise ValueError('Missing authoritative team stream')
        common_start=max(first.values())
        for side,e in enumerate(episodes):
            meta=pair[side][1]
            # An archived alias can be byte-identical to the current policy
            # (notably initial MIA0 self-play). Those trajectories are on-policy.
            if not meta['learner'] and meta['policy_sha256']!=current_hash:continue
            if meta['policy_sha256']!=current_hash:raise ValueError('Off-policy learner trajectory')
            if not len(e['tick']):raise ValueError('Empty completed trajectory')
            # Streams may open one pre-game observation apart. Preserve that
            # prefix for exact recurrent replay, but never invent opponent HP or
            # a reward target for it. Both streams must be complete thereafter.
            prefix=int(np.searchsorted(e['tick'],common_start))
            if prefix>=len(e['tick']) or (prefix and np.any(e['time'][:prefix]>=0)):
                raise ValueError('Unaligned reward stream extends into gameplay')
            try:
                a=np.stack([server[(2,int(t))] for t in e['tick'][prefix:]]);b=np.stack([server[(3,int(t))] for t in e['tick'][prefix:]])
            except KeyError as error:raise ValueError('Missing same-tick authoritative reward state') from error
            np.testing.assert_allclose(e['reward_state'][prefix:], a if side==0 else b,rtol=0,atol=0)
            phi=potential(a,b)*(1 if side==0 else -1);own_result=result*(1 if side==0 else -1)
            reward=np.diff(phi,append=0.);reward[-1]+=own_result
            if not np.isclose(reward.sum(),own_result-phi[0],atol=1e-6):raise AssertionError('Shaping did not telescope')
            e['scheduled_tick']=e['tick']+match['offset_ticks']
            acknowledgements=[executions.get((side+2,int(seq))) for seq in e['sequence']]
            e['execution_known']=np.asarray([ack is not None for ack in acknowledgements],bool)
            e['execution_tick']=np.asarray([ack.tick if ack is not None else -1 for ack in acknowledgements],np.int64)
            e['execution_accepted']=np.asarray([bool(ack and ack.accepted) for ack in acknowledgements],bool)
            e['loss_valid'][:prefix]=False
            e['rewards']=np.concatenate((np.zeros(prefix,np.float32),reward.astype(np.float32)))
            e['terminals']=np.zeros(len(e['tick']),bool);e['terminals'][-1]=True
            path=out/f'match-{index:04d}-team-{side+2}';path.mkdir()
            for key,value in e.items():np.save(path/(key+'.npy'),value,allow_pickle=False)
            manifest.append({'path':path.name,'team':side+2,'match':match['run'],'frames':len(e['tick']),'masked_startup_frames':prefix})
    data={'policy_sha256':current_hash,'episodes':manifest,'matches':len(matches),'gamma':1,'complete_episodes':True}
    (out/'dataset.json').write_text(json.dumps(data,indent=2));return data
