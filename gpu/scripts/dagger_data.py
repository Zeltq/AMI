"""DAgger manifests and sequence sampling; all paths may point outside git."""
import bisect
from collections import OrderedDict
from pathlib import Path
import random
import numpy as np
from ggd_dataset import read_block, decode


def stage_of(actor):
    return 'validation' if actor//2%8==6 else 'test' if actor//2%8==7 else 'train'


def eligible(actor, stage, source):
    return stage_of(actor)==stage and (source=='teacher-demo' or actor%2==(actor//2)%2)


def pack(sequences):
    """Union compact columns before decoding, preserving global target indices."""
    columns=sorted(set().union(*(s[0]['columns'] for s in sequences)))
    inverse={v:i for i,v in enumerate(columns)}
    header=dict(sequences[0][0],columns=columns,
                buckets=[max(s[0]['buckets'][i] for s in sequences) for i in range(3)])
    obs=np.zeros((len(sequences),128,len(columns)),dtype=np.float16)
    signals=np.stack([s[2] for s in sequences])
    for i,(h,o,_) in enumerate(sequences):
        obs[i][:,[inverse[c] for c in h['columns']]]=o
    arrays=decode(header,obs,signals)
    return arrays, arrays.pop('labels'), arrays.pop('loss_valid'), len(sequences), signals[:,:,10].copy()


class OldSequences:
    """Uniform sampling with replacement over old train sequence identities."""
    def __init__(self, root, manifest, rng):
        self.root=Path(root);self.rng=rng;self.entries=[];self.ends=[];self.cache=OrderedDict()
        total=0
        for entry in manifest['files']:
            ids=[a for a in entry['actors'] if eligible(a,'train',entry.get('source','teacher-demo'))]
            if ids:
                self.entries.append((entry,ids));total+=len(ids);self.ends.append(total)
        if not total:raise ValueError('No old train sequences')

    def draw(self):
        # Empty/padded actor blocks are rejected, so accepted draws are uniform
        # over sequences with at least one valid decision.
        for _ in range(10000):
            pick=self.rng.randrange(self.ends[-1]);i=bisect.bisect_right(self.ends,pick)
            entry,ids=self.entries[i];actor=ids[pick-(self.ends[i-1] if i else 0)]
            path=self.root/entry['file']
            if path not in self.cache:
                self.cache[path]=read_block(path,entry['sha256'])
                if len(self.cache)>8:self.cache.popitem(last=False)
            self.cache.move_to_end(path)
            h,o,s=self.cache[path];index=h['actors'].index(actor)
            if (s[index,:,11]>.5).any():return h,o[index],s[index]
        raise ValueError('No valid old train sequences after 10000 draws')


def mixed_batches(root, manifest, batch_size, seed):
    if batch_size<2 or batch_size%2:raise ValueError('DAgger needs an even batch size')
    rng=random.Random(seed);new=[e for e in manifest['files'] if e.get('source')==manifest['new_source']]
    old={'files':[e for e in manifest['files'] if e.get('source')!=manifest['new_source']]}
    sampler=OldSequences(root,old,rng);rng.shuffle(new);pending=[]
    def combine(fresh):
        replay=[sampler.draw() for _ in fresh]
        # Terminal sequences may have fewer valid decisions than replay. Keep
        # >=50% new not only by sequence count, but also by supervised frames.
        budget=sum(int((s[:,11]>.5).sum()) for _,_,s in fresh)
        locations=[(i,t) for i,(_,_,s) in enumerate(replay) for t in np.flatnonzero(s[:,11]>.5)]
        if len(locations)>budget:
            replay=[(h,o,s.copy()) for h,o,s in replay]
            for i,t in rng.sample(locations,len(locations)-budget):replay[i][2][t,11]=0
        return pack(fresh+replay)
    for entry in new:
        h,o,s=read_block(Path(root)/entry['file'],entry['sha256'])
        ids=[i for i,a in enumerate(h['actors']) if eligible(a,'train',h['source']) and (s[i,:,11]>.5).any()]
        rng.shuffle(ids)
        for i in ids:
            pending.append((h,o[i],s[i]))
            if len(pending)==batch_size//2:
                yield combine(pending);pending=[]
    if pending:yield combine(pending)
