"""Inspect GGDDemo2 blocks and make a small cross-runtime parity fixture."""
import argparse,gzip,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.spec import *

def layout():
    at=0;groups={};creeps=256;trees=256;shots=512;targets=2+creeps+4+trees+1
    def add(name,shape,boolean=False):
        nonlocal at
        count=int(np.prod(shape));groups[name]=(at,shape,boolean);at+=count
    for name,count,fields in [('heroes',2,UNIT_FIELDS),('creeps',creeps,UNIT_FIELDS),('towers',4,UNIT_FIELDS),('trees',trees,TREE_FIELDS),('projectiles',shots,PROJECTILE_FIELDS),('skills',4,SKILL_FIELDS),('items',8,ITEM_FIELDS),('enemy_items',8,ITEM_FIELDS)]:add(name,(count,2*len(fields)))
    add('terrain',(4,8,8));add('global_features',(2*len(GLOBAL_FIELDS),));add('previous',(len(Act)+4+len(ITEMS)+len(BUYS),))
    for name,n in [('hero_valid',2),('creep_valid',creeps),('creep_ally',creeps),('tree_valid',trees),('projectile_valid',shots),('action_mask',len(Act)),('buy_mask',len(BUYS)),('item_mask',len(ITEMS)),('attack_mask',targets),('tree_mask',targets),('tp_mask',targets),('self_mask',targets)]:add(name,(n,),True)
    for name,n,fields in [('hero_combat',2,COMBAT_UNIT_FIELDS),('creep_combat',creeps,COMBAT_UNIT_FIELDS),('tower_combat',4,COMBAT_UNIT_FIELDS),('projectile_combat',shots,COMBAT_PROJECTILE_FIELDS)]:add(name,(n,2*len(fields)))
    return groups,at

def read_block(path,checksum=None):
    path=Path(path)
    if checksum and hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:raise ValueError('Dataset checksum mismatch')
    with gzip.open(path,'rb') as f:
        if f.read(8)!=b'GGDDEMO2':raise ValueError('Wrong GGD format')
        size=struct.unpack('<I',f.read(4))[0]
        if not 0<size<4_000_000:raise ValueError('Invalid header size')
        header=json.loads(f.read(size));n=len(header['actors']);width=len(header['columns']);horizon=header['horizon'];signals=header['signal_fields']
        if header['version']!=GGD_SPEC_VERSION or horizon!=128 or signals!=13:raise ValueError('Wrong GGD contract')
        raw=f.read(n*horizon*width*2);obs=np.frombuffer(raw,dtype='<f2').reshape(n,horizon,width)
        raw=f.read(n*horizon*signals*4);sig=np.frombuffer(raw,dtype='<f4').reshape(n,horizon,signals)
        if f.read(1):raise ValueError('Unexpected block tail')
    return header,obs,sig

def decode(header,compact,signals):
    groups,width=layout();n,t,_=compact.shape;pc,pt,pp=header['buckets'];flat=compact.reshape(n*t,-1)
    inverse={column:i for i,column in enumerate(header['columns'])}
    arrays={}
    for name,(offset,shape,boolean) in groups.items():
        indices=np.arange(offset,offset+int(np.prod(shape))).reshape(shape)
        if name in ('creeps','creep_valid','creep_ally','creep_combat'):indices=indices[:pc]
        elif name in ('trees','tree_valid'):indices=indices[:pt]
        elif name in ('projectiles','projectile_valid','projectile_combat'):indices=indices[:pp]
        elif name in ('attack_mask','tree_mask','tp_mask','self_mask'):indices=np.concatenate((indices[:2+pc],indices[258:262+pt],indices[-1:]))
        a=flat[:,[inverse[int(i)] for i in indices.reshape(-1)]].reshape(n*t,*indices.shape)
        arrays[name]=a.astype(bool) if boolean else a.astype(np.float32)
    labels=signals[:,:,:6].reshape(n*t,6).astype(np.int64);targets=labels[:,3].copy()
    labels[:,3]-=(targets>=258)*(256-pc)+(targets==518)*(256-pt)
    arrays['labels']=labels;arrays['loss_valid']=signals[:,:,11].reshape(-1)>.5
    return arrays

def inspect(root,fixture=None):
    root=Path(root);manifest=json.loads((root/'dataset.json').read_text());frames=0;attacks=0;files=0;selected=None;best=-1;split_counts={}
    for entry in manifest['files']:
        header,obs,sig=read_block(root/entry['file'],entry['sha256']);valid=sig[:,:,11]>.5;actual=int(valid.sum());attack=int(((sig[:,:,0]==3)&valid).sum())
        if actual!=entry['valid_frames'] or attack!=entry['attacks']:raise ValueError('Manifest count differs from data')
        if not np.isfinite(obs).all() or not np.isfinite(sig).all():raise ValueError('Nonfinite dataset values')
        for i,actor in enumerate(header['actors']):
            split='validation' if actor//2%8==6 else 'test' if actor//2%8==7 else 'train'
            split_counts[split]=split_counts.get(split,0)+int(valid[i].sum())
            score=int(((sig[i,:,0]==3)&valid[i]).sum())
            if score>best:best=score;selected=(header,obs[i:i+1].copy(),sig[i:i+1].copy())
        frames+=actual;attacks+=attack;files+=1
    if fixture:
        header,obs,sig=selected;arrays=decode(header,obs,sig);end=int(np.flatnonzero(arrays['loss_valid'])[-1])+1
        with Path(fixture).open('wb') as f:np.savez_compressed(f,**{k:v[:end] for k,v in arrays.items()})
    return dict(files=files,valid_frames=frames,attacks=attacks,frames_by_split=split_counts,fixture=str(fixture) if fixture else None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--fixture');a=p.parse_args();print(json.dumps(inspect(a.dataset,a.fixture),indent=2))
