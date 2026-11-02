"""Prepared recurrent imitation episodes; whole matches remain split together."""
import json
from pathlib import Path
import numpy as np
from .native_data import examples
from .observations import numpy_batch
HEADS=('action','buy','item','target','x','y')

def prepare_match(run,destination,index):
 destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
 sides={2:[],3:[]};labels={2:[],3:[]};valid={2:[],3:[]}
 for r in examples(run):
  t=r['team'];sides[t].append(r['encoded'].arrays);labels[t].append(r['action']);valid[t].append(r['action_loss_mask'])
 files=[]
 for team,rows in sides.items():
  b=numpy_batch(rows);pc=b['creeps'].shape[1];pt=b['trees'].shape[1]
  y=np.asarray([[a[k] for k in HEADS] for a in labels[team]],np.int64)
  for i,row in enumerate(rows):
   n=len(row['creeps']);trees=len(row['trees']);target=y[i,3]
   if target>=6+n+trees:y[i,3]=6+pc+pt
   elif target>=6+n:y[i,3]=6+pc+target-(6+n)
   elif target>=2+n:y[i,3]=2+pc+target-(2+n)
  arrays={k:v.astype(np.float16) if v.dtype==np.float32 else v for k,v in b.items()}
  arrays.update(labels=y,loss_valid=np.asarray(valid[team],bool))
  path=destination/f'match-{index:04d}-team-{team}.npz';temp=path.with_suffix('.tmp')
  with temp.open('wb') as f:np.savez_compressed(f,**arrays)
  temp.replace(path);files.append({'path':str(path),'team':team,'frames':len(rows),'match':index})
 (destination/f'match-{index:04d}.json').write_text(json.dumps({'run':str(run),'episodes':files},indent=2))
 return files

def sequence_batch(episodes,selections,length=80,burn=16):
 pc=max(episodes[e]['creeps'].shape[1] for e,start in selections)
 pt=max(episodes[e]['trees'].shape[1] for e,start in selections)
 pp=max(episodes[e]['projectiles'].shape[1] for e,start in selections)
 output={};labels=[];valid=[]
 for e,start in selections:
  episode=episodes[e];count=len(episode['labels']);indices=np.minimum(np.arange(start,start+length),count-1)
  actual=np.arange(start,start+length)<count;mask=episode['loss_valid'][indices]&actual
  if start:mask[:burn]=False
  valid.append(mask);y=episode['labels'][indices].copy();oldpc=episode['creeps'].shape[1];oldpt=episode['trees'].shape[1]
  target=y[:,3].copy();y[:,3]=np.where(target>=6+oldpc+oldpt,6+pc+pt,np.where(target>=6+oldpc,6+pc+target-(6+oldpc),np.where(target>=2+oldpc,2+pc+target-(2+oldpc),target)))
  labels.append(y)
  for key,values in episode.items():
   if key in ('labels','loss_valid'):continue
   a=values[indices]
   if key in ('attack_mask','tree_mask','tp_mask','self_mask'):
    a=np.concatenate((a[:,:2+oldpc],np.zeros((length,pc-oldpc),bool),a[:,2+oldpc:6+oldpc],a[:,6+oldpc:6+oldpc+oldpt],np.zeros((length,pt-oldpt),bool),a[:,-1:]),axis=1)
   elif key in ('creeps','creep_valid','creep_ally','creep_combat','trees','tree_valid','projectiles','projectile_valid','projectile_combat'):
    maximum=pc if key.startswith('creep') else pt if key.startswith('tree') else pp
    a=np.pad(a,((0,0),(0,maximum-a.shape[1]))+((0,0),)*(a.ndim-2))
   output.setdefault(key,[]).append(a)
 result={k:np.concatenate(v,axis=0) for k,v in output.items()}
 return result,np.concatenate(labels),np.concatenate(valid)
