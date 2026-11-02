"""Aggregate observed lane-creep stats on train and validation splits separately."""
import argparse,collections,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/teacher-pilot-192-cache');p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.cache);dataset=json.loads((root/'dataset.json').read_text());out={}
 for split in ('train','validation'):
  counts=collections.Counter();stats=collections.defaultdict(collections.Counter);spawns={2:[],3:[]}
  for match in dataset['matches']:
   if match['split']!=split:continue
   for team in (2,3):
    with np.load(root/f"match-{match['index']:04d}-team-{team}.npz") as d:
     c=d['creeps'];times=d['global_features'][:,0]*600;valid=d['creep_valid']&d['creep_ally']
     first=np.flatnonzero(times>=0)
     if len(first):
      i=first[0];spawns[team].append(c[i][valid[i]][:,:2].astype(np.float32).tolist())
     for kind,field in [('melee',45),('ranged',46),('siege',47)]:
      select=valid&(c[:,:,field]>.5)
      for bucket in range(21):
       mask=select&((times>=bucket*30)&(times<(bucket+1)*30))[:,None]
       values=c[mask].astype(np.float32)
       if len(values)==0:continue
       counts[kind,bucket]+=len(values)
       for name,index,scale in [('hp',8,2000),('damage',16,200),('interval',18,3),('armor',21,20),('range',17,1000)]:
        v,n=np.unique(np.round(values[:,index]*scale,3),return_counts=True)
        stats[kind,bucket,name].update(dict(zip(map(float,v),map(int,n))))
  out[split]={kind:[{'time':bucket*30,'samples':counts[kind,bucket],**{
    name:stats[kind,bucket,name].most_common(3) for name in ('hp','damage','interval','armor','range')}}
    for bucket in range(21) if counts[kind,bucket]] for kind in ('melee','ranged','siege')}
  out[split]['spawns']={team:[[round(x*8192,1),round(y*8192,1)] for x,y in next((v for v in values if v),[])] for team,values in spawns.items()}
 Path(a.output).write_text(json.dumps(out,indent=2));print(a.output,flush=True)
if __name__=='__main__':main()
