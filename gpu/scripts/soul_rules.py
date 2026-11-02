"""Check installed Necromastery level scaling against split native recordings."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.spec import UNIT_FIELDS

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/teacher-pilot-192-cache');p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.cache)
 dataset=json.loads((root/'dataset.json').read_text())['matches'];f={k:i for i,k in enumerate(UNIT_FIELDS)};values={};sources={}
 for split,count in (('train',4),('validation',2)):
  group=sorted((m for m in dataset if m['split']==split),key=lambda m:m['index']);indices=np.linspace(0,len(group)-1,count,dtype=int);rows=[];files=[]
  if split=='validation':
   group=[m for m in group if any(v.get('level',0)>=6 for v in m['sides'].values())]
   if len(group)<count:raise RuntimeError('Not enough held-out level-six games to validate soul scaling')
   indices=np.linspace(0,len(group)-1,count,dtype=int)
  for index in indices:
   for team in (2,3):
    file=root/f'match-{group[index]["index"]:04d}-team-{team}.npz';files.append(str(file))
    with np.load(file)as data:
     hero=data['heroes'][:,0].astype(np.float32);items=data['items'].astype(np.float32)
     level=np.rint(hero[:,f['level']]*15);souls=np.rint(hero[:,f['souls']]*30);damage=np.rint(hero[:,f['attack_damage']]*200)
     base=44.+3.6*(level-1)+np.rint(items[:,0,0]*10)*5+np.rint(items[:,2,0]*10)*2
     keep=(souls>=5)&(hero[:,48+f['souls']]>0);rows.append(np.stack([level[keep],souls[keep],damage[keep],base[keep]],axis=1))
  values[split]=np.concatenate(rows);sources[split]=files
 def metric(v,offset):
  level,souls,damage,base=v.T;predicted=np.floor(base+souls*(1.+.8*np.floor((level-offset)/6))+1e-4);delta=np.abs(damage-predicted)
  return dict(samples=len(v),mean_absolute_damage_error=float(delta.mean()),maximum=float(delta.max()),within_one_damage=float(np.mean(delta<=1)))
 train={str(offset):metric(values['train'],offset)for offset in (0,1)};offset=min((0,1),key=lambda k:train[str(k)]['mean_absolute_damage_error'])
 report={'fitted_on':'train','sources':sources,'candidates':train,'selected_level_offset':offset,'validation':metric(values['validation'],offset),
  'by_level':{split:{str(int(level)):dict(samples=int(np.sum(v[:,0]==level)),median_observed_bonus_per_soul=float(np.median(((v[:,2]-v[:,3])/v[:,1])[v[:,0]==level])))for level in np.unique(v[:,0])}for split,v in values.items()},
  'rule':'1 + 0.8 * floor((level - selected_level_offset) / 6) physical attack damage per soul; no legal damage items beyond WB/bracer'}
 Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k]for k in ('selected_level_offset','candidates','validation','by_level')},indent=2))

if __name__=='__main__':main()
