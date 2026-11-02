"""Infer XP thresholds from training match level transitions, then check held-out matches.

A single four-tick transition can contain several creep deaths. Candidate grants
come from installed creep bounties and half-XP denies (integer rounding included).
Validation matches never select the rule. No record is made training-eligible here.
"""
import argparse,collections,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/teacher-pilot-192-cache');p.add_argument('--output',required=True)
 a=p.parse_args();root=Path(a.cache);dataset=json.loads((root/'dataset.json').read_text())
 grants={0}
 for _ in range(4):grants|={x+y for x in grants for y in (28,29,34,35,57,69,88)}
 splits={s:{'max':collections.defaultdict(int),'transitions':collections.defaultdict(list),'matches':0} for s in ('train','validation')}
 for match in dataset['matches']:
  s=splits[match['split']];s['matches']+=1
  for team in (2,3):
   with np.load(root/f"match-{match['index']:04d}-team-{team}.npz") as data:
    own=data['heroes'][:,0];level=np.rint(own[:,23]*15).astype(int);xp=np.rint(own[:,25]*2000).astype(int)
   for l in np.unique(level):s['max'][int(l)]=max(s['max'][int(l)],int(xp[level==l].max()))
   for i in np.where(np.diff(level)==1)[0]:s['transitions'][int(level[i+1])].append(int(xp[i+1]-xp[i]))
  if s['matches']%32==0:print(f"{match['split']}: {s['matches']} matches",flush=True)
 train=splits['train'];thresholds={1:240};evidence={}
 for level,deltas in sorted(train['transitions'].items()):
  # Infer an integer multiple of 20 above every observed remaining-XP value.
  lower=((train['max'][level]+19)//20)*20
  candidates=range(lower,lower+101,20)
  counts={v:sum(v-d in grants for d in deltas) for v in candidates}
  # Prefer the smallest consistent threshold. Unweighted sums allow implausible
  # large multi-death grants to outvote ordinary single-creep transitions.
  supported=[v for v in candidates if counts[v]>=.6*len(deltas)]
  if not supported:raise RuntimeError(f'No supported XP threshold at level {level}: {counts}')
  best=min(supported);thresholds[level]=best
  evidence[level]={'observed_max_remaining':train['max'][level],'transitions':len(deltas),'candidate_support':counts}
 validation={}
 for level,deltas in splits['validation']['transitions'].items():
  if level in thresholds:
   value=thresholds[level];validation[level]={'threshold':value,'transitions':len(deltas),
     'consistent_grants':sum(value-d in grants for d in deltas),'max_remaining':splits['validation']['max'][level],
     'remaining_bound_passed':splits['validation']['max'][level]<=value}
 report={'source':str(root),'train_matches':train['matches'],'validation_matches':splits['validation']['matches'],
  'thresholds':thresholds,'training_evidence':evidence,'held_out':validation,
  'scope':'XP thresholds only. Does not validate XP radius, deny sharing or current installed build.'}
 Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(thresholds),flush=True)
if __name__=='__main__':main()
