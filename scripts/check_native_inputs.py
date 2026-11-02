"""Validate every model input and teacher label; never update model weights."""
import argparse,collections,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import examples,padded_label,decode_action,action_label
from sf_duel.spec import ENEMY_PRIVATE,Act

def check(run):
 counts=collections.Counter();reasons=collections.Counter();actions=collections.Counter();last={};first={}
 for example in examples(run):
  e=example['encoded'];a=example['action'];team=example['team']
  for field in ('health','health_max','mana','mana_max','attack_damage','attack_range','attack_interval','facing','gold','level','xp_to_next','last_hits','denies'):
   if field not in e.snapshot['self']:raise ValueError('Missing own hero input: '+field)
  for key,value in e.arrays.items():
   if not np.isfinite(value).all():raise ValueError('Non-finite '+key)
  if e.snapshot.get('enemy'):
   counts['visible_enemy']+=1
   if ENEMY_PRIVATE.intersection(e.refs[1]):raise ValueError('Enemy private data leaked')
  n,t=len(e.arrays['creeps']),len(e.arrays['trees']);pc=max(16,1<<max(0,n-1).bit_length());pt=max(32,1<<max(0,t-1).bit_length())
  padded=padded_label(a,e,pc,pt)
  command=decode_action(padded,e,pc,pt,example['sequence'],example['observed_tick'])
  recovered=action_label(command,e)
  if any(recovered[k]!=a[k] for k in ('action','item','buy','target','x','y')):raise ValueError('Action roundtrip changed target or coordinates')
  counts['observations']+=1;counts['creeps']+=n;counts['projectiles']+=len(e.arrays['projectiles']);counts['trees']+=t
  reasons[example['reason']]+=1;actions[str(a['action'])]+=1
  first.setdefault(team,example['sequence']);last[team]=example['sequence']
 report={'passed':True,'counts':dict(counts),'supervision':dict(reasons),'actions':dict(actions),'first_sequence':first,'last_sequence':last,'model_observation_adapter_validated':True,'targets_survive_padding':True,'training_started':False,'adapter_sha256':hashlib.sha256((ROOT/'src/sf_duel/native_data.py').read_bytes()).hexdigest()}
 Path(run,'model-input-audit.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();print(json.dumps(check(a.run)),flush=True)
