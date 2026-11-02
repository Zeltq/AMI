"""Frozen before/after transfer evaluation through the existing real-Dota executor."""
import argparse,hashlib,json,msvcrt,os,random,sys,time
from math import log,sqrt
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from selfplay_cycle import Config,atomic,collect,evaluation_score

def uncertainty(games,seed):
 scores=[1. if m['outcome'].get('winner')==m['model_team'] else .5 if m['outcome'].get('winner') is None else 0. for m in sorted(games,key=lambda m:m['index'])]
 pairs=[scores[i:i+2] for i in range(0,len(scores),2)];rng=random.Random(seed);samples=[]
 for _ in range(10000):
  sample=[x for _ in pairs for x in rng.choice(pairs)];samples.append(sum(sample)/len(sample))
 samples.sort();mean=sum(scores)/len(scores);radius=sqrt(log(40)/(2*len(pairs)))
 return {'score_95_percent_interval':[max(0.,mean-radius),min(1.,mean+radius)],
  'method':'95% Hoeffding bound for independent alternating-side pair scores; engine seed uncontrolled',
  'pair_count':len(pairs),'bootstrap_diagnostic_interval':[samples[249],samples[9749]],'bootstrap_samples':10000,'seed':seed}

def map_sides(games):
 n=len(games);wins={side:sum(m['outcome'].get('winner')==side for m in games)for side in (2,3)}
 def interval(k):
  if not n:return None
  z=1.959963984540054;p=k/n;denominator=1+z*z/n;center=(p+z*z/(2*n))/denominator;radius=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
  return [max(0.,center-radius),min(1.,center+radius)]
 return {'matches':n,'radiant_wins':wins[2],'dire_wins':wins[3],'draws':n-wins[2]-wins[3],
  'radiant_win_fraction_95_percent_interval':interval(wins[2]),'interval_method':'Wilson; independent matches; draws are not wins',
  'mean_game_seconds':sum(m['time']for m in games)/max(1,n),
  'sides':{str(s):{key:sum(m['sides'][str(s)][key]for m in games)/max(1,n)for key in ('last_hits','denies','level')}for s in (2,3)}}

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--before',required=True);p.add_argument('--run',required=True);p.add_argument('--inference-host',choices=('wsl','windows'),default='wsl')
 p.add_argument('--matches',type=int,default=20);p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=20261030);p.add_argument('--resume',action='store_true');p.add_argument('--inference-device',choices=['cpu','cuda'],default='cpu');p.add_argument('--selection',choices=('argmax','sampling'),default='sampling');p.add_argument('--reset-origin',type=float);p.add_argument('--record-prefix');p.add_argument('--deadline-utc');a=p.parse_args()
 if a.matches<2 or a.matches%2:p.error('Use a positive even match count for balanced sides')
 if a.inference_host=='windows'and a.inference_device!='cpu':p.error('Windows host requires --inference-device cpu')
 deadline=datetime.fromisoformat(a.deadline_utc).timestamp()if a.deadline_utc else None
 if deadline is not None and time.time()>=deadline-1200:raise RuntimeError('Insufficient remaining time for native evaluation warmup and a bounded game')
 run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=True);checkpoint=Path(a.checkpoint).resolve();before=Path(a.before).resolve()
 cfg=replace(Config(),workers=a.workers,timescale=1.,seed=a.seed,evaluation_matches=a.matches,enforce_quality=True).validate()
 sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (checkpoint,before)}
 with (run/'evaluation.lock').open('a+b') as lock:
  lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  identity={'checkpoints':sources,'matches':a.matches,'workers':a.workers,'seed':a.seed,'timescale':1.,'launch_in_batches':True,'inference_device':a.inference_device,'selection':a.selection,'memory_reset_origin_seconds':a.reset_origin,'record_prefix':a.record_prefix,'deadline_utc':a.deadline_utc}
  if a.resume:
   if json.loads((run/'evaluation-config.json').read_text())!=identity:raise ValueError('Evaluation identity changed on resume')
  else:
   if (run/'evaluation-config.json').exists():raise FileExistsError('Use --resume for this evaluation')
   atomic(run/'evaluation-config.json',identity);cfg.write(run/'config.json')
  def status(phase,**extra):atomic(run/'evaluation-status.json',{'phase':phase,'pid':os.getpid(),'updated_utc':datetime.now(timezone.utc).isoformat(),**extra})
  try:
   parts=sorted(run.glob('collection-*/collection.json'));games=[m for part in parts for m in json.loads(part.read_text())['matches']]
   completed=[m['index'] for m in games]
   if len(completed)!=len(set(completed)):raise ValueError('Duplicate evaluation games')
   missing=[i for i in range(1,a.matches+1) if i not in completed]
   if missing:
    folder=run/f'collection-{len(parts)+1:03d}';status('evaluating',completed=len(games),target=a.matches,collection=str(folder))
    games+=collect(checkpoint,folder,cfg,[str(before)],matches=len(missing),versus_previous=True,match_indices=missing,launch_in_batches=True,inference_device=a.inference_device,selection=a.selection,reset_origin=a.reset_origin,record_prefix=a.record_prefix,deadline=deadline-900 if deadline is not None else None,inference_host=a.inference_host)
    parts.append(folder/'collection.json')
   if len(games)!=a.matches:raise RuntimeError(f'Only {len(games)}/{a.matches} admitted matches; inspect transport admission and disk capacity before resuming')
   if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest for path,digest in sources.items()):raise RuntimeError('Evaluation weights changed')
   score=evaluation_score(games);ci=uncertainty(games,a.seed)
   stats={}
   for part in parts:stats.update(json.loads((part.parent/'service/stats.json').read_text())['episodes'])
   behaviors=[stats[m['run']+':'+str(m['model_team'])] for m in games]
   self_control=checkpoint==before
   result={**score,**ci,'sides':{str(side):evaluation_score([m for m in games if m['model_team']==side]) for side in (2,3)},'map_sides':map_sides(games),'identical_policy_control':self_control,
    'checkpoints':sources,'games':games,'behaviors':behaviors,'timescale':1.,'workers':a.workers,'policy_seed':a.seed,'inference_device':a.inference_device,'selection':a.selection,'memory_reset_origin_seconds':a.reset_origin,
    'reached_mid':sum(b['first_mid_time'] is not None for b in behaviors),'attacked_creeps':sum(b['creep_projectiles']>0 for b in behaviors),
    'cast_razes':sum(b['raze_casts']>0 for b in behaviors),'parts':[str(p) for p in parts],
    'execution_parts':[json.loads((part.parent/'service/launch.json').read_text())if(part.parent/'service/launch.json').exists()else{'inference_host':'wsl','provenance':'legacy collector command; unchanged logical policy contract'}for part in parts],
    'transfer_improvement_supported':not self_control and ci['score_95_percent_interval'][0]>.5,
    'note':'Same input rounding and bounded recurrent-memory contract before and after training. Engine random seed is not controlled; native records preserve actual actions and state.'}
   atomic(run/'evaluation.json',result);status('complete',score=score,uncertainty=ci);print(json.dumps({**score,**ci}),flush=True)
  except BaseException as e:
   partial={'complete':False,'target':a.matches,'admitted':len(games) if 'games'in locals()else 0,'error':repr(e),
    'partial_score':evaluation_score(games) if 'games'in locals()and games else None,
    'transfer_improvement_supported':False,'note':'Incomplete cohort; quality-based exclusion may bias the partial score. No transfer conclusion.'}
   atomic(run/'partial-evaluation.json',partial);status('failed',**partial);raise
if __name__=='__main__':main()
