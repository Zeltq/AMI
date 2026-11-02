"""Independent native workers, a shared job queue and whole-match validation."""
import argparse
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
import msvcrt
from pathlib import Path
from queue import Empty
import random
import shutil
import subprocess
import sys
import time
from prepare_native_teacher import prepare
from audit_native_teacher import audit
from check_native_inputs import check
ROOT=Path(__file__).resolve().parents[1]

def atomic_json(path,value):
 temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)

def worker(slot,base,args,jobs,stop):
 base=Path(base);completed=[]
 while not stop.is_set():
  try:index=jobs.get_nowait()
  except Empty:break
  attempt=1
  run=base/f'match-{index:04d}'
  while run.exists():
   attempt+=1;run=base/f'match-{index:04d}-attempt-{attempt:02d}'
  command=[sys.executable,str(ROOT/'scripts/run_native_teacher.py'),'--run',str(run),'--worker',str(slot),'--timescale',str(args.timescale),'--until',str(args.until),'--wall-timeout',str(args.wall_timeout),'--frozen-bots',str(base/'teacher_bots'),'--frozen-collector',args.collector_path]
  try:
   for retry in range(args.max_attempts):
    with (base/f'launch-{index:04d}-attempt-{attempt:02d}.log').open('wb') as output:
     result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    if not result.returncode:break
    failure=json.loads((run/'duel-result.json').read_text()) if (run/'duel-result.json').exists() else {}
    error=failure.get('error','launch failure')
    retryable=error in ('EOF','unexpected EOF','launch failure','context deadline exceeded') or error.startswith(('read tcp','dial tcp'))
    if not retryable or retry+1==args.max_attempts:raise RuntimeError(f'Native match {index} failed: {error}; inspect {run}')
    attempt+=1;run=base/f'match-{index:04d}-attempt-{attempt:02d}';command[command.index('--run')+1]=str(run)
   else:raise RuntimeError('No native attempt completed')
   native=audit(run);inputs=check(run)
   report=json.loads((run/'duel-result.json').read_text())
   report.update(index=index,run=str(run),worker=slot,eligible=bool(native['passed'] and inputs['passed'] and report['reason'] in ('death','tower','last_hits')),
    model_observation_adapter_validated=True,teacher_policy_sha256=args.teacher_hash,baseline='user-approved bootstrap teacher; not optimized further')
   atomic_json(run/'result.json',report)
   with (run/'dota.log').open('rb') as source,gzip.open(run/'dota.log.gz','wb',compresslevel=1) as target:shutil.copyfileobj(source,target,1024*1024)
   (run/'dota.log').unlink()
   completed.append(report)
   atomic_json(base/f'worker-{slot:02d}.json',{'slot':slot,'completed':len(completed),'indices':[r['index'] for r in completed]})
  except BaseException:
   stop.set();raise
 return completed

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--transport-update',action='store_true');p.add_argument('--max-attempts',type=int,default=3);p.add_argument('--matches',type=int,required=True);p.add_argument('--workers',type=int,required=True)
 p.add_argument('--timescale',type=float,default=4);p.add_argument('--until',type=float,default=600);p.add_argument('--wall-timeout',type=float,default=360);p.add_argument('--validation-matches',type=int,default=64);args=p.parse_args()
 if not 1<=args.workers<=min(20,args.matches):p.error('Use 1..20 workers, no more than matches')
 if args.max_attempts<1:p.error('max-attempts must be positive')
 if not 0<=args.validation_matches<args.matches:p.error('Validation count must be smaller than match count')
 base=Path(args.run).resolve()
 if args.resume:
  if not (base/'pool.json').exists():p.error('No resumable pool manifest')
  saved=json.loads((base/'pool.json').read_text())
  for key in ('matches','validation_matches','timescale','until'):
   if getattr(args,key)!=saved[key]:p.error('Resume must preserve '+key)
 else:base.mkdir(parents=True,exist_ok=False)
 (ROOT/'workers').mkdir(exist_ok=True)
 with (ROOT/'workers/pool.lock').open('a+b') as lock:
  lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  if not args.resume:
   prepare(base/'teacher_bots');collector=base/'collector';collector.mkdir()
   for original in (ROOT/'tools/nativecollect').glob('*'):
    if original.is_file() and (original.suffix in ('.go','.proto') or original.name in ('go.mod','go.sum','nativecollect.exe')):shutil.copy2(original,collector/original.name)
   for folder in ('duelwire','worldstate'):shutil.copytree(ROOT/'tools/nativecollect'/folder,collector/folder)
   args.teacher_hash=hashlib.sha256((collector/'teacher.go').read_bytes()).hexdigest()
   contract=base/'model_contract';contract.mkdir()
   for name in ('native_data.py','observations.py','spec.py'):shutil.copy2(ROOT/'src/sf_duel'/name,contract/name)
   atomic_json(base/'source-manifest.json',{'teacher_policy_sha256':args.teacher_hash,'collector_sha256':hashlib.sha256((collector/'nativecollect.exe').read_bytes()).hexdigest(),'contract':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in contract.iterdir()}})
   atomic_json(base/'pool.json',{'matches':args.matches,'validation_matches':args.validation_matches,'timescale':args.timescale,'until':args.until,'started_unix':time.time()})
  manifest=json.loads((base/'source-manifest.json').read_text());args.teacher_hash=manifest['teacher_policy_sha256']
  if hashlib.sha256((base/'collector/nativecollect.exe').read_bytes()).hexdigest()!=manifest['collector_sha256']:raise RuntimeError('Frozen collector changed')
  if hashlib.sha256((base/'collector/teacher.go').read_bytes()).hexdigest()!=args.teacher_hash:raise RuntimeError('Frozen teacher changed')
  if args.transport_update:
   if hashlib.sha256((ROOT/'tools/nativecollect/teacher.go').read_bytes()).hexdigest()!=args.teacher_hash:raise RuntimeError('Transport update cannot change teacher policy')
   revisions=manifest.setdefault('transport_revisions',[]);revision=base/f'collector-r{len(revisions)+2}';shutil.copytree(ROOT/'tools/nativecollect',revision)
   revisions.append({'path':str(revision),'sha256':hashlib.sha256((revision/'nativecollect.exe').read_bytes()).hexdigest()});atomic_json(base/'source-manifest.json',manifest)
  active=manifest.get('transport_revisions',[])
  args.collector_path=active[-1]['path'] if active else str(base/'collector')
  expected=active[-1]['sha256'] if active else manifest['collector_sha256']
  if hashlib.sha256((Path(args.collector_path)/'nativecollect.exe').read_bytes()).hexdigest()!=expected:raise RuntimeError('Transport revision changed')
  results=[]
  for path in base.glob('match-*/result.json'):
   result=json.loads(path.read_text())
   if result.get('eligible') and result.get('teacher_policy_sha256')==args.teacher_hash:
    validation=json.loads((path.parent/'native-audit.json').read_text());model_validation=json.loads((path.parent/'model-input-audit.json').read_text())
    with (path.parent/'native-decisions.pb.gz').open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if not validation['passed'] or not model_validation['passed'] or digest!=validation['sha256']:raise RuntimeError('Completed capture changed or lost validation')
    if any(r['index']==result['index'] for r in results):raise RuntimeError('Duplicate completed match index')
    results.append(result)
  complete={r['index'] for r in results};previous_count=len(complete);started=time.monotonic()
  if args.resume and len(complete)==args.matches and (base/'dataset.json').exists():
   print(json.dumps({'already_complete':True,'matches':len(complete),'new_matches_launched':0}),flush=True);return
  for path in base.glob('worker-*.json'):atomic_json(path,{'completed':0,'indices':[]})
  with multiprocessing.Manager() as manager:
   jobs=manager.Queue();stop=manager.Event()
   for index in range(1,args.matches+1):
    if index not in complete:jobs.put(index)
   with ProcessPoolExecutor(max_workers=args.workers) as pool:
    futures=[pool.submit(worker,slot,str(base),args,jobs,stop) for slot in range(args.workers)]
    while not all(f.done() for f in futures):
     count=previous_count+sum(json.loads(path.read_text())['completed'] for path in base.glob('worker-*.json'))
     print(json.dumps({'completed':count,'target':args.matches,'workers':args.workers,'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
     time.sleep(5)
    for f in futures:results.extend(f.result())
  results.sort(key=lambda r:r['index']);eligible=[r['index'] for r in results if r['eligible']]
  if len(eligible)!=args.matches and args.until==600:raise RuntimeError('Not all requested matches passed validation')
  validation=set(random.Random(20261030).sample(eligible,min(len(eligible),args.validation_matches)))
  for r in results:r['split']='validation' if r['index'] in validation else 'train' if r['eligible'] else 'diagnostic'
  elapsed=time.time()-json.loads((base/'pool.json').read_text())['started_unix']
  dataset={'matches':results,'eligible_matches':len(eligible),'validation_matches':len(validation),'train_matches':len(eligible)-len(validation),'split_seed':20261030,'split_unit':'whole match, both perspectives','teacher_policy_sha256':args.teacher_hash,'wall_seconds':elapsed,'matches_per_hour':len(eligible)/elapsed*3600,'training_started':False}
  atomic_json(base/'dataset.json',dataset);print(json.dumps({k:v for k,v in dataset.items() if k!='matches'}),flush=True)
if __name__=='__main__':main()
