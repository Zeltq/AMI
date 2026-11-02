"""Real-game evaluation against the frozen Go teacher, with alternating sides."""
import argparse,hashlib,json,msvcrt,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def linux(p):
 p=Path(p).resolve();return '/mnt/'+p.drive[0].lower()+'/'+p.as_posix()[3:]

def validate_result(report,full_match):
 # Raw Go reports reserve 'eligible' for subsequent dataset audits. Evaluation
 # requires a genuine terminal outcome, not eligibility as a teacher example.
 if report.get('error'):raise ValueError('Native evaluation error')
 if full_match and (report.get('reason') not in ('death','tower','last_hits') or not report.get('outcome')):raise ValueError('Incomplete evaluation game')

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--run',required=True);p.add_argument('--matches',type=int,default=10);p.add_argument('--until',type=float,default=600);p.add_argument('--worker',type=int,default=18);p.add_argument('--workers',type=int,default=1);p.add_argument('--timescale',type=float,default=2);p.add_argument('--frozen-bots');p.add_argument('--teacher-hash');p.add_argument('--offset',type=int,default=4);p.add_argument('--allow-late',action='store_true');p.add_argument('--record-first');p.add_argument('--max-attempts',type=int,default=3);a=p.parse_args()
 if not 1<=a.workers<=10 or not 0<=a.worker<=20-a.workers:p.error('Use 1..10 workers in stable slots 0..19')
 if a.offset<4 or a.max_attempts<1:p.error('Offset must be >=4; attempts must be positive')
 expected=a.teacher_hash or hashlib.sha256((ROOT/'tools/nativecollect/teacher.go').read_bytes()).hexdigest()
 if hashlib.sha256((ROOT/'tools/nativecollect/teacher.go').read_bytes()).hexdigest()!=expected:raise ValueError('Teacher policy changed')
 base=Path(a.run).resolve();base.mkdir(parents=True,exist_ok=False);service=base/'service';service.mkdir();outcomes=[]
 def play(i,slot):
  side=2 if i%2 else 3
  for attempt in range(1,a.max_attempts+1):
   run=base/f'match-{i:03d}-attempt-{attempt:02d}'
   command=[sys.executable,str(ROOT/'scripts/run_native_teacher.py'),'--run',str(run),'--worker',str(slot),'--timescale',str(a.timescale),'--until',str(a.until),'--wall-timeout','900','--offset',str(a.offset),'--policy-address','127.0.0.1:12360','--policy-team',str(side)]
   if a.allow_late:command+=['--allow-late']
   if a.record_first and i==1:command+=['--record',a.record_first]
   if a.frozen_bots:command+=['--frozen-bots',a.frozen_bots]
   with (base/f'match-{i:03d}-attempt-{attempt:02d}.log').open('wb') as output:result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
   report=json.loads((run/'duel-result.json').read_text()) if (run/'duel-result.json').exists() else {}
   if not result.returncode:
    validate_result(report,a.until>=600)
    report.update(model_team=side,index=i,run=str(run),attempts=attempt);return report
   error=report.get('error','launch failure')
   if attempt==a.max_attempts or not (error in ('EOF','unexpected EOF','context deadline exceeded') or error.startswith(('read tcp','dial tcp'))):raise RuntimeError(f'Evaluation match {i} failed: {error}; inspect {run}')
 # Protect evaluation worker content from another collector/evaluator.
 with (ROOT/'workers/pool.lock').open('a+b') as lock, (service/'output.log').open('wb') as log:
  lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  process=subprocess.Popen(['wsl','-d','Ubuntu','--','bash','/mnt/c/AMI/scripts/wsl_python.sh','-m','sf_duel.native_policy_server','--checkpoint',linux(a.checkpoint),'--output',linux(service),'--teacher-hash',expected,'--max-batch-warmup',str(a.workers)],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
  try:
   deadline=time.monotonic()+300
   while not (service/'ready.json').exists():
    if process.poll() is not None:raise RuntimeError('Policy service failed; inspect service/output.log')
    if time.monotonic()>deadline:raise TimeoutError('Policy service startup timeout')
    time.sleep(.2)
   # Each slot independently starts its next game immediately after completion.
   with ThreadPoolExecutor(max_workers=a.workers) as pool:
    pending={pool.submit(play,i+1,a.worker+i):(i+1,a.worker+i) for i in range(min(a.workers,a.matches))};next_index=len(pending)+1
    while pending:
     completed=next(as_completed(pending));_,slot=pending.pop(completed);report=completed.result();outcomes.append(report)
     temporary=base/'progress.tmp';temporary.write_text(json.dumps({'completed':len(outcomes),'target':a.matches,'outcomes':outcomes},indent=2));temporary.replace(base/'progress.json');print(json.dumps({'completed':len(outcomes),'outcome':report.get('outcome'),'model_side':report['model_team'],'stats':report['sides'].get(str(report['model_team']))}),flush=True)
     if next_index<=a.matches:pending[pool.submit(play,next_index,slot)]=(next_index,slot);next_index+=1
  finally:
   (service/'stop').touch()
   try:process.wait(timeout=30)
   except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10)
 stats=json.loads((service/'stats.json').read_text());behaviors=[stats['episodes'][r['run']+':'+str(r['model_team'])] for r in outcomes]
 summary={'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'teacher_policy_sha256':expected,'allow_late':a.allow_late,'offset_ticks':a.offset,'timescale':a.timescale,'workers':a.workers,'matches':len(outcomes),'outcomes':outcomes,'behaviors':behaviors,'reached_mid':sum(r['first_mid_time'] is not None for r in behaviors),'attacked_creeps':sum(r['creep_projectiles']>0 for r in behaviors),'cast_razes':sum(r['raze_casts']>0 for r in behaviors),'inference_errors':stats['errors'],'diagnostic':a.until<600}
 (base/'evaluation.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ('outcomes','behaviors')}),flush=True)
 if stats['errors']:raise RuntimeError('Policy inference errors recorded')
if __name__=='__main__':main()
