"""Prepare/collect/check an RL cycle. Only explicit --execute enables training."""
import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import random
import subprocess
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.selfplay.config import Config
from sf_duel.selfplay.rollouts import prepare_rollouts


def linux(path):
    p=Path(path).resolve();return '/mnt/'+p.drive[0].lower()+'/'+p.as_posix()[3:]


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


@contextmanager
def policy_service(checkpoint,base,cfg,opponents,record=True,inference_device='cuda',selection='sampling',reset_origin=None,inference_host='wsl'):
    if selection not in ('sampling','argmax'):raise ValueError('Unknown action selection')
    if inference_host not in ('wsl','windows'):raise ValueError('Unknown inference host')
    if inference_host=='windows'and(inference_device!='cpu'or os.name!='nt'):raise ValueError('Windows inference requires Windows CPU')
    service=base/'service';service.mkdir()
    location=linux if inference_host=='wsl'else lambda p:str(Path(p).resolve())
    command=(['wsl','-d','Ubuntu','--','bash','/mnt/c/AMI/scripts/wsl_python.sh']if inference_host=='wsl'else[sys.executable])+['-m','sf_duel.native_policy_server',
             '--checkpoint',location(checkpoint),'--output',location(service),'--max-batch-warmup',str(cfg.workers*2),'--seed',str(cfg.seed),'--device',inference_device]
    if selection=='sampling':command+=['--stochastic']
    if reset_origin is not None:command+=['--reset-origin',str(reset_origin)]
    if record:command+=['--record-root',location(base/'rollouts')]
    for path in opponents:command+=['--opponent',location(path)]
    environment=None
    if inference_host=='windows':
        environment=os.environ.copy();environment['PYTHONPATH']=str(ROOT/'src')+os.pathsep+environment.get('PYTHONPATH','')
    with (service/'output.log').open('wb') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,cwd=ROOT,env=environment)
        atomic(service/'launch.json',{'inference_host':inference_host,'device':inference_device,'pid':process.pid,'command':command,'started_wall_time':time.time()})
        try:
            deadline=time.monotonic()+300
            while not (service/'ready.json').exists():
                if process.poll() is not None:raise RuntimeError('Policy service failed')
                if time.monotonic()>deadline:raise TimeoutError('Policy service warmup timed out')
                time.sleep(.2)
            yield
        finally:
            (service/'stop').touch()
            try:code=process.wait(timeout=60)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10);raise RuntimeError('Policy service did not stop')
        if code:raise RuntimeError('Policy service or rollout writer failed')
    stats=json.loads((service/'stats.json').read_text())
    if stats['errors']:raise RuntimeError('Inference errors; reject collection')


def play(base,index,slot,cfg,routing,until=600,teacher=False,record_name=None,wall_timeout=900,attempts=3):
    route=base/f'route-{index:04d}.json';atomic(route,routing)
    for attempt in range(1,attempts+1):
        run=base/f'match-{index:04d}-attempt-{attempt:02d}'
        registration=base/'service/registrations';registration.mkdir(exist_ok=True)
        atomic(registration/f'{index:04d}-{attempt:02d}.json',{'episode':str(run),'routing':routing})
        command=[sys.executable,str(ROOT/'scripts/run_native_teacher.py'),'--run',str(run),'--worker',str(slot),
                 '--timescale',str(cfg.timescale),'--period',str(cfg.period_ticks),'--offset',str(cfg.offset_ticks),
                 '--wall-timeout',str(wall_timeout),'--until',str(until),'--policy-address','127.0.0.1:12360',
                 '--policy-team',str(2 if index%2 else 3) if teacher else '0','--routing-file',str(route),'--allow-late']
        actual_record=f'{record_name}_a{attempt}' if record_name else None
        if actual_record:command+=['--record',actual_record]
        with (base/f'match-{index:04d}-attempt-{attempt:02d}.log').open('wb') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW).returncode
        report=json.loads((run/'duel-result.json').read_text()) if (run/'duel-result.json').exists() else {'error':'launch failure'}
        if not code and not report.get('error'):
            if until>=600 and report.get('reason') not in ('death','tower','last_hits'):raise RuntimeError('Incomplete game')
            report.update(run=str(run),index=index,routing=routing,teacher=teacher,model_team=2 if index%2 else 3,attempt=attempt,record_name=actual_record)
            if cfg.enforce_quality:
                from sf_duel.selfplay.quality import audit_match
                report['admission']=audit_match(report)
            return report
        # A failed transport attempt never contributes a loss or synthetic state.
        if attempt==attempts:raise RuntimeError(f'Game {index} failed after{attempts}attempts: {report.get("error")}')


def collect(checkpoint,base,cfg,opponents=(),matches=None,until=600,teacher=False,versus_previous=False,record_prefix=None,deadline=None,match_indices=None,launch_in_batches=False,inference_device='cuda',selection='sampling',wall_timeout=900,attempts=3,reset_origin=None,inference_host='wsl'):
    if selection not in ('sampling','argmax')or not 1<=attempts<=3 or wall_timeout<=0:raise ValueError('Invalid diagnostic selection or run bounds')
    if match_indices is not None and (not versus_previous or matches!=len(match_indices) or len(set(match_indices))!=matches or any(i<1 for i in match_indices)):
        raise ValueError('Explicit evaluation indices must be positive, unique and match the requested count')
    base.mkdir(parents=True,exist_ok=False);results=[];quarantine=[];rng=random.Random(cfg.seed)
    with (ROOT/'workers/pool.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        with policy_service(checkpoint,base,cfg,opponents,record=not teacher and not versus_previous,inference_device=inference_device,selection=selection,reset_origin=reset_origin,inference_host=inference_host):
            started=time.monotonic();next_index=1
            def routing(index):
                side=2 if index%2 else 3;route={'2':'current','3':'current'}
                if opponents and (versus_previous or rng.random()<cfg.archive_fraction):route[str(5-side)]=str(rng.randrange(len(opponents)))
                return route
            with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                pending={};free_slots=[]
                def launch(slot):
                    nonlocal next_index
                    index=next_index if match_indices is None else match_indices[next_index-1];next_index+=1
                    demo=f'{record_prefix}_{index:02d}' if record_prefix and index<=2 else None
                    bounds={}if wall_timeout==900 and attempts==3 else {'wall_timeout':wall_timeout,'attempts':attempts}
                    pending[pool.submit(play,base,index,slot,cfg,routing(index),until,teacher,demo,**bounds)]=(slot,index,demo)
                for slot in range(min(cfg.workers,matches or cfg.workers)):launch(slot)
                while pending:
                    done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                    for future in done:
                        slot,index,demo=pending.pop(future);report=future.result()
                        if demo:
                            source=ROOT/'workers'/f'{slot:02d}'/'game/dota'/f'{report["record_name"]}.dem'
                            if not source.is_file():raise FileNotFoundError(f'Replay missing: {source}')
                            saved=base/f'{demo}.dem';shutil.copy2(source,saved);report['replay']=str(saved)
                        if report.get('admission',{}).get('passed',True):results.append(report)
                        else:
                            quarantine.append(report);(base/'drain').touch()
                            atomic(base/'quality-blocked.json',{'quarantine':quarantine})
                        free=shutil.disk_usage(ROOT).free/2**30
                        if not (base/'drain').exists() and free>20 and (deadline is None or time.time()<deadline) and ((matches is not None and next_index<=matches) or (matches is None and time.monotonic()-started<cfg.collection_minutes*60)):
                            if launch_in_batches:free_slots.append(slot)
                            else:launch(slot)
                    if launch_in_batches and not pending and not (base/'drain').exists():
                        # Dedicated-server startup can stall another live game's
                        # command path. Evaluation can wait for a whole cohort;
                        # the GPU simulator's independent reset is unaffected.
                        for slot in free_slots:
                            if matches is not None and next_index>matches:break
                            if shutil.disk_usage(ROOT).free/2**30<=20:break
                            if deadline is not None and time.time()>=deadline:break
                            if matches is None and time.monotonic()-started>=cfg.collection_minutes*60:break
                            launch(slot)
                        free_slots.clear()
                    elapsed=time.monotonic()-started
                    atomic(base/'progress.json',{'completed':len(results),'quarantined':len(quarantine),'started':next_index-1,'active_workers':len(pending),'target_matches':matches,'nominal_matches':cfg.nominal_matches,'elapsed_seconds':elapsed,'budget_seconds':cfg.collection_minutes*60,'disk_free_gib':shutil.disk_usage(ROOT).free/2**30,'outcomes':results})
    result={'matches':results,'quarantine':quarantine,'wall_seconds':time.monotonic()-started,'diagnostic':until<600}
    atomic(base/'collection.json',result);return results


def evaluation_score(matches):
    wins=draws=0
    for m in matches:
        winner=m['outcome'].get('winner');wins+=winner==m['model_team'];draws+=winner is None
    return {'matches':len(matches),'wins':wins,'draws':draws,'losses':len(matches)-wins-draws,'score':(wins+.5*draws)/len(matches)}


def archive_paths(previous, explicit, current):
    return list(explicit or previous.get('archive',[]) or [str(current)])


def teacher_required(streak, cycle, cfg):
    return streak<cfg.teacher_streak or cycle%cfg.teacher_recheck_cycles==0


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/selfplay.json'));p.add_argument('--checkpoint',default=str(ROOT/'runs/bc-continue-30m/best.pt'))
    p.add_argument('--run',required=True);p.add_argument('--opponent',action='append',default=[]);p.add_argument('--execute',action='store_true')
    p.add_argument('--collect-only',action='store_true');p.add_argument('--matches',type=int);p.add_argument('--until',type=float,default=600);p.add_argument('--previous-cycle')
    a=p.parse_args();cfg=Config.read(a.config);checkpoint=Path(a.checkpoint).resolve();base=Path(a.run).resolve()
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if not a.execute and not a.collect_only:
        print(json.dumps({'ready_to_prepare':True,'training_started':False,'config':a.config,'checkpoint':str(checkpoint),'collection_minutes':cfg.collection_minutes}));return
    if a.execute and a.until<600:raise ValueError('Diagnostic cuts cannot train PPO')
    with (ROOT/'runs/selfplay-cycle.lock').open('a+b') as cycle_lock:
        cycle_lock.seek(0);cycle_lock.write(b'1');cycle_lock.flush();cycle_lock.seek(0)
        msvcrt.locking(cycle_lock.fileno(),msvcrt.LK_NBLCK,1)
        previous=json.loads(Path(a.previous_cycle).read_text()) if a.previous_cycle else {}
        if previous and hashlib.sha256(Path(previous['checkpoint']).read_bytes()).hexdigest()!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError('Previous-cycle metadata does not match the starting checkpoint')
        base.mkdir(parents=True,exist_ok=False);cfg.write(base/'config.json')
        source_checkpoint=checkpoint;shutil.copy2(checkpoint,base/'behavior.pt');checkpoint=base/'behavior.pt'
        original_opponents=archive_paths(previous,a.opponent,source_checkpoint);opponents=[]
        (base/'opponents').mkdir()
        for i,path in enumerate(original_opponents):
            dest=base/'opponents'/f'{i}.pt';shutil.copy2(path,dest);opponents.append(str(dest))
        atomic(base/'status.json',{'stage':'collecting','weights_updated':False,'behavior_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
        matches=collect(checkpoint,base/'collection',cfg,opponents,a.matches,a.until)
        if a.collect_only:
            if a.until>=600:prepare_rollouts(base/'collection/rollouts',matches,base/'dataset',hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            atomic(base/'complete.json',{'collected_matches':len(matches),'weights_updated':False});return
        atomic(base/'status.json',{'stage':'preparing','weights_updated':False})
        prepare_rollouts(base/'collection/rollouts',matches,base/'dataset',hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        command=['wsl','-d','Ubuntu','--','bash','/mnt/c/AMI/scripts/wsl_python.sh','/mnt/c/AMI/scripts/train_selfplay.py',
                 '--config',linux(base/'config.json'),'--dataset',linux(base/'dataset'),'--checkpoint',linux(checkpoint),'--output',linux(base/'learner'),'--train']
        if a.previous_cycle:command+=['--skip-value-warmup']
        atomic(base/'status.json',{'stage':'learning','command':command})
        with (base/'learner.log').open('wb') as log:
            if subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW).returncode:raise RuntimeError('PPO failed; checkpoint not promoted')
        current=base/'learner/checkpoint.pt'
        cycle=int(previous.get('cycle',0))+1;streak=int(previous.get('teacher_streak',0))
        score={}
        if teacher_required(streak,cycle,cfg):
            atomic(base/'status.json',{'stage':'evaluating_teacher'})
            score['teacher']=evaluation_score(collect(current,base/'vs-teacher',cfg,matches=cfg.evaluation_matches,teacher=True))
            streak=streak+1 if score['teacher']['wins']>=cfg.teacher_win_threshold else 0
        atomic(base/'status.json',{'stage':'evaluating_previous'})
        score['previous']=evaluation_score(collect(current,base/'vs-previous',cfg,[str(checkpoint)],cfg.evaluation_matches,versus_previous=True))
        atomic(base/'status.json',{'stage':'complete'})
        archive=list(dict.fromkeys([*previous.get('archive',[]),str(checkpoint)]))
        atomic(base/'complete.json',{'cycle':cycle,'teacher_streak':streak,'checkpoint':str(current),'archive':archive,'evaluation':score,'training_started':True})


if __name__=='__main__':main()
