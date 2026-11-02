"""User-authorized, resumable MIA stages. Run requires --execute."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from selfplay_cycle import ROOT, Config, atomic, collect, evaluation_score, linux, prepare_rollouts

TEACHER_HASH='281b0d7b6626e1e359fd1dccde65c884140e7c919758a8eb47195ddb12c7d371'
REPLAYS=Path('C:/Program Files (x86)/Steam/steamapps/common/dota 2 beta/game/dota/replays')


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def opponents_for(version, teacher_retired, models):
    previous=[(f'MIA{version-1}',models[version-1])]
    if teacher_retired and version>=2:return previous+[(f'MIA{version-2}',models[version-2])]
    return [('teacher',None)]+previous


def may_continue(now, deadline, remaining):
    return now<deadline or remaining<30*60


def available_dir(parent, name):
    path=parent/name
    if not path.exists():return path
    for i in range(2,1000):
        path=parent/f'{name}-attempt-{i}'
        if not path.exists():return path
    raise RuntimeError('Too many attempts')


def export_replay(source):
    source=Path(source);REPLAYS.mkdir(parents=True,exist_ok=True);target=REPLAYS/source.name
    if target.exists():
        if hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(source.read_bytes()).digest():
            raise FileExistsError(f'Will not overwrite unrelated replay: {target}')
    else:shutil.copy2(source,target)
    return {'path':str(target),'command':f'playdemo replays/{target.name}'}


def recover_completed_matches(folder):
    """After the old service has flushed, recover independently finished games."""
    folder=Path(folder);success={}
    for path in sorted(folder.glob('match-*-attempt-*/duel-result.json')):
        report=read(path)
        if report.get('error') or report.get('reason') not in ('death','tower','last_hits') or not report.get('outcome'):continue
        pieces=path.parent.name.split('-');index=int(pieces[1]);attempt=int(pieces[3])
        routing=read(folder/f'route-{index:04d}.json')
        if routing is None:raise ValueError('Missing recovered game routing')
        report.update(run=str(path.parent.resolve()),index=index,attempt=attempt,routing=routing,teacher=False,model_team=2 if index%2 else 3)
        success[index]=report
    return [success[i] for i in sorted(success)]


def prepare_parts(parts,output,digest):
    output.mkdir(parents=True,exist_ok=False);episodes=[];matches=0
    for i,part in enumerate(parts):
        name=f'part-{i:03d}'
        meta=prepare_rollouts(Path(part['path'])/'rollouts',part['matches'],output/name,digest)
        episodes.extend({**e,'path':name+'/'+e['path']} for e in meta['episodes']);matches+=meta['matches']
    meta={'policy_sha256':digest,'episodes':episodes,'matches':matches,'gamma':1,'complete_episodes':True}
    atomic(output/'dataset.json',meta);return meta


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--run',required=True)
    p.add_argument('--deadline',default='2026-10-31T09:00:00+00:00');a=p.parse_args()
    if not a.execute:print('No work started. Explicit --execute required.');return
    deadline=datetime.fromisoformat(a.deadline).timestamp();base=Path(a.run).resolve();base.mkdir(parents=True,exist_ok=True)
    cfg=replace(Config.read(ROOT/'configs/selfplay.json'),evaluation_matches=20,teacher_win_threshold=16,teacher_streak=1)
    cfg.write(base/'config.json')
    if hashlib.sha256((ROOT/'tools/nativecollect/teacher.go').read_bytes()).hexdigest()!=TEACHER_HASH:raise RuntimeError('Teacher changed')
    with (ROOT/'runs/selfplay-cycle.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        models_dir=base/'models';models_dir.mkdir(exist_ok=True)
        if not (models_dir/'MIA0.pt').exists():shutil.copy2(ROOT/'runs/bc-continue-30m/best.pt',models_dir/'MIA0.pt')
        state=read(base/'state.json',{'completed_stages':0,'teacher_retired':False,'models':[str(models_dir/'MIA0.pt')],'stages':[]})
        def status(phase, **extra):
            value={'phase':phase,'pid':os.getpid(),'updated_utc':datetime.now(timezone.utc).isoformat(),'deadline_utc':a.deadline,'completed_stages':state['completed_stages'],'disk_free_gib':shutil.disk_usage(ROOT).free/2**30,**extra}
            atomic(base/'status.json',value);print(json.dumps(value),flush=True)
        while time.time()<deadline:
            version=state['completed_stages']+1;stage=base/f'MIA{version}';stage.mkdir(exist_ok=True)
            checkpoint=Path(state['models'][-1]);cfg_stage=replace(cfg,seed=cfg.seed+version*100)
            for cycle in (1,2):
                work=stage/f'cycle-{cycle}';work.mkdir(exist_ok=True)
                complete=read(work/'complete.json')
                if complete:checkpoint=Path(complete['checkpoint']);continue
                # Conservative initial estimate; replaced by measured prior durations when available.
                remaining=(3-cycle)*cfg.collection_minutes*60+1800+1200
                if not may_continue(time.time(),deadline,remaining):status('deadline_saved',version=version,cycle=cycle);return
                behavior=work/'behavior.pt'
                if not behavior.exists():shutil.copy2(checkpoint,behavior)
                if hashlib.sha256(behavior.read_bytes()).digest()!=hashlib.sha256(checkpoint.read_bytes()).digest():raise RuntimeError('Cycle behavior changed')
                local_cfg=replace(cfg_stage,seed=cfg_stage.seed+cycle);local_cfg.write(work/'config.json')
                collection=read(work/'collection-reference.json')
                if not collection:
                    parts=[{'path':str(p.parent),'matches':read(p)['matches'],'wall_seconds':read(p)['wall_seconds']} for p in sorted(work.glob('collection*/collection.json'))]
                    spent=sum(part['wall_seconds'] for part in parts)
                    while spent<cfg.collection_minutes*60 and time.time()<deadline:
                        if shutil.disk_usage(ROOT).free<24*2**30:raise RuntimeError('Disk headroom below 24 GiB; inspect old test artifacts before continuing')
                        atomic(work/'collection-parts.json',parts)
                        while (work/'maintenance.json').exists():
                            status('maintenance_wait',version=version,cycle=cycle);time.sleep(5)
                        folder=available_dir(work,'collection');part_cfg=replace(local_cfg,collection_minutes=(cfg.collection_minutes*60-spent)/60)
                        status('collecting',version=version,cycle=cycle,path=str(folder),nominal_matches=cfg.nominal_matches,budget_minutes=part_cfg.collection_minutes,completed_prior=sum(len(p['matches']) for p in parts),elapsed_prior_seconds=spent)
                        matches=collect(behavior,folder,part_cfg,state['models'],deadline=deadline)
                        if not matches:raise RuntimeError('No completed collection matches')
                        measured=read(folder/'collection.json')['wall_seconds'];parts.append({'path':str(folder),'matches':matches,'wall_seconds':measured});spent+=measured
                        if shutil.disk_usage(ROOT).free<24*2**30:break
                    if not parts:status('deadline_saved',version=version,cycle=cycle);return
                    collection={'path':parts[0]['path'],'parts':parts,'matches':[m for part in parts for m in part['matches']]};atomic(work/'collection-reference.json',collection)
                if time.time()>=deadline and not may_continue(time.time(),deadline,((2-cycle)*cfg.collection_minutes*60)+2400):
                    status('deadline_saved',version=version,cycle=cycle,collected_matches=len(collection['matches']));return
                dataset=read(work/'dataset-reference.json')
                if not dataset:
                    folder=available_dir(work,'dataset');status('preparing',version=version,cycle=cycle,matches=len(collection['matches']))
                    meta=prepare_parts(collection.get('parts',[collection]),folder,hashlib.sha256(behavior.read_bytes()).hexdigest())
                    dataset={'path':str(folder),'matches':meta['matches'],'frames':sum(e['frames'] for e in meta['episodes']),'masked_startup_frames':sum(e.get('masked_startup_frames',0) for e in meta['episodes'])};atomic(work/'dataset-reference.json',dataset)
                learner=available_dir(work,'learner')
                command=['wsl','-d','Ubuntu','--','bash','/mnt/c/AMI/scripts/wsl_python.sh','/mnt/c/AMI/scripts/train_selfplay.py','--config',linux(work/'config.json'),'--dataset',linux(dataset['path']),'--checkpoint',linux(behavior),'--output',linux(learner),'--train']
                if version>1 or cycle>1:command+=['--skip-value-warmup']
                with (work/f'{learner.name}.log').open('ab') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    status('learning',version=version,cycle=cycle,path=str(learner),child_pid=child.pid,command=command,frames=dataset['frames'])
                    code=child.wait()
                if code:raise RuntimeError(f'Learner exited {code}; inspect {learner}')
                checkpoint=learner/'checkpoint.pt'
                if not checkpoint.exists():raise RuntimeError('Learner did not publish checkpoint')
                atomic(work/'complete.json',{'checkpoint':str(checkpoint),'learner':str(learner),'dataset':dataset,'collection':collection['path']})
            candidate=stage/f'MIA{version}.candidate.pt'
            if not candidate.exists():shutil.copy2(checkpoint,candidate)
            validation=read(stage/'validation.json',{})
            for name,opponent in opponents_for(version,state['teacher_retired'],state['models']):
                if name in validation:continue
                if not may_continue(time.time(),deadline,1200 if validation else 2400):status('deadline_saved',version=version,candidate=str(candidate),validation=validation);return
                folder=available_dir(stage,f'vs-{name}');status('evaluating',version=version,opponent=name,path=str(folder),target=20)
                results=collect(candidate,folder,cfg_stage,[] if opponent is None else [opponent],matches=20,teacher=opponent is None,versus_previous=opponent is not None,record_prefix=f'mia{version}_vs_{name.lower()}_20261031')
                if len(results)!=20:raise RuntimeError('Incomplete validation; preserve results and inspect disk')
                stats=read(folder/'service/stats.json');behaviors=[stats['episodes'][m['run']+':'+str(m['model_team'])] for m in results]
                summary={**evaluation_score(results),'path':str(folder),'reached_mid':sum(b['first_mid_time'] is not None for b in behaviors),'attacked_creeps':sum(b['creep_projectiles']>0 for b in behaviors),'cast_razes':sum(b['raze_casts']>0 for b in behaviors),'behaviors':behaviors,'replays':[export_replay(m['replay']) for m in results if 'replay' in m]}
                validation[name]=summary;atomic(stage/'validation.json',validation)
                status('validation_series_complete',version=version,opponent=name,result=summary)
            named=models_dir/f'MIA{version}.pt'
            if not named.exists():shutil.copy2(candidate,named)
            completed={'version':version,'checkpoint':str(named),'validation':validation,'completed_utc':datetime.now(timezone.utc).isoformat()}
            atomic(stage/'complete.json',completed)
            state['completed_stages']=version;state['models'].append(str(named));state['stages'].append(completed)
            if validation.get('teacher',{}).get('wins',0)>=16:state['teacher_retired']=True
            atomic(base/'state.json',state);status('stage_complete',version=version,checkpoint=str(named),teacher_retired=state['teacher_retired'])
        status('finished_at_deadline');atomic(base/'complete.json',state)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        print(f'FAILED: {type(error).__name__}: {error}',file=sys.stderr,flush=True)
        raise
