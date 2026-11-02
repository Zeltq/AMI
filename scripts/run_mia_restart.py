"""Resumable coordinator for the explicitly approved nine-hour MIA restart."""
import argparse
import ctypes
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from selfplay_cycle import ROOT,Config,atomic,collect,evaluation_score,linux
from run_mia_autonomous import available_dir,export_replay,prepare_parts,opponents_for,TEACHER_HASH


def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default


def collection_parts(work):
    """Recover complete accepted games after a graceful coordinator handover."""
    parts=[]
    for source in sorted(work.glob('collection*/collection.json')):
        data=read(source)
        if data.get('quarantine') and not (source.parent/'quality-repair.json').exists():
            raise RuntimeError('Collection quarantine requires diagnosis before refill')
        parts.append({'path':str(source.parent),'matches':data['matches']})
    identities=[m['run'] for part in parts for m in part['matches']]
    if len(identities)!=len(set(identities)):raise ValueError('Duplicate recovered match')
    return parts


def alive(pid):
    kernel=ctypes.windll.kernel32;kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return False
    code=ctypes.c_ulong()
    try:return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
    finally:kernel.CloseHandle(handle)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:raise SystemExit('Explicit --execute required')
    base=Path(a.run).resolve();bootstrap=read(base/'bootstrap.json');schedule=read(base/'schedule.json',{});training_only=bool(schedule.get('training_only',False))
    deadline_iso=None if schedule.get('continue_until_stopped',False) else schedule.get('deadline_utc',bootstrap['deadline_utc'])
    deadline=float('inf') if deadline_iso is None else datetime.fromisoformat(deadline_iso).timestamp();cfg=Config.read(base/'config.json')
    if hashlib.sha256((ROOT/'tools/nativecollect/teacher.go').read_bytes()).hexdigest()!=TEACHER_HASH:raise RuntimeError('Teacher policy changed')
    with (base/'coordinator.lock').open('a+b') as lock:
        lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        def status(phase,**extra):
            row={'phase':phase,'pid':os.getpid(),'updated_utc':datetime.now(timezone.utc).isoformat(),'deadline_utc':deadline_iso,'training_only':training_only,'disk_free_gib':shutil.disk_usage(base).free/2**30,**extra}
            atomic(base/'status.json',row);print(json.dumps(row),flush=True)
        def wait_external(name,pid,artifact):
            if artifact.exists():return
            status('waiting_'+name,external_pid=pid,artifact=str(artifact))
            while alive(pid):time.sleep(2)
            if not artifact.exists():raise RuntimeError(f'{name} exited without {artifact}')
        def child(command,log,allow_result_failure=False):
            with log.open('ab') as out:
                proc=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                status('running_child',child_pid=proc.pid,command=command,log=str(log));code=proc.wait()
            if code and not allow_result_failure:raise RuntimeError(f'Child exited {code}; inspect {log}')
            return code
        def wsl(script,*args):return ['wsl','-d','Ubuntu','--','bash','/mnt/c/AMI/scripts/wsl_python.sh',linux(ROOT/'scripts'/script),*args]
        def learner(work,behavior,dataset,local_cfg,initializer=None,critic_gate=False):
            pointer=read(work/'selected-learner.json')
            if pointer:return pointer
            candidates=[]
            # The only approved fallback changes effective batch16 ->32.
            for accumulation in (1,2):
                candidate=work/f'learner-batch{16*accumulation}';conf=work/f'config-batch{16*accumulation}.json';replace(local_cfg,gradient_accumulation=accumulation).write(conf)
                result=read(candidate/'complete.json')
                if result is None:
                    if candidate.exists():candidate=available_dir(work,candidate.name)
                    command=wsl('train_selfplay.py','--config',linux(conf),'--dataset',linux(dataset),'--checkpoint',linux(behavior),'--output',linux(candidate),'--train','--skip-value-warmup')
                    if initializer:command+=['--value-initializer',linux(initializer)]
                    if critic_gate:command+=['--critic-gate']
                    child(command,work/f'{candidate.name}.log',allow_result_failure=True);result=read(candidate/'complete.json')
                    if result is None:raise RuntimeError(f'Learner technical/critic-gate failure; inspect {candidate}')
                candidates.append({'path':str(candidate),'result':result})
                if result['publishable'] and result['frame_coverage']>=.8:
                    pointer={'checkpoint':str(candidate/'checkpoint.pt'),'learner':str(candidate),'accumulation':accumulation,'result':result}
                    atomic(work/'selected-learner.json',pointer);return pointer
            status('pilot_or_cycle_efficiency_review',work=str(work),candidates=candidates)
            raise RuntimeError('Neither approved batch configuration reached 80% committed coverage; diagnose before further collection')
        def evaluate(checkpoint,stage,name,opponent,local_cfg,pilot=False):
            if training_only:raise RuntimeError('Evaluation disabled by user schedule')
            stem='pilot-vs-MIA0' if pilot else 'vs-'+name;pointer=read(stage/(stem+'-reference.json'))
            if pointer:return pointer
            folder=available_dir(stage,stem);status('evaluating',path=str(folder),opponent=name,target=20)
            games=collect(checkpoint,folder,local_cfg,[] if opponent is None else [opponent],matches=20,teacher=opponent is None,versus_previous=opponent is not None,
                          record_prefix=f'restart_{stage.name.lower()}_{stem.lower()}_20261031')
            if len(games)!=20 or read(folder/'collection.json').get('quarantine'):raise RuntimeError('Validation quality gate/incomplete series; inspect preserved games')
            stats=read(folder/'service/stats.json');behaviors=[stats['episodes'][m['run']+':'+str(m['model_team'])] for m in games]
            summary={**evaluation_score(games),'path':str(folder),'reached_mid':sum(b['first_mid_time'] is not None for b in behaviors),'attacked_creeps':sum(b['creep_projectiles']>0 for b in behaviors),'cast_razes':sum(b['raze_casts']>0 for b in behaviors),'behaviors':behaviors,'replays':[export_replay(m['replay']) for m in games if m.get('replay')]}
            atomic(stage/(stem+'-reference.json'),summary);return summary
        if time.time()>=deadline:
            status('deadline_saved');return
        status('waiting_for_readiness')
        while not (base/'implementation-ready.json').exists():time.sleep(2)
        wait_external('archive',bootstrap['archive_pid'],base/'critic-data/dataset.json')
        wait_external('pilot_collection',bootstrap['pilot_pid'],base/'pilot/complete.json')
        pilot_parts=[]
        sources=[base/'pilot/collection/collection.json',*sorted(base.glob('pilot-extra*/collection.json'))]
        for source in sources:
            data=read(source);pilot_parts.append({'path':str(source.parent),'matches':data['matches']})
            if data.get('quarantine') and not (source.parent/'quality-repair.json').exists():
                raise RuntimeError('Pilot transport quarantine needs diagnosis and recorded repair before refill')
        while sum(len(x['matches']) for x in pilot_parts)<32:
            folder=available_dir(base,'pilot-extra');remaining=32-sum(len(x['matches']) for x in pilot_parts)
            status('collecting_pilot_refill',path=str(folder),target=remaining,completed_prior=32-remaining)
            games=collect(base/'pilot/behavior.pt',folder,cfg,matches=remaining,deadline=deadline-900)
            pilot_parts.append({'path':str(folder),'matches':games})
            if read(folder/'collection.json').get('quarantine'):raise RuntimeError('Pilot refill also quarantined; inspect technical cause')
        prior_data=read(base/'pilot-data-reference.json')
        pilot_data=Path(prior_data['path']) if prior_data else base/'pilot-dataset'
        if not (pilot_data/'dataset.json').exists():
            if pilot_data.exists():pilot_data=available_dir(base,'pilot-dataset')
            status('preparing_pilot',path=str(pilot_data),matches=32)
            prepare_parts(pilot_parts,pilot_data,hashlib.sha256((base/'pilot/behavior.pt').read_bytes()).hexdigest())
        atomic(base/'pilot-data-reference.json',{'path':str(pilot_data),'parts':pilot_parts})
        models=base/'models';models.mkdir(exist_ok=True)
        initial=models/'MIA0.pt'
        if not initial.exists():shutil.copy2(ROOT/'runs/bc-continue-30m/best.pt',initial)
        state=read(base/'state.json',{'completed_stages':0,'teacher_retired':False,'models':[str(initial)],'stages':[]})
        prior_warm=read(base/'critic-pretrain-reference.json')
        warm=Path(prior_warm['path']) if prior_warm else base/'critic-pretrain'
        if not (warm/'checkpoint.pt').exists():
            warm=available_dir(base,'critic-pretrain')
            child(wsl('pretrain_value.py','--checkpoint',linux(initial),'--dataset',linux(base/'critic-data'),'--config',linux(base/'config.json'),'--output',linux(warm)),base/(warm.name+'.log'))
            atomic(base/'critic-pretrain-reference.json',{'path':str(warm)})
        stage=base/'MIA1';stage.mkdir(exist_ok=True);first=stage/'cycle-1';first.mkdir(exist_ok=True)
        if not (first/'complete.json').exists():
            selection=learner(first,base/'pilot/behavior.pt',pilot_data,cfg,warm/'checkpoint.pt',critic_gate=True)
            result=evaluate(Path(selection['checkpoint']),stage,'MIA0',str(initial),cfg,pilot=True)
            functional=min(result['reached_mid'],result['attacked_creeps'],result['cast_razes'])>=18
            if not functional or result['wins']<8:
                status('pilot_game_review',result=result)
                raise RuntimeError('Pilot needs behavioral/strength review before long training;20games are not proof of strength')
            atomic(first/'complete.json',{**selection,'collection':pilot_parts,'dataset':str(pilot_data),'pilot_evaluation':result})
        while time.time()<deadline:
            version=state['completed_stages']+1;stage=base/f'MIA{version}';stage.mkdir(exist_ok=True);checkpoint=Path(state['models'][-1]);local_cfg=replace(cfg,seed=cfg.seed+version*100)
            for cycle in (1,2):
                work=stage/f'cycle-{cycle}';work.mkdir(exist_ok=True);completed=read(work/'complete.json')
                if completed:checkpoint=Path(completed['checkpoint']);continue
                if time.time()>=deadline or (not training_only and deadline-time.time()<60*60):
                    status('deadline_reserve',version=version,cycle=cycle,checkpoint=str(checkpoint));atomic(base/'resume-point.json',{'version':version,'cycle':cycle,'checkpoint':str(checkpoint)});return
                behavior=work/'behavior.pt'
                if not behavior.exists():shutil.copy2(checkpoint,behavior)
                if hashlib.sha256(behavior.read_bytes()).digest()!=hashlib.sha256(checkpoint.read_bytes()).digest():raise RuntimeError('Behavior identity changed')
                parts=collection_parts(work);total=sum(len(p['matches']) for p in parts)
                while total<32:
                    if time.time()>=deadline:
                        status('deadline_saved_partial_collection',version=version,cycle=cycle,accepted=total);return
                    folder=available_dir(work,'collection');status('collecting',version=version,cycle=cycle,path=str(folder),target=32,completed_prior=total)
                    prefix=f'restart_mia{version}_cycle{cycle}_selfplay_{folder.name}_20261031' if training_only else None
                    games=collect(behavior,folder,replace(local_cfg,seed=local_cfg.seed+cycle+len(parts)),state['models'],matches=32-total,deadline=deadline if training_only else deadline-15*60,record_prefix=prefix)
                    parts.append({'path':str(folder),'matches':games});total+=len(games)
                    atomic(work/'collection-parts.json',{'parts':parts,'accepted':total})
                    if read(folder/'collection.json').get('quarantine'):raise RuntimeError('Collection quarantined; data preserved, PPO blocked')
                    if not games or (folder/'drain').exists():
                        status('collection_drained',version=version,cycle=cycle,accepted=total);return
                if total!=32:raise ValueError('Unexpected recovered collection size')
                collection_ref={'parts':parts,'matches':[m for p in parts for m in p['matches']]}
                atomic(work/'collection-reference.json',collection_ref)
                replays=[export_replay(m['replay']) for m in collection_ref['matches'] if m.get('replay')]
                if replays:atomic(work/'selfplay-replays.json',replays)
                data_ref=read(work/'dataset-reference.json')
                if not data_ref:
                    folder=available_dir(work,'dataset');status('preparing',version=version,cycle=cycle,path=str(folder))
                    meta=prepare_parts(parts,folder,hashlib.sha256(behavior.read_bytes()).hexdigest());data_ref={'path':str(folder),'frames':sum(e['frames'] for e in meta['episodes']),'matches':meta['matches']};atomic(work/'dataset-reference.json',data_ref)
                selected=learner(work,behavior,Path(data_ref['path']),replace(local_cfg,seed=local_cfg.seed+cycle),critic_gate=True)
                checkpoint=Path(selected['checkpoint']);atomic(work/'complete.json',{**selected,'collection':collection_ref,'dataset':data_ref})
            candidate=stage/f'MIA{version}.candidate.pt'
            if not candidate.exists():shutil.copy2(checkpoint,candidate)
            validation=read(stage/'validation.json',{})
            for name,opponent in ([] if training_only else opponents_for(version,state['teacher_retired'],state['models'])):
                if name in validation:continue
                if time.time()>=deadline and not validation:
                    status('deadline_saved',version=version,candidate=str(candidate));return
                validation[name]=evaluate(candidate,stage,name,opponent,local_cfg);atomic(stage/'validation.json',validation)
            named=models/f'MIA{version}.pt'
            if not named.exists():shutil.copy2(candidate,named)
            row={'version':version,'checkpoint':str(named),'validation':validation,'validation_status':'deferred_by_user' if training_only else 'completed','completed_utc':datetime.now(timezone.utc).isoformat()};atomic(stage/'complete.json',row)
            state['completed_stages']=version;state['models'].append(str(named));state['stages'].append(row)
            if validation.get('teacher',{}).get('wins',0)>=16:state['teacher_retired']=True
            atomic(base/'state.json',state);status('stage_complete',result=row)
        status('deadline_saved',state=state)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        print(f'FAILED: {type(error).__name__}: {error}',file=sys.stderr,flush=True);raise
