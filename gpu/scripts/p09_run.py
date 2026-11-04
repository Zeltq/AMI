"""Sequential P09 jobs, OS lock, exact commands and a cumulative 240-minute cap."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
RUN=Path('C:/AMI/runs/p09');REPORT=ROOT/'reports/P09a'
ENGINE='C:/AMI/runs/p07/dota-calibration-005/engine.json'
EXE=str(RUN/'sf-duel-gpu.exe')

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pending')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    for attempt in range(20):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)

def linux(p):
    s=str(p).replace('\\','/');return '/mnt/'+s[0].lower()+s[2:]

def status(row):
    value={**row,'runner_pid':os.getpid(),'checked_utc':datetime.now(timezone.utc).isoformat(),'worktree':str(ROOT),'heartbeat_automation':'p09a-zog'}
    save(RUN/'state.json',value)
    shared=RUN.parent/'pilot-status.json';current=json.loads(shared.read_text(encoding='utf-8-sig')) if shared.exists() else {}
    current['P09a']=value;save(shared,current)

def run(name,kind,command,marker=None):
    history_path=RUN/'commands.json';history=json.loads(history_path.read_text()) if history_path.exists() else []
    prior=[x for x in history if x['stage']==name]
    if prior and prior[-1].get('exit_code')==0:
        if marker and not Path(marker).exists():raise RuntimeError('Missing completed artifact '+str(marker))
        return
    if prior:raise RuntimeError('Use a new stage name and fresh output for a failed attempt: '+name)
    used=sum(x.get('minutes',0) for x in history if x['gpu_budget'])
    gpu=kind in ('collection','training','evaluation','verification','export')
    if gpu and used>=240:raise RuntimeError('GPU budget exhausted')
    if gpu and command[0]=='wsl':
        # Killing wsl.exe alone need not terminate the Linux trainer.
        at=command.index('--')+1
        command=command[:at]+['timeout','--signal=TERM','--kill-after=10s',str(max(1,int((240-used)*60)-15))+'s']+command[at:]
    started=time.monotonic()
    row=dict(stage=name,kind=kind,command=command,command_literal=subprocess.list2cmdline(command),environment={k:os.environ[k] for k in ('SF_DUEL_DATA_ROOT','PYTHONPATH') if k in os.environ},artifact=str(marker) if marker else None,gpu_budget=gpu,started_utc=datetime.now(timezone.utc).isoformat(),background_work_remaining=True)
    with (RUN/(name+'.stdout.log')).open('w') as out,(RUN/(name+'.stderr.log')).open('w') as err:
        child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err)
        row['pid']=child.pid;history.append(row);save(history_path,history)
        try:
            while child.poll() is None:
                elapsed=(time.monotonic()-started)/60;progress={}
                if marker:
                    try:progress=json.loads(Path(marker).with_name('status.json').read_text())
                    except (OSError,ValueError):pass
                if 'cohort_offset' in progress:
                    progress['completed_games']=progress['cohort_offset']+progress['finished_in_cohort']
                    progress['completed_games_per_minute']=progress['completed_games']/max(elapsed,1e-6)
                    progress['blocks_per_minute']=progress['blocks']/max(elapsed,1e-6)
                if progress.get('stage')=='gameplay_evaluation':
                    native=progress['report'].replace('/mnt/c/','C:/')
                    try:progress['evaluation']=json.loads(Path(native).with_name('status.json').read_text())
                    except (OSError,ValueError):pass
                status({**row,'elapsed_minutes':elapsed,'gpu_minutes_used':used+(elapsed if gpu else 0),'progress':progress})
                if (gpu and used+elapsed>=240) or (RUN/'stop.json').exists():
                    # Hard cap also terminates WSL/native descendants; preserve existing checkpoints.
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
                    raise RuntimeError('Budget/stop requested')
                time.sleep(10)
        finally:
            row.update(exit_code=child.poll(),minutes=(time.monotonic()-started)/60,background_work_remaining=False)
            save(history_path,history)
            if name=='publish-metrics' and child.poll()==0:
                published=json.loads((REPORT/'commands.json').read_text(encoding='utf-8'))
                published.append({k:row[k] for k in ('stage','kind','command_literal','exit_code')});save(REPORT/'commands.json',published)
            else:
                save(REPORT/'commands.json',[{k:v for k,v in r.items() if k!='minutes' or r['kind'] in ('collection','training','evaluation')} for r in history])
            status(row)
    if child.returncode or (marker and not Path(marker).exists()):raise RuntimeError(f'{name}: exit={child.returncode}, artifact={marker}')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','tests','main']);a=p.parse_args()
    import msvcrt
    RUN.mkdir(parents=True,exist_ok=True);lock=(RUN/'runner.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (RUN/'state.json').exists():
        prior=json.loads((RUN/'state.json').read_text())
        if prior.get('background_work_remaining'):
            import ctypes
            handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,prior['pid'])
            if handle:
                code=ctypes.c_ulong();ctypes.windll.kernel32.GetExitCodeProcess(handle,ctypes.byref(code));ctypes.windll.kernel32.CloseHandle(handle)
                if code.value==259:raise RuntimeError('Previous child remains alive')
    py=[sys.executable];wsl=['wsl','-d','Ubuntu','--','bash',linux(ROOT/'scripts/wsl_python.sh')]
    if a.stage=='smoke':
        run('smoke4-collect','collection',[EXE,'collect-teacher',str(REPORT/'collect-smoke.json'),str(RUN/'smoke4')],RUN/'smoke4/report.json')
        run('smoke4-audit','audit',py+[str(ROOT/'gpu/scripts/p09_data.py'),str(RUN/'smoke4'),str(REPORT/'smoke-audit.json')],REPORT/'smoke-audit.json')
        run('p07-parity','audit',py+[str(ROOT/'gpu/scripts/p09_parity.py'),'C:/AMI/runs/p07/collect-v4',str(RUN/'smoke4'),str(REPORT/'p07-parity.json')],REPORT/'p07-parity.json')
    elif a.stage=='tests':
        os.environ['SF_DUEL_DATA_ROOT']='C:/AMI'
        run('rust-targeted-2','verification',['cargo','test','--release','--locked','--manifest-path',str(ROOT/'gpu/Cargo.toml'),'--target-dir','C:/AMI/gpu/target','p09_','--','--nocapture'])
        run('rust-tests','verification',['cargo','test','--release','--locked','--manifest-path',str(ROOT/'gpu/Cargo.toml'),'--target-dir','C:/AMI/gpu/target','--','--test-threads=1'])
        env=os.environ;env['PYTHONPATH']=str(ROOT/'src')+';'+str(ROOT/'tests')+';C:/AMI/tools/lua_test_runtime'
        run('python-tests','cpu-check',py+['-m','unittest','discover','-s','tests','-v'])
        run('prepare-training-smoke','audit',py+[str(ROOT/'gpu/scripts/p09_training_smoke.py')],REPORT/'train-smoke.json')
        run('training-smoke','training',wsl+[linux(ROOT/'gpu/scripts/train_ggd.py'),linux(REPORT/'train-smoke.json'),linux(RUN/'training-smoke'),'--max-updates','2'],RUN/'training-smoke/result.json')
        run('export-smoke','export',wsl+[linux(ROOT/'gpu/scripts/weights.py'),'--checkpoint',linux(RUN/'training-smoke/epoch-001.pt'),'--output',linux(RUN/'smoke-export'),'--fixture',linux(RUN/'training-fixture.npz'),'--steps','128'],RUN/'smoke-export/weights/manifest.json')
        run('validate-smoke','verification',[EXE,'validate',str(RUN/'smoke-export')])
        run('publish-verification','audit',py+[str(ROOT/'gpu/scripts/p09_verification.py')],REPORT/'verification.json')
    else:
        assert (REPORT/'smoke-audit.json').exists()
        history=json.loads((RUN/'commands.json').read_text())
        for required in ('smoke4-audit','p07-parity','rust-targeted-2','rust-tests','python-tests','training-smoke','export-smoke','validate-smoke','publish-verification'):
            assert any(r['stage']==required and r.get('exit_code')==0 for r in history),required
        run('main-collect','collection',[EXE,'collect-teacher',str(REPORT/'collect.json'),str(RUN/'dataset')],RUN/'dataset/report.json')
        run('main-audit','audit',py+[str(ROOT/'gpu/scripts/p09_data.py'),str(RUN/'dataset'),str(REPORT/'dataset-audit.json')],REPORT/'dataset-audit.json')
        audit=json.loads((REPORT/'dataset-audit.json').read_text())
        config=dict(dataset=str(RUN/'dataset'),model_name='ZOg',epochs=3,batch_sequences=8,learning_rate=1e-4,seed=20261102,gradient_norm=1.,attack_weight=4.,timing_honest=True,timing_oracle=False,action_weights=audit['action_weights'],hero_attack_weight=audit['hero_attack_weight'],evaluation=dict(executable=EXE,engine=ENGINE,matches=64,seconds=600,seeds=[20261103,20261104],modes=['teacher'],teacher_version='v10'))
        save(REPORT/'train.json',config)
        run('train','training',wsl+[linux(ROOT/'gpu/scripts/train_ggd.py'),linux(REPORT/'train.json'),linux(RUN/'ZOg')],RUN/'ZOg/result.json')
        for seed in (20261103,20261104):
            target=RUN/f'teacher-{seed}'
            run(f'teacher-{seed}','evaluation',[EXE,'evaluate-teachers',ENGINE,str(target),'--matches','64','--seconds','600','--seed',str(seed),'--left','v10','--right','v10'],target/'report.json')
        run('publish-metrics','audit',py+[str(ROOT/'gpu/scripts/p09_results.py')],ROOT/'reports/P09a.md')
    status(dict(stage=a.stage+' complete',pid=None,background_work_remaining=False,artifact=str(RUN)))

if __name__=='__main__':main()
