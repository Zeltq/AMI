"""P09b sequential jobs with OS lock, restart records and total GPU budget."""
import argparse,hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=Path('C:/AMI/runs/p09/P09b'); REPORT=ROOT/'reports/P09b'
EXE=RUN/'sf-duel-gpu.exe'
ENGINE='C:/AMI/runs/p07/dota-calibration-005/engine.json'
DATA=Path('C:/AMI/runs/p09/dataset')
INITIAL=Path('C:/AMI/runs/p09/ZOg/epoch-003.pt')
WEIGHTS=Path('C:/AMI/runs/p09/ZOg/epoch-003/weights')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rust_sources():return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted((ROOT/'gpu/src').glob('*.rs'))}
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    for i in range(30):
        try:tmp.replace(p);return
        except PermissionError:
            if i==29:raise
            time.sleep(.1)
def linux(p):
    s=str(p).replace('\\','/');return '/mnt/'+s[0].lower()+s[2:]
def pid_alive(pid):
    if not pid:return False
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:
        if ctypes.get_last_error()==87:return False
        raise OSError(ctypes.get_last_error(),'Cannot inspect previous child')
    try:
        code=wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
        return code.value==259
    finally:kernel.CloseHandle(handle)
def recover_interrupted_jobs(now=None):
    """Charge unknown process lifetimes conservatively before any new GPU job."""
    path=RUN/'commands.json'
    if not path.exists():return
    history=read(path);pending=[r for r in history if r.get('exit_code') is None]
    if any(pid_alive(r.get('pid')) for r in pending):raise RuntimeError('A previous recorded child is still alive')
    now=now or datetime.now(timezone.utc)
    for row in pending:
        observed=(now-datetime.fromisoformat(row['started_utc'])).total_seconds()/60
        row.update(exit_code='controller_interrupted',minutes=max(row.get('minutes',0),observed),recovery_note='Upper bound through first observation that the child is dead; may include idle time.',recovered_utc=now.isoformat())
    if pending:save(path,history)
def exclusive():
    """One controller, including after a Windows/WSL launcher failure."""
    RUN.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(RUN/'controller.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    query='Get-CimInstance Win32_Process | Where-Object { $_.Name -match "^(sf[-_]duel[-_]gpu.*|rl-gpu)\\.exe$" } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress'
    native=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',query],capture_output=True,text=True,timeout=30,check=True)
    found=json.loads(native.stdout) if native.stdout.strip() else [];found=found if isinstance(found,list) else [found]
    busy=[r for r in found if not r.get('CommandLine') or 'c:/ami' in r['CommandLine'].replace('\\','/').lower()]
    if busy:raise RuntimeError('Native simulator/test process remains alive: '+str([r['ProcessId'] for r in busy]))
    linux_jobs=subprocess.run(['wsl','-d','Ubuntu','--exec','pgrep','-af','/mnt/c/AMI-wt/P09b/gpu/scripts/(train_ggd|p09b_critic)\\.py'],capture_output=True,text=True,timeout=30)
    if linux_jobs.returncode not in (0,1):raise RuntimeError('Cannot check WSL child processes: '+linux_jobs.stderr)
    if linux_jobs.returncode==0:raise RuntimeError('WSL trainer remains alive: '+linux_jobs.stdout.strip())
    recover_interrupted_jobs();return lock
def publish(reason):
    """Only P09b reports are staged; every evaluation is published promptly."""
    subprocess.run([sys.executable,str(ROOT/'gpu/scripts/p09b_results.py')],cwd=ROOT,check=True)
    for path in REPORT.rglob('*'):
        if path.is_file() and path.stat().st_size>=1_000_000:raise RuntimeError('Report file too large: '+str(path))
    subprocess.run(['git','add','reports/P09b.md','reports/P09b'],cwd=ROOT,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    if any(p!='reports/P09b.md' and not p.startswith('reports/P09b/') for p in staged):raise RuntimeError('Unrelated staged files; defer automated publication')
    subprocess.run(['git','diff','--cached','--stat'],cwd=ROOT,check=True)
    if staged:subprocess.run(['git','commit','-m','[P09b] '+reason],cwd=ROOT,check=True)
    subprocess.run(['git','push','-u','origin','agent/P09b-zog-rl'],cwd=ROOT,check=True)
    save(RUN/'publication.json',dict(reason=reason,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),updated_utc=datetime.now(timezone.utc).isoformat()))
def status(row):
    value={**row,'controller_pid':os.getpid(),'updated_utc':datetime.now(timezone.utc).isoformat(),'worktree':str(ROOT),'artifacts':str(RUN)}
    save(RUN/'state.json',value)
    path=RUN.parents[1]/'pilot-status.json';shared=read(path) if path.exists() else {}
    shared['P09b']=value;save(path,shared)
def run(name,kind,command,marker=None,gpu=False):
    path=RUN/'commands.json';history=read(path) if path.exists() else []
    prior=[r for r in history if r['stage']==name]
    if prior and prior[-1].get('exit_code')==0:
        if marker and not Path(marker).exists():raise RuntimeError('Completed artifact missing: '+str(marker))
        return
    if prior:raise RuntimeError('Failed stage needs explicit resume/new attempt: '+name)
    used=sum(r.get('minutes',0) for r in history if r.get('gpu_budget'))
    if gpu and used>=840:raise RuntimeError('GPU budget exhausted')
    command=list(map(str,command))
    if gpu and command[0]=='wsl':
        i=command.index('--')+1
        command=command[:i]+['timeout','--signal=TERM','--kill-after=10s',str(max(1,int((840-used)*60)-15))+'s']+command[i:]
    row=dict(stage=name,kind=kind,command=command,command_literal=subprocess.list2cmdline(command),gpu_budget=gpu,started_utc=datetime.now(timezone.utc).isoformat(),artifact=str(marker) if marker else None)
    started=time.monotonic();last_publication=started;last_evaluation=None;publication_retry=0
    with (RUN/(name+'.stdout.log')).open('w',encoding='utf-8') as out,(RUN/(name+'.stderr.log')).open('w',encoding='utf-8') as err:
        child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err)
        row['pid']=child.pid;history.append(row);save(path,history)
        try:
            while child.poll() is None:
                minutes=(time.monotonic()-started)/60;progress={}
                if marker:
                    try:progress=read(Path(marker).parent/'status.json')
                    except (OSError,ValueError):pass
                status({**row,'background_work_remaining':True,'elapsed_minutes':minutes,'gpu_minutes':used+(minutes if gpu else 0),'progress':progress})
                if kind=='training' and name.startswith('rl'):
                    evaluation=Path(marker).parent/'evaluation-complete.json' if marker else RUN/'absent'
                    stamp=evaluation.stat().st_mtime_ns if evaluation.exists() else None
                    now=time.monotonic()
                    if now>=publication_retry and (now-last_publication>=3600 or stamp!=last_evaluation):
                        try:
                            publish('publish scheduled evaluation' if stamp!=last_evaluation else 'publish hourly RL curves')
                            last_publication=now;last_evaluation=stamp
                        except Exception as exc:
                            save(RUN/'publication-error.json',dict(error=str(exc),updated_utc=datetime.now(timezone.utc).isoformat()))
                            publication_retry=now+300
                if (RUN/'stop.json').exists() or (gpu and used+minutes>=840):
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
                    raise RuntimeError('Stop requested or GPU budget reached')
                time.sleep(10)
        finally:
            row.update(exit_code=child.poll(),minutes=(time.monotonic()-started)/60)
            save(path,history)
            save(REPORT/'commands.json',[{k:v for k,v in r.items() if k not in ('pid','gpu_budget','started_utc') and (k!='minutes' or r['kind'] in ('collection','training','evaluation'))} for r in history])
            save(REPORT/'budget.json',dict(used_minutes=sum(r.get('minutes',0) for r in history if r.get('gpu_budget')),limit_minutes=840))
            status({**row,'background_work_remaining':False})
    if child.returncode or (marker and not Path(marker).exists()):raise RuntimeError(f'{name}: exit={child.returncode}; marker={marker}')
    if kind=='training' and name.startswith('rl'):publish('publish completed RL stage')

def collect_config(games,weights,beta,round_number):
    return dict(collection=dict(engine=ENGINE,games=games,parallel_matches=min(games,128),seconds=600,seed=20261105,observation_profile='ggd-v4',teacher_version='v10',telemetry=True),weights=str(weights),beta=beta,round=round_number)
def dagger():
    from run_dagger import aggregate,compare
    wsl=['wsl','-d','Ubuntu','--','bash',linux(ROOT/'scripts/wsl_python.sh')]
    cfg=collect_config(2,WEIGHTS,1.,1)
    save(REPORT/'parity-dagger.json',cfg);save(REPORT/'parity-teacher.json',cfg['collection'])
    for mode in ('teacher','dagger'):
        target=RUN/('parity-'+mode)
        run('parity-'+mode,'collection',[EXE,'collect-'+mode,REPORT/('parity-'+mode+'.json'),target],target/'report.json',True)
    save(REPORT/'parity.json',compare(RUN/'parity-teacher',RUN/'parity-dagger'))
    datasets=[DATA];checkpoint=INITIAL;weights=WEIGHTS
    for k,beta in enumerate((.5,.25),1):
        target=RUN/f'd{k}-data';cfg=collect_config(256,weights,beta,k);save(REPORT/f'd{k}-collect.json',cfg)
        run(f'd{k}-collect','collection',[EXE,'collect-dagger',REPORT/f'd{k}-collect.json',target],target/'report.json',True)
        run(f'd{k}-audit','audit',[sys.executable,ROOT/'gpu/scripts/p09_data.py',target,REPORT/f'd{k}-audit.json'],REPORT/f'd{k}-audit.json')
        datasets.append(target);aggregate(datasets,RUN/f'd{k}-aggregate',k)
        cfg=dict(dataset=str(RUN/f'd{k}-aggregate'),model_name=f'ZOg-d{k}',epochs=1,batch_sequences=8,learning_rate=1e-4,seed=20261102,gradient_norm=1.,attack_weight=4.,hero_attack_weight=1.,action_weights={'6':4.,'7':2.93,'8':4.},dagger=True,timing_honest=True,timing_oracle=False)
        save(REPORT/f'd{k}-train.json',cfg)
        if k==1:
            run('d1-train-smoke','training',wsl+[linux(ROOT/'gpu/scripts/train_ggd.py'),linux(REPORT/'d1-train.json'),linux(RUN/'d1-train-smoke'),'--initialize',linux(checkpoint),'--max-updates','2'],RUN/'d1-train-smoke/result.json',True)
        trained=RUN/f'ZOg-d{k}'
        run(f'd{k}-train','training',wsl+[linux(ROOT/'gpu/scripts/train_ggd.py'),linux(REPORT/f'd{k}-train.json'),linux(trained),'--initialize',linux(checkpoint)],trained/'result.json',True)
        checkpoint=trained/'last.pt';weights=trained/'export/weights'
        run(f'd{k}-export','export',wsl+[linux(ROOT/'gpu/scripts/weights.py'),'--checkpoint',linux(checkpoint),'--output',linux(trained/'export')],weights/'manifest.json',True)
        for seed in (20261103,20261104):
            evaluation=RUN/f'd{k}-vs-v10-{seed}'
            run(f'd{k}-eval-{seed}','evaluation',[EXE,'evaluate-ggd',ENGINE,weights,evaluation,'--mode','teacher','--matches','64','--seconds','600','--seed',seed,'--sampling','--teacher-version','v10'],evaluation/'report.json',True)
    status(dict(stage='dagger complete; selection and critic preparation pending',background_work_remaining=True,pid=None))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','dagger','verify']);a=p.parse_args()
    lock=exclusive()
    os.environ['SF_DUEL_DATA_ROOT']='C:/AMI'
    if a.stage=='build':
        run('build-dagger','build',['cargo','build','--release','--locked','--manifest-path',ROOT/'gpu/Cargo.toml','--target-dir','C:/AMI/gpu/target'])
        import shutil
        shutil.copy2('C:/AMI/gpu/target/release/sf-duel-gpu.exe',EXE)
    elif a.stage=='verify':
        sources=rust_sources()
        cargo=['cargo','test','--release','--locked','--manifest-path',ROOT/'gpu/Cargo.toml','--target-dir','C:/AMI/gpu/target']
        run('rust-p09b-tests','verification',cargo+['p09b_','--','--test-threads=1'],gpu=True)
        run('rust-full-tests','verification',cargo+['--','--test-threads=1'],gpu=True)
        run('build-rl-verified','build',['cargo','build','--release','--locked','--manifest-path',ROOT/'gpu/Cargo.toml','--target-dir','C:/AMI/gpu/target'])
        import shutil
        binary=RUN/'rl-gpu.exe';shutil.copy2('C:/AMI/gpu/target/release/sf-duel-gpu.exe',binary)
        for name,opponent in [('teacher',None),('different-weights',Path('C:/AMI/runs/p09/ZOg/epoch-002/weights'))]:
            target=RUN/('check-'+name);command=[binary,'evaluate-ggd',ENGINE,WEIGHTS,target,'--mode','teacher','--matches','2','--seconds','600','--seed','20261107','--sampling','--teacher-version','v10']
            if opponent:command+=['--opponent-weights',opponent]
            run('check-'+name,'evaluation',command,target/'report.json',True)
            report=read(target/'report.json');assert report['complete'] and len(report['per_match'])==2
            residual=[]
            for game in report['per_match']:
                for side,hero in enumerate(game['sides']):
                    assert hero['reward_components'][0]==10*game['result']*(1 if side==0 else -1)
                    assert abs(hero['reward']-sum(hero['reward_components']))<1e-4
                    residual+=list(map(abs,hero['shaping_telescoping_residual']))
                assert abs(sum(h['reward'] for h in game['sides']))<1e-4
            assert max(residual)<.02,(name,max(residual))
            save(REPORT/('check-'+name+'.json'),dict(matches=2,passed=True,max_shaping_telescoping_residual=max(residual),residual_tolerance=.02,command_literal=__import__('subprocess').list2cmdline(list(map(str,command))),minutes=report['wall_seconds']/60))
        assert sources==rust_sources(),'Rust source changed during verification'
        save(REPORT/'verified-gpu.json',dict(passed=True,executable_sha256=sha(binary),sources=sources,checks=['rust-p09b-tests','rust-full-tests','check-teacher','check-different-weights']))
        status(dict(stage='verification complete; selection and critic training pending',pid=None,background_work_remaining=True))
    else:dagger()

if __name__=='__main__':main()
