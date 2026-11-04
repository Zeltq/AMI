"""P09c sequential process ledger, bounded GPU time and reproducible preparation."""
import argparse,json,os,subprocess,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
from p09b_run import read,save,sha,linux,pid_alive
ROOT=Path(__file__).resolve().parents[2]
RUN=Path('C:/AMI/runs/p09/P09c');REPORT=ROOT/'reports/P09c'
ENGINE='C:/AMI/runs/p07/dota-calibration-005/engine.json'
INITIAL=Path('C:/AMI/runs/p09/P09b/ZOg-d1/last.pt')
WEIGHTS=INITIAL.parent/'export/weights'
EXE=RUN/'sf-duel-gpu.exe';TARGET=RUN/'target'
WSL=['wsl','-d','Ubuntu','--','bash',linux(ROOT/'scripts/wsl_python.sh')]
LIMIT=300

def history():return read(RUN/'commands.json') if (RUN/'commands.json').exists() else []
def used():return sum(r.get('minutes',0) for r in history() if r.get('gpu_budget'))
def status(row):
    value={**row,'controller_pid':os.getpid(),'updated_utc':datetime.now(timezone.utc).isoformat(),'worktree':str(ROOT),'artifacts':str(RUN)}
    save(RUN/'state.json',value)
    path=Path('C:/AMI/runs/pilot-status.json');shared=read(path) if path.exists() else {};shared['P09c']=value;save(path,shared)

def exclusive():
    import msvcrt
    RUN.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'controller.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    pending=[r for r in history() if r.get('exit_code') is None]
    if any(pid_alive(r.get('pid')) for r in pending):raise RuntimeError('Recorded child still alive')
    probe=subprocess.run(['wsl','-d','Ubuntu','--exec','pgrep','-af','/mnt/c/AMI-wt/P09c/gpu/scripts/(train_ggd|p09b_critic)\\.py'],capture_output=True,text=True,timeout=30)
    if probe.returncode!=1:raise RuntimeError('WSL process present or probe failed: '+probe.stdout+probe.stderr)
    if pending:
        rows=history();now=datetime.now(timezone.utc)
        for r in rows:
            if r.get('exit_code') is None:r.update(exit_code='interrupted',minutes=max(r.get('minutes',0),(now-datetime.fromisoformat(r['started_utc'])).total_seconds()/60))
        save(RUN/'commands.json',rows)
    return lock

def publish(message):
    if (ROOT/'gpu/scripts/p09c_results.py').exists():subprocess.run([sys.executable,str(ROOT/'gpu/scripts/p09c_results.py')],cwd=ROOT,check=True)
    for path in REPORT.rglob('*'):
        if path.is_file() and path.stat().st_size>=1_000_000:raise RuntimeError('Report exceeds 1 MB: '+str(path))
    subprocess.run(['git','add','reports/P09c.md','reports/P09c'],cwd=ROOT,check=True)
    names=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    if any(p!='reports/P09c.md' and not p.startswith('reports/P09c/') for p in names):raise RuntimeError('Unexpected staged files')
    subprocess.run(['git','diff','--cached','--stat'],cwd=ROOT,check=True)
    if names:subprocess.run(['git','commit','-m','[P09c] '+message],cwd=ROOT,check=True)
    ssh='C:/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=2'
    subprocess.run(['git','-c','core.sshCommand='+ssh,'push','-u','origin','agent/P09c-zog-rl2'],cwd=ROOT,check=True,timeout=120)

def run(name,kind,command,marker=None,gpu=False):
    rows=history();prior=[r for r in rows if r['stage']==name]
    if prior and prior[-1].get('exit_code')==0:
        if marker and not Path(marker).exists():raise RuntimeError('Missing completed artifact '+str(marker))
        return
    if prior:raise RuntimeError('Use a new attempt name after diagnosing failure: '+name)
    spent=used();command=list(map(str,command))
    if gpu and spent>=LIMIT:raise RuntimeError('GPU budget exhausted')
    if gpu and command[0]=='wsl':
        i=command.index('--')+1;command=command[:i]+['timeout','--signal=TERM','--kill-after=5s',str(max(1,int((LIMIT-spent)*60)-10))+'s']+command[i:]
    row=dict(stage=name,kind=kind,command=command,command_literal=subprocess.list2cmdline(command),gpu_budget=gpu,started_utc=datetime.now(timezone.utc).isoformat(),artifact=str(marker) if marker else None)
    start=time.monotonic();last_publish=start;last_evaluation=None
    with (RUN/(name+'.stdout.log')).open('w',encoding='utf-8') as out,(RUN/(name+'.stderr.log')).open('w',encoding='utf-8') as err:
        child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err);row['pid']=child.pid;rows.append(row);save(RUN/'commands.json',rows)
        while child.poll() is None:
            elapsed=(time.monotonic()-start)/60;row['minutes']=elapsed;save(RUN/'commands.json',rows)
            progress={}
            if marker:
                try:progress=read(Path(marker).parent/'status.json')
                except (OSError,ValueError):pass
            status({**row,'background_work_remaining':True,'gpu_minutes':spent+(elapsed if gpu else 0),'progress':progress})
            if gpu and spent+elapsed>=LIMIT:
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);child.wait();break
            evaluation=Path(marker).parent/'evaluation-complete.json' if marker else RUN/'absent'
            stamp=evaluation.stat().st_mtime_ns if evaluation.exists() else None
            if kind=='training' and (time.monotonic()-last_publish>=3500 or stamp!=last_evaluation):
                try:publish('publish training curves')
                except Exception as exc:save(RUN/'publication-error.json',{'error':str(exc)})
                last_publish=time.monotonic();last_evaluation=stamp
            time.sleep(5)
    row.update(exit_code=child.returncode,minutes=(time.monotonic()-start)/60);save(RUN/'commands.json',rows)
    save(REPORT/'commands.json',[{k:v for k,v in r.items() if k in ('stage','kind','command_literal','exit_code') or (k=='minutes' and r['kind'] in ('training','evaluation'))} for r in rows])
    save(REPORT/'budget.json',{'used_gpu_minutes':used(),'limit_gpu_minutes':LIMIT})
    status({**row,'background_work_remaining':False})
    if child.returncode or (marker and not Path(marker).exists()):raise RuntimeError(f'{name}: exit={child.returncode}, artifact={marker}')

def prepare():
    os.environ['SF_DUEL_DATA_ROOT']='C:/AMI';os.environ['PYTHONPATH']=str(ROOT/'src')+';'+str(ROOT/'tests')+';C:/AMI/tools/lua_test_runtime'
    seal=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'gpu/src').glob('*.rs')))).hexdigest()[:12]
    cargo=['cargo','test','--release','--locked','--manifest-path',ROOT/'gpu/Cargo.toml','--target-dir',TARGET]
    run('rust-compile-'+seal,'build',cargo+['--no-run'])
    run('rust-targeted-'+seal,'verification',cargo+['p09c_','--','--nocapture','--test-threads=1'],gpu=True)
    run('rust-full-'+seal,'verification',cargo+['--','--test-threads=1'],gpu=True)
    run('python-full-'+seal,'verification',[sys.executable,'-m','unittest','discover','-s','tests','-v'])
    run('build-'+seal,'build',['cargo','build','--release','--locked','--manifest-path',ROOT/'gpu/Cargo.toml','--target-dir',TARGET])
    import shutil
    shutil.copy2(TARGET/'release/sf-duel-gpu.exe',EXE)
    save(REPORT/'verified.json',dict(source_seal=seal,exe_sha256=sha(EXE)))
    for name,old,config,mode in [
        ('demo',Path('C:/AMI/runs/p09/dataset'),ROOT/'reports/P09a/collect.json','teacher'),
        ('d1',Path('C:/AMI/runs/p09/P09b/d1-data'),ROOT/'reports/P09b/d1-collect.json','dagger'),
        ('d2',Path('C:/AMI/runs/p09/P09b/d2-data'),ROOT/'reports/P09b/d2-collect.json','dagger')]:
        target=RUN/(name+'-data');save(REPORT/(name+'-collect.json'),read(config))
        run(name+'-collect','collection',[EXE,'collect-'+mode,REPORT/(name+'-collect.json'),target],target/'report.json',True)
        run(name+'-audit','audit',[sys.executable,ROOT/'gpu/scripts/p09_data.py',target,REPORT/(name+'-audit.json')],REPORT/(name+'-audit.json'))
        run(name+'-parity','audit',[sys.executable,ROOT/'gpu/scripts/p09c_data_parity.py',old,target,REPORT/(name+'-parity.json')],REPORT/(name+'-parity.json'))
        publish('publish exact telemetry '+name)
    status(dict(stage='exact telemetry complete',background_work_remaining=True,pid=None))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare']);args=p.parse_args();lock=exclusive()
    try:prepare()
    except Exception as exc:status(dict(stage='failed',error=str(exc),background_work_remaining=True));raise
