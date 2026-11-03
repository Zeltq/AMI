"""Run the single P06 dataset and three equal-budget ablations sequentially."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=Path('C:/AMI/runs/p06')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.p06-{os.getpid()}.pending')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def linux(path):
    s=str(path).replace('\\','/')
    return '/mnt/'+s[0].lower()+s[2:]


def jobs(out):
    binary=str(RUN/'sf-duel-gpu.exe');engine=str(RUN/'dota-calibration-004/engine.json')
    wsl=['wsl','-d','Ubuntu','--','bash',linux(ROOT/'scripts/wsl_python.sh')]
    yield 'collect','collection',[binary,'collect-teacher',str(ROOT/'reports/P06/collect.json'),str(out/'dataset')],out/'dataset/report.json'
    for name in ('baseline','honest','oracle'):
        model=out/name
        yield name+'-train','training',wsl+[linux(ROOT/'gpu/scripts/train_ggd.py'),linux(ROOT/f'reports/P06/train-{name}.json'),linux(model)],model/'result.json'
        yield name+'-export','export',wsl+[linux(ROOT/'gpu/scripts/weights.py'),'--checkpoint',linux(model/'epoch-002.pt'),'--output',linux(model/'export')],model/'export/weights/manifest.json'
        for mode in ('sampling','argmax'):
            result=out/(name+'-'+mode)
            yield name+'-'+mode,'evaluation',[binary,'evaluate-ggd',engine,str(model/'export/weights'),str(result),'--mode','teacher','--matches','64','--seconds','600','--seed','20261103','--'+mode,'--trace'],result/'report.json'


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    out=RUN/'experiment';out.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(out/'runner.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    history_path=out/'commands.json'
    if history_path.exists() and not a.resume:raise FileExistsError('Use --resume for an existing experiment')
    history=json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
    state_path=out/'state.json'
    if state_path.exists():
        import ctypes
        previous=json.loads(state_path.read_text(encoding='utf-8'))
        pid=previous.get('pid')
        if pid:
            handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
            if handle:
                code=ctypes.c_ulong()
                ctypes.windll.kernel32.GetExitCodeProcess(handle,ctypes.byref(code))
                ctypes.windll.kernel32.CloseHandle(handle)
                if code.value==259:raise RuntimeError(f'Previous child PID {pid} is still alive; refusing duplicate work')
    def status(value):
        value.update(runner_pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),worktree=str(ROOT))
        save(state_path,value)
        shared=RUN.parent/'pilot-status.json'
        current=json.loads(shared.read_text(encoding='utf-8-sig')) if shared.exists() else {}
        current['P06']=value;save(shared,current)
    for name,kind,command,marker in jobs(out):
        done=next((r for r in history if r['stage']==name and r.get('validated')),None)
        if done:
            if not marker.exists():raise RuntimeError('Completed stage artifact disappeared: '+name)
            continue
        if marker.exists():raise RuntimeError('Unaccounted stage output: '+name)
        if (out/'stop.json').exists():raise InterruptedError('stop.json requested')
        started=time.monotonic()
        with (out/(name+'.stdout.log')).open('w') as stdout,(out/(name+'.stderr.log')).open('w') as stderr:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr)
            record=dict(stage=name,kind=kind,pid=child.pid,argv=command,command=subprocess.list2cmdline(command),
                        artifact=str(marker),started_utc=datetime.now(timezone.utc).isoformat(),background_work_remaining=True)
            status(record.copy())
            while child.poll() is None:
                # An explicit stop is passed through to the active job's own
                # checkpoint/collection stop mechanism; no child is killed.
                progress=marker.parent/'status.json'
                if (out/'stop.json').exists() and kind in ('collection','training','evaluation'):
                    save(marker.parent/'stop.json',{'requested_by':'P06 runner stop.json'})
                details={}
                if progress.exists():
                    try:details=json.loads(progress.read_text(encoding='utf-8'))
                    except (json.JSONDecodeError,OSError):pass
                status({**record,'elapsed_minutes':(time.monotonic()-started)/60,'progress':details})
                time.sleep(10)
        record.update(exit_code=child.returncode,minutes=(time.monotonic()-started)/60,background_work_remaining=False)
        history.append(record);save(history_path,history)
        if child.returncode or not marker.exists():
            status({**record,'stage':'failed','failed_stage':name});raise RuntimeError(f'{name} failed, exit={child.returncode}')
        result=json.loads(marker.read_text(encoding='utf-8'))
        if kind=='collection' and (result.get('partial') or result.get('games_recorded')!=1024):raise RuntimeError('Incomplete collection')
        if kind in ('training','evaluation') and not result.get('complete'):raise RuntimeError('Incomplete '+kind)
        if kind=='training' and result.get('epochs')!=2:raise RuntimeError('Wrong epoch budget')
        record['validated']=True;save(history_path,history)
    subprocess.run([sys.executable,str(ROOT/'gpu/scripts/p06_results.py'),'--root',str(out)],cwd=ROOT,check=True)
    status(dict(stage='experiment complete; report and git verification pending',pid=None,background_work_remaining=False,artifact=str(out)))


if __name__=='__main__':main()
