"""Run one P01 background command, retain its exact argv, time, log and PID."""
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def save(path,value):
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)

def pilot_status(record):
    run=(ROOT/'runs/p01').resolve();pilot=run.parent/'pilot-status.json';status=json.loads(pilot.read_text(encoding='utf-8'))
    jobs=[json.loads(p.read_text(encoding='utf-8')) for p in run.glob('*/job.json')]
    status['P01']={**record,'background_work_remaining':any(j['stage']=='running' for j in jobs),'run':str(run),'workspace':str(ROOT),'jobs':jobs}
    save(pilot,status)

def main():
    p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    directory=ROOT/'runs/p01'/a.job;directory.mkdir(parents=True,exist_ok=True)
    with (directory/'started.json').open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid()},f)
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    started=time.perf_counter();record={'job':a.job,'command':command,'command_display':subprocess.list2cmdline(command),'cwd':str(ROOT),'started_utc':datetime.now(timezone.utc).isoformat(),'wrapper_pid':os.getpid(),'log':str(directory/'output.log'),'stage':'running'}
    with (directory/'output.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        record['pid']=process.pid;save(directory/'job.json',record)
        pilot_status(record)
        code=process.wait()
    record.update(exit_code=code,seconds=time.perf_counter()-started,stage='complete' if code==0 else 'failed');save(directory/'job.json',record)
    pilot_status(record)
    raise SystemExit(code)

if __name__=='__main__':main()
