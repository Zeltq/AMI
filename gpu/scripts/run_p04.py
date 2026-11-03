"""Sequential P04 evaluations/parity; artifacts and process state stay outside git."""
import argparse, datetime, json, os, subprocess, time
from pathlib import Path

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('binary',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--engine',default='C:/AMI/runs/gpu/ggd0-il-20261102-202637/dataset/environment/engine.json')
    p.add_argument('--weights',default='C:/AMI/runs/gpu/ggd0-il-20261102-202637/GGD0-resumed/epoch-003/weights')
    p.add_argument('--phase',choices=['evaluation','parity','revision'],default='evaluation')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    # OS lock is released on exit; a stale file never authorizes duplicate work.
    lock=(a.output/'runner.lock').open('a+b')
    import msvcrt
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    history_path=a.output/'commands.json';history=json.loads(history_path.read_text()) if history_path.exists() else []
    jobs=[]
    if a.phase=='evaluation':
        jobs.append(('regression',['teacher',a.engine,'{out}','--matches','200','--seconds','600','--seed','20261103','--wall-seconds','1800']))
        jobs.append(('v9-v9',['evaluate-teachers',a.engine,'{out}','--matches','200','--seed','20261103']))
    if a.phase in ('evaluation','revision'):
        suffix='-revision' if a.phase=='revision' else ''
        jobs.extend([('v10-v9'+suffix,['evaluate-teachers',a.engine,'{out}','--matches','200','--seed','20261103','--left','v10','--right','v9','--swap','--trace']),
                     ('v10-v10'+suffix,['evaluate-teachers',a.engine,'{out}','--matches','200','--seed','20261103','--left','v10','--right','v10'])])
    else:
        if a.phase=='parity':
            collection={'teacher_version':'v10','engine':a.engine,'games':2,'parallel_matches':2,'seconds':600,'seed':20261103}
            for name,command,config in [('parity-teacher','collect-teacher',collection),('parity-dagger','collect-dagger',{'collection':collection,'weights':a.weights,'beta':1.,'round':1})]:
                config_path=a.output/(name+'-input.json');write(config_path,config)
                jobs.append((name,[command,str(config_path),'{out}']))
    for name,args in jobs:
        out=a.output/name
        if (out/'report.json').exists():
            report=json.loads((out/'report.json').read_text())
            if report.get('complete',report.get('finished_prefixes',report.get('games_recorded',0)) in (2,200)) and not report.get('partial',False):continue
            raise RuntimeError(f'Incomplete existing stage: {out}')
        if (a.output/'stop.json').exists():raise InterruptedError('stop.json requested')
        command=[str(a.binary),*[str(out) if v=='{out}' else v for v in args]]
        started=time.monotonic()
        with (a.output/(name+'.stdout.log')).open('w') as stdout,(a.output/(name+'.stderr.log')).open('w') as stderr:
            child=subprocess.Popen(command,stdout=stdout,stderr=stderr)
            state={'task':'P04','stage':name,'pid':child.pid,'runner_pid':os.getpid(),'command':subprocess.list2cmdline(command),'artifacts':str(out),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            write(a.output/'pilot-status.json',state)
            # Update only our namespace in the shared status document.
            shared=a.output.parent/'pilot-status.json'
            if shared.exists():
                status=json.loads(shared.read_text(encoding='utf-8-sig'));status['P04']=state;write(shared,status)
            code=child.wait()
        record={**state,'exit_code':code,'minutes':(time.monotonic()-started)/60};history.append(record);write(history_path,history)
        if code:write(a.output/'pilot-status.json',{**record,'stage':'failed'});raise RuntimeError(f'{name}: exit {code}')
    if a.phase=='parity':
        from run_dagger import compare
        write(a.output/'parity.json',compare(a.output/'parity-teacher',a.output/'parity-dagger'))
    write(a.output/'pilot-status.json',{'task':'P04','stage':a.phase+' complete','pid':None,'artifacts':str(a.output)})

if __name__=='__main__':main()
