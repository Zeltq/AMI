"""Run one named P16 background job and persist its exact command and exit code."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from p15_probe import save

def shared_status(path):
    for attempt in range(20):
        try:return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        except (PermissionError,json.JSONDecodeError):
            if attempt==19:raise
            time.sleep(.05)

def adopt_process(pid):
    """Keep a Windows process handle so exit status survives process termination."""
    import ctypes
    kernel=ctypes.windll.kernel32;kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:raise ProcessLookupError(pid)
    def poll():
        code=ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(code)):raise OSError('GetExitCodeProcess failed')
        return None if code.value==259 else code.value
    return poll,lambda:kernel.CloseHandle(ctypes.c_void_p(handle))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adopt',action='store_true',help='Monitor the existing job child; never spawn another process')
    p.add_argument('name')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    root=Path('C:/AMI/runs/p16');job=root/'jobs'/a.name
    job.mkdir(parents=True,exist_ok=a.adopt)
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not command and not a.adopt:
        p.error('command required')
    started=time.monotonic()
    if a.adopt:
        record=shared_status(job/'status.json');poll,close=adopt_process(record['child_pid'])
        prior_elapsed=record.get('elapsed_minutes',0);record['previous_wrapper_pid']=record['pid'];record['pid']=os.getpid()
        record['adopted']=True;record.pop('error',None)
    else:
        prior_elapsed=0
        record=dict(stage='starting',pid=os.getpid(),cwd=str(Path.cwd()),command=subprocess.list2cmdline(command),
                    argv=command,background_work_remaining=True,artifacts=[str(job/'stdout.log'),str(job/'stderr.log')])
    def publish(**values):
        record.update(values,elapsed_minutes=prior_elapsed+(time.monotonic()-started)/60)
        save(job/'status.json',record)
        path=Path('C:/AMI/runs/pilot-status.json')
        current=shared_status(path)
        current.setdefault('P16_jobs',{})[a.name]=record
        save(path,current)
    publish(stage='starting',background_work_remaining=True)
    try:
        with (job/'stdout.log').open('a' if a.adopt else 'w') as out,(job/'stderr.log').open('a' if a.adopt else 'w') as err:
            if not a.adopt:
                process=subprocess.Popen(command,stdout=out,stderr=err);poll=process.poll
                record['child_pid']=process.pid
            while True:
                code=poll()
                publish(stage='running' if code is None else ('complete' if code==0 else 'failed'),
                        exit_code=code,background_work_remaining=code is None)
                if code is not None:
                    break
                time.sleep(5)
        raise SystemExit(code)
    except Exception as error:
        publish(stage='failed',error=str(error),background_work_remaining=False)
        raise
    finally:
        if a.adopt:close()


if __name__=='__main__':
    main()
