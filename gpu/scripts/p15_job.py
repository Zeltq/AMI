"""Run one named P15 background job and persist its exact command and exit code."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from p15_probe import save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('name')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    root=Path('C:/AMI/runs/p15');job=root/'jobs'/a.name
    job.mkdir(parents=True,exist_ok=False)
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not command:
        p.error('command required')
    started=time.monotonic()
    record=dict(stage='starting',pid=os.getpid(),cwd=str(Path.cwd()),command=subprocess.list2cmdline(command),
                argv=command,background_work_remaining=True,artifacts=[str(job/'stdout.log'),str(job/'stderr.log')])
    def publish(**values):
        record.update(values,elapsed_minutes=(time.monotonic()-started)/60)
        save(job/'status.json',record)
        path=Path('C:/AMI/runs/pilot-status.json')
        current=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        current.setdefault('P15_jobs',{})[a.name]=record
        save(path,current)
    publish()
    try:
        with (job/'stdout.log').open('w') as out,(job/'stderr.log').open('w') as err:
            process=subprocess.Popen(command,stdout=out,stderr=err)
            while True:
                code=process.poll()
                publish(stage='running' if code is None else ('complete' if code==0 else 'failed'),
                        child_pid=process.pid,exit_code=code,background_work_remaining=code is None)
                if code is not None:
                    break
                time.sleep(5)
        raise SystemExit(code)
    except Exception as error:
        publish(stage='failed',error=str(error),background_work_remaining=False)
        raise


if __name__=='__main__':
    main()
