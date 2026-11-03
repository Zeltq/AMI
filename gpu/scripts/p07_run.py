"""Run one bounded P07 job with durable status, logs and elapsed time."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

name, *command = sys.argv[1:]
root=Path('C:/AMI/runs/p07');root.mkdir(parents=True,exist_ok=True)
start=time.monotonic()
with (root/f'{name}.stdout.log').open('w') as out, (root/f'{name}.stderr.log').open('w') as err:
    child=subprocess.Popen(command,stdout=out,stderr=err)
    while True:
        code=child.poll()
        status=dict(stage=name if code is None else 'complete' if code==0 else 'failed',pid=child.pid if code is None else None,
                    runner_pid=os.getpid(),command=subprocess.list2cmdline(command),cwd=str(Path.cwd()),
                    exit_code=code,minutes=(time.monotonic()-start)/60,background_work_remaining=code is None,
                    checked_utc=datetime.now(timezone.utc).isoformat(),artifacts=[str(root/f'{name}.stdout.log'),str(root/f'{name}.stderr.log')])
        (root/f'{name}.status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
        pilot=Path('C:/AMI/runs/pilot-status.json');current=json.loads(pilot.read_text(encoding='utf-8'))
        current.setdefault('P07_jobs',{})[name]=status
        temporary=pilot.with_suffix(f'.p07-{os.getpid()}.tmp');temporary.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(pilot)
        if code is not None:break
        time.sleep(10)
sys.exit(code)
