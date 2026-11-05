"""Run one P10 job with persistent process status (no detached children)."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifacts',type=Path,default=Path('C:/AMI/runs/p10'))
    p.add_argument('--status-key',default='P10_jobs')
    p.add_argument('name')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    root = a.artifacts
    root.mkdir(parents=True, exist_ok=True)
    path = root / (a.name + '-job.json')
    if path.exists():
        raise FileExistsError(path)
    start = time.monotonic()
    with (root / (a.name + '.log')).open('wb') as log:
        child = subprocess.Popen(a.command, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        def save():
            row = dict(stage='running' if child.poll() is None else 'complete' if child.returncode == 0 else 'failed',
                       pid=os.getpid(), child_pid=child.pid, command=subprocess.list2cmdline(a.command),
                       cwd=str(Path.cwd()), artifact=str(root), minutes=(time.monotonic()-start)/60,
                       exit_code=child.returncode, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            path.write_text(json.dumps(row, indent=2))
            pilot = Path('C:/AMI/runs/pilot-status.json')
            tmp = pilot.with_name(pilot.name+f'.p10-{os.getpid()}.pending')
            for attempt in range(20):
                try:
                    current = json.loads(pilot.read_text(encoding='utf-8'))
                    current.setdefault(a.status_key, {})[a.name] = row
                    tmp.write_text(json.dumps(current, indent=2), encoding='utf-8')
                    tmp.replace(pilot)
                    break
                except PermissionError:
                    time.sleep(.1)
        try:
            while child.poll() is None:
                save()
                time.sleep(5)
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
            save()
    raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()
