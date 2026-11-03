"""Supply the missing teacher timeout vote to a running pre-fix P05 server.

Only referee telemetry is emitted, after both the model's native timeout and
the teacher's exact 600-second hero sample exist. No policy order is issued.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.console import write_console_command


def timeout_vote(log):
    native=None;heroes={}
    for line in log.splitlines():
        if line.startswith('SF_NATIVE_END|'):
            parts=line.split('|')
            if len(parts)==5 and parts[3]=='timeout' and abs(float(parts[2])-600)<.04:
                native=int(parts[1])
        elif line.startswith('SF_P05_HERO|'):
            row=json.loads(line.split('|',1)[1])
            if abs(row['time']-600)<.04:heroes[row['team']]=row
    if native is None or 5-native not in heroes:return None
    teacher=5-native
    return f'SF_NATIVE_END|{teacher}|600|timeout|{heroes[teacher]["lh"]}'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path)
    p.add_argument('--wall-timeout',type=float,default=300);a=p.parse_args()
    deadline=time.monotonic()+a.wall_timeout
    while time.monotonic()<deadline:
        if (a.run/'duel-result.json').exists():return
        path=a.run/'dota.log'
        vote=timeout_vote(path.read_text(errors='replace')) if path.exists() else None
        if vote:
            pid=int((a.run/'dota.pid').read_text())
            command='script print('+json.dumps(vote)+')'
            write_console_command(pid,command)
            (a.run/'timeout-repair.json').write_text(json.dumps(dict(pid=pid,command=command,
                evidence='Native model timeout and teacher hero telemetry both at 600 seconds',policy_changed=False),indent=2))
            print(vote,flush=True);return
        time.sleep(2)
    raise TimeoutError('No verified terminal state within wall-timeout')


if __name__=='__main__':main()
