"""Best-effort shared heartbeat index; each job also has its durable local status."""
import json
import os
from pathlib import Path
import sys
import time

def publish(name,record,pilot=Path('C:/AMI/runs/pilot-status.json')):
    temporary=pilot.with_name(pilot.name+f'.p07-{os.getpid()}.pending')
    for attempt in range(20):
        try:
            current=json.loads(pilot.read_text(encoding='utf-8'))
            current.setdefault('P07_jobs',{})[name]=record
            temporary.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
            temporary.replace(pilot)
            return True
        except PermissionError:
            time.sleep(.05)
    print('P07 status index busy; local job status retained, retry next update',file=sys.stderr)
    return False
