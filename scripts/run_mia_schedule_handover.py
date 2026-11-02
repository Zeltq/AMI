"""Wait for an already-draining coordinator, then resume with its new schedule."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from run_mia_restart import alive
from selfplay_cycle import ROOT,atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--old-pid',type=int,required=True);a=p.parse_args();base=Path(a.run).resolve()
    atomic(base/'handover-status.json',{'phase':'waiting_for_current_matches','old_pid':a.old_pid})
    started=time.monotonic()
    while alive(a.old_pid):
        if time.monotonic()-started>900:raise RuntimeError('Coordinator did not drain; inspect without killing active games')
        time.sleep(2)
    # The resumed coordinator also takes the exclusive coordinator.lock.
    command=[sys.executable,str(ROOT/'scripts/run_mia_restart.py'),'--execute','--run',str(base)]
    with (base/'coordinator-training-only.log').open('ab') as log,(base/'coordinator-training-only-error.log').open('ab') as err:
        child=subprocess.Popen(command,stdout=log,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    atomic(base/'handover-status.json',{'phase':'resumed_training_only','pid':child.pid,'updated_utc':datetime.now(timezone.utc).isoformat()})
    f=ROOT/'runs/pilot-status.json';state=json.loads(f.read_text());state.update(mia_pid=child.pid,mia_coordinator_alive=True,stage='training_only_resumed',mia_log=str(base/'coordinator-training-only.log'),mia_error_log=str(base/'coordinator-training-only-error.log'));atomic(f,state)


if __name__=='__main__':main()
