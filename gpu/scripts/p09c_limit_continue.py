"""Apply the user's reduced training limit to an already-running native process."""
import json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
from p09b_run import read,save,pid_alive
ROOT=Path(__file__).resolve().parents[2]
RUN=Path('C:/AMI/runs/p09/P09c/continue90')
REPORT=ROOT/'reports/P09c/continue90'

def status(**values):
    values.update(updated_utc=datetime.now(timezone.utc).isoformat())
    save(RUN/'limit-status.json',values)
    path=Path('C:/AMI/runs/pilot-status.json');shared=read(path);shared['P09c_limit']=values;save(path,shared)

def main():
    import msvcrt
    lock=(RUN/'limit-controller.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    change=read(REPORT/'limit-change.json');folder=Path(change['native_run']);target=change['target_training_seconds']
    native=change['native_pid'];controller=change['old_controller_pid'];last_status=0.;sent=False
    while pid_alive(native):
        p=folder/'status.json';s=read(p);row=s.get('latest',{});trained=row.get('training_seconds')
        age=max(0.,time.time()-p.stat().st_mtime)
        # The native check occurs at a complete PPO-block boundary. Set its
        # supported stop file shortly before the target within the current
        # block; never terminate GPU work or rewrite a checkpoint counter.
        if trained is not None and target-trained<=60 and age>=max(0.,target-trained-.5):
            save(folder/'stop.json',dict(reason='User reduced continuation to 50 total PPO minutes',target_training_seconds=target,observed_training_seconds=trained,observed_status_age_seconds=age,requested_utc=datetime.now(timezone.utc).isoformat()))
            sent=True;break
        if time.monotonic()-last_status>=5:
            status(stage='waiting for requested 50-minute boundary',pid=os.getpid(),native_pid=native,controller_pid=controller,command='C:/Python314/python.exe gpu/scripts/p09c_limit_continue.py',target_training_seconds=target,observed_training_seconds=trained,background_work_remaining=True,artifacts=str(RUN));last_status=time.monotonic()
        time.sleep(.1)
    if sent:
        status(stage='native stop requested; waiting for checkpoint and old controller',pid=os.getpid(),native_pid=native,controller_pid=controller,background_work_remaining=True)
    while pid_alive(native) or pid_alive(controller):time.sleep(.25)
    result=read(folder/'result.json');actual=result['training_seconds'];checkpoint=Path(result['latest'])
    assert abs(actual-target)<=30,'Stop outside one PPO-block tolerance: '+str(actual-target)
    assert all((checkpoint/p).exists() for p in ('model.bpk','optimizer.bpk','progress.json','weights/manifest.json'))
    jobs=read(RUN/'commands.json');job=next(r for r in reversed(jobs) if r['stage']==folder.name);assert job['exit_code']==0
    result.update(user_requested_training_minutes=50,limit_amended=True)
    save(RUN/'training-result.json',result)
    save(REPORT/'limit-result.json',dict(native_exit_code=0,requested_training_seconds=target,actual_training_seconds=actual,additional_training_minutes=(actual-change['start_training_seconds'])/60,stop_file_used=sent,checkpoint=str(checkpoint),old_controller_guard='The old Python controller still expected 90 minutes and may reject the requested early completion; native training completed normally. New controller reads the saved training result and performs evaluations only.'))
    with (RUN/'controller-002.stdout.log').open('w',encoding='utf-8') as out,(RUN/'controller-002.stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen([sys.executable,str(ROOT/'gpu/scripts/p09c_continue90.py')],cwd=ROOT,stdout=out,stderr=err)
    status(stage='handoff complete; evaluations controller started',pid=None,controller_pid=child.pid,background_work_remaining=False,training_result=str(RUN/'training-result.json'))

if __name__=='__main__':main()
