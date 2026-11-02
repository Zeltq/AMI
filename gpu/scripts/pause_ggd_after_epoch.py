"""Pause this GGD job after both evaluations, retaining its optimizer checkpoint."""
import argparse,ctypes,datetime,json,os,subprocess,sys,time
from pathlib import Path

def write(path,value):
    temp=path.with_suffix('.pause-tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}

def alive(pid):
    kernel=ctypes.windll.kernel32;handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return False
    code=ctypes.c_ulong()
    try:return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
    finally:kernel.CloseHandle(handle)

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--epoch',type=int,required=True);p.add_argument('--linux-pid',type=int,required=True);a=p.parse_args()
    run=Path(a.run).resolve();out=run/'GGD0';request=run/'pause-request.json';epoch=out/f'epoch-{a.epoch:03d}'
    status={'requested_epoch':a.epoch,'stage':'waiting_for_second_evaluation','pid':os.getpid(),'linux_pid':a.linux_pid,'requested_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with request.open('x') as f:json.dump(status,f,indent=2)
    def progress(stage,**extra):
        status.update(stage=stage,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra);write(request,status);print(json.dumps(status),flush=True)
    expected=str(out).replace('\\','/').replace('C:/','/mnt/c/')
    cmd=subprocess.check_output(['wsl','-d','Ubuntu','--','cat',f'/proc/{a.linux_pid}/cmdline']).replace(b'\0',b' ').decode()
    if 'train_ggd.py' not in cmd or expected not in cmd:raise RuntimeError('Refusing to signal a process that does not match this training run')
    # Stop only the Python parent while its Windows evaluator keeps running.
    # Its parameters/optimizer are already saved. It cannot enter the next epoch.
    while True:
        live=read(out/'status.json');native=read(epoch/'vs-mirror/status.json')
        if read(epoch/'vs-mirror/report.json').get('complete') and read(epoch/'vs-teacher/report.json').get('complete'):break
        if live.get('epoch',0)>a.epoch:raise RuntimeError('Requested boundary has already passed')
        if native.get('pid') and alive(native['pid']):break
        time.sleep(.2)
    subprocess.run(['wsl','-d','Ubuntu','--','kill','-STOP',str(a.linux_pid)],check=True)
    progress('waiting_for_evaluation_finish',evaluator_pid=native.get('pid'),interrupted_epoch=live.get('epoch'),discarded_updates_after_boundary=max(0,live.get('updates',0)-read(epoch/'report.json').get('optimizer_steps',a.epoch*7995)))
    while not read(epoch/'vs-mirror/report.json').get('complete'):
        if not alive(native['pid']):raise RuntimeError('Evaluator exited without a complete report; trainer remains suspended for inspection')
        time.sleep(.2)
    import torch
    from train_ggd import write_epoch_csv
    source=out/f'epoch-{a.epoch:03d}.pt';checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    if checkpoint['epoch']!=a.epoch:raise RuntimeError('Checkpoint epoch mismatch')
    gameplay={}
    for mode in ('teacher','mirror'):
        path=epoch/f'vs-{mode}/report.json';report=read(path)
        if not report.get('complete'):raise RuntimeError('Both complete evaluations are required')
        gameplay[mode]={'report':str(path),'summary':report['summary'],'wall_seconds':report['wall_seconds']}
    checkpoint['history'][-1]['gameplay']=gameplay
    checkpoint['paused_after_epoch']=a.epoch
    target=out/f'paused-epoch-{a.epoch:03d}.pt';temporary=target.with_suffix('.pause-tmp')
    torch.save(checkpoint,temporary);temporary.replace(target)
    for name in (f'epoch-{a.epoch:03d}.pt','last.pt'):torch.save(checkpoint,out/name)
    best=min(checkpoint['history'],key=lambda v:v['validation']['nll'])['epoch']
    best_saved=checkpoint if best==a.epoch else torch.load(out/f'epoch-{best:03d}.pt',map_location='cpu',weights_only=False)
    torch.save(best_saved,out/'best.pt')
    write(out/'history.json',checkpoint['history']);write(epoch/'report.json',checkpoint['history'][-1]);write_epoch_csv(out,checkpoint['history'])
    progress('releasing_training_process',checkpoint=str(target),optimizer_steps=checkpoint['optimizer_steps'])
    # TERM is delivered on CONT before Python can resume optimization.
    subprocess.run(['wsl','-d','Ubuntu','--','kill','-TERM',str(a.linux_pid)],check=True)
    subprocess.run(['wsl','-d','Ubuntu','--','kill','-CONT',str(a.linux_pid)],check=True)
    parent=read(run/'status.json').get('pid')
    deadline=time.monotonic()+60
    while parent and alive(parent) and time.monotonic()<deadline:time.sleep(.2)
    if parent and alive(parent):raise RuntimeError('Supervisor did not finish after requested pause')
    if (run/'error.json').exists():
        (run/'error.json').replace(run/'user-pause-process-exit.json')
    result={'family':'GGD','model_name':'GGD0','stage':'paused','complete':False,'paused_at_user_request':True,'completed_epochs':a.epoch,'target_epochs':checkpoint['config']['epochs'],'next_epoch':a.epoch+1,'optimizer_steps':checkpoint['optimizer_steps'],'checkpoint':str(target),'history':checkpoint['history'],'background_work_remaining':False,'pid':None,'child_pid':None,'command':None,'run':str(run),'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(out/'status.json',result);write(run/'status.json',result);write(run/'pause-result.json',result)
    pilot=run.parents[1]/'pilot-status.json';saved=read(pilot);saved['ggd0_il_20261102']=result;write(pilot,saved)
    progress('paused',checkpoint=str(target),next_epoch=a.epoch+1)

if __name__=='__main__':main()
