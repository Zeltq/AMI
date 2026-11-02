"""Sequential Vulkan qualification, scaling and real-PPO measurements on one GPU."""
import argparse,datetime,json,os,shutil,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--binary',default='gpu/target/release/sf-duel-gpu.exe');p.add_argument('--steps',type=int,default=6144);p.add_argument('--cycles',type=int,default=44);a=p.parse_args()
    root=Path(a.run).resolve();root.mkdir(parents=True,exist_ok=False);binary=root/'sf-duel-gpu.exe';shutil.copyfile(a.binary,binary)
    config=json.loads((ROOT/'gpu/config-100.json').read_text());engine=json.loads(Path(config['engine']).read_text());assets=root/'environment';assets.mkdir()
    for key,name in (('geometry','geometry.f32'),('navigation','navigation.bin'),('fow_blockers','fow_blockers.f32'),('movement_geometry','movement-geometry.f32')):
        if engine.get(key):shutil.copyfile(Path(engine['calibration'])/engine[key],assets/name);engine[key]=name
    engine['calibration']=str(assets);atomic(assets/'engine.json',engine);config['engine']=str(assets/'engine.json')
    configs={}
    for count in (100,500,600,650,750):
        value=dict(config,matches=count);file=root/f'config-{count}.json';atomic(file,value);configs[count]=file
    records=[];started=time.time()
    def status(phase,**kwargs):
        atomic(root/'suite-status.json',dict(phase=phase,pid=os.getpid(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.time()-started,completed=records,**kwargs))
    def execute(label,args,allowed_failure=False):
        command=[str(binary),*map(str,args)]
        with(root/f'{label}.out.log').open('w')as out,(root/f'{label}.err.log').open('w')as err:
            process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err);status(label,child_pid=process.pid,command=command)
            try:
                while process.poll()is None:
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:status(label,child_pid=process.pid,command=command)
            except BaseException:
                # The suite owns this child. Training accepts a boundary stop.
                if args[0]=='benchmark'and'--train'in args:
                    subprocess.run([str(binary),'stop',str(args[2])],cwd=ROOT,check=False,stdout=subprocess.DEVNULL)
                    process.wait(timeout=180)
                else:process.terminate();process.wait(timeout=30)
                raise
        row=dict(phase=label,exit_code=process.returncode,command=command,stdout=str(root/f'{label}.out.log'),stderr=str(root/f'{label}.err.log'));records.append(row)
        if process.returncode and not allowed_failure:status('failed',failure=row);raise RuntimeError(f'{label} failed; see {row["stderr"]}')
        return process.returncode==0
    execute('qualify',['qualify'])
    execute('combat',['validate-combat',assets/'engine.json',ROOT/'runs/gpu/dota-combat-008/scenarios.json'])
    for count in (100,500):
        execute(f'collection-{count}',['benchmark',configs[count],root/f'collection-{count}','--steps',str(a.steps),'--warmup','128','--profile'])
        execute(f'physics-{count}',['benchmark',configs[count],root/f'physics-{count}','--steps',str(a.steps),'--warmup','128','--simulation-only','--profile'])
    maximum=500
    for count in (600,650,750):
        if execute(f'capacity-{count}',['benchmark',configs[count],root/f'capacity-{count}','--train','--cycles','2'],True):maximum=count
        else:break
    if maximum>500:
        execute(f'collection-{maximum}',['benchmark',configs[maximum],root/f'collection-{maximum}','--steps',str(a.steps),'--warmup','128','--profile'])
        execute(f'physics-{maximum}',['benchmark',configs[maximum],root/f'physics-{maximum}','--steps',str(a.steps),'--simulation-only','--profile'])
    for count in dict.fromkeys((100,500,maximum)):
        execute(f'train-{count}',['benchmark',configs[count],root/f'train-{count}','--train','--cycles',str(a.cycles)])
    execute('train-100-no-export',['benchmark',configs[100],root/'train-100-no-export','--train','--cycles',str(a.cycles),'--no-artifacts'])
    status('complete',maximum_capacity_probe_passed=maximum)

if __name__=='__main__':main()
