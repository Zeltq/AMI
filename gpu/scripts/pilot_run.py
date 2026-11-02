"""Bounded frozen-environment pilot:25min PPO,freeze,100GPU+30native games."""
import argparse,hashlib,json,msvcrt,os,shutil,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=True)+'\n',encoding='utf-8');temp.replace(path)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--minutes',type=float,default=25)
    parser.add_argument('--config',required=True);parser.add_argument('--binary',default='gpu/target/release/sf-duel-gpu.exe')
    parser.add_argument('--native-workers',type=int,choices=(1,2,4),default=2)
    parser.add_argument('--gpu-seed',type=int,default=2026110216);parser.add_argument('--native-seed',type=int,default=2026110217)
    parser.add_argument('--reset-origin',type=float,default=-3.3);parser.add_argument('--readiness',required=True)
    parser.add_argument('--deadline-utc');args=parser.parse_args()
    if not 20<=args.minutes<=30:parser.error('Pilot duration must be20..30minutes')
    os.chdir(ROOT);run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=False)
    baseline=ROOT/'runs/gpu/mgia-series-001/baseline';original=baseline/'MIA2-original.pt';before=baseline/'MIA2-executor.pt'
    identity={str(p):digest(p)for p in [original,before,baseline/'burn-weights/manifest.json']}
    readiness=json.loads(Path(args.readiness).read_text())
    if not readiness.get('functional_pilot_ready'):raise ValueError('Functional pilot readiness has not been recorded')
    config=json.loads(Path(args.config).read_text())
    if config['matches']!=100 or Path(config['initial_checkpoint']).resolve()!=baseline/'burn-weights' or [Path(p).resolve()for p in config['opponents']]!=[baseline/'burn-weights']:
        raise ValueError('Pilot requires100matches and immutable originalMIA2 as initial policy and opponent')
    exe=run/'sf-duel-gpu.exe';shutil.copy2(Path(args.binary),exe)
    source=Path(config['engine']);environment=run/'locked-environment';environment.mkdir()
    engine=json.loads(source.read_text());calibration=Path(engine['calibration'])
    for key,name in [('geometry','geometry.f32'),('navigation','navigation.bin'),('fow_blockers','fow_blockers.f32'),('movement_geometry','movement-geometry.f32'),('static_vision','static-vision.u32')]:
        if engine.get(key):shutil.copy2(calibration/engine[key],environment/name);engine[key]=name
    engine['calibration']=str(environment);atomic(environment/'engine.json',engine)
    shutil.copy2(args.readiness,run/'readiness.json')
    shutil.copytree(ROOT/'gpu/src',run/'source/gpu-src');shutil.copy2(ROOT/'gpu/Cargo.lock',run/'source/Cargo.lock');shutil.copy2(ROOT/'gpu/Cargo.toml',run/'source/Cargo.toml')
    config.update(engine=str(environment/'engine.json'))
    atomic(run/'training-config.json',config)
    manifest={'baseline':identity,'executable_sha256':digest(exe),'source_revision':'84 GPU and119 Python checks; native common-clock and core lane diagnostics in readiness.json',
        'duration_minutes':args.minutes,'matches':100,'gpu_evaluation_matches':100,'native_evaluation_matches':30,'native_workers':args.native_workers,
        'evaluation_seeds':{'gpu':args.gpu_seed,'dota':args.native_seed},'evaluation_selection':'argmax','memory_reset_origin_seconds':args.reset_origin,
        'learning_rate':config['learning_rate'],'environment_files':{p.name:digest(p)for p in environment.iterdir()if p.is_file()},
        'environment_changes_allowed':'No fidelity expansion; only demonstrated critical observation/action/reward/terminal fixes, separately recorded',
        'approximations':readiness['remaining_approximations'],
        'acceptance':'Absence of obvious degradation in reaching mid,LH/deny,levels,deaths and stuck behavior; no claim of proven improvement or full DoD',
        'created_utc':datetime.now(timezone.utc).isoformat()}
    atomic(run/'pilot-manifest.json',manifest)
    status={'pid':os.getpid(),'run':str(run),'training':False,'baseline':identity}
    def update(phase,**extra):
        status.update(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**extra);atomic(run/'pilot-status.json',status)
    def command(argv,label,phase):
        with (run/(label+'.stdout.log')).open('w')as out,(run/(label+'.stderr.log')).open('w')as err:
            child=subprocess.Popen([str(v)for v in argv],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            update(phase,child_pid=child.pid,command=list(map(str,argv)),training=False)
            while child.poll()is None:time.sleep(2)
            if child.returncode:raise RuntimeError(f'{label} exited {child.returncode}; inspect logs')
    with (run/'pilot.lock').open('a+b')as lock:
        lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            training=run/'training';argv=[str(exe),'train',str(run/'training-config.json'),str(training),'--draft']
            with (run/'training.stdout.log').open('w')as out,(run/'training.stderr.log').open('w')as err:
                started=time.monotonic();child=subprocess.Popen(argv,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                deadline=started+args.minutes*60;requested=False
                update('training',child_pid=child.pid,command=argv,training=True,started_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=0)
                while child.poll()is None:
                    elapsed=time.monotonic()-started
                    if not requested and time.monotonic()>=deadline:
                        atomic(training/'stop.json',{'reason':'Authorized bounded25minute pilot complete; freeze before evaluation'});requested=True
                    update('stopping'if requested else'training',elapsed_seconds=elapsed,stop_requested=requested)
                    if elapsed>args.minutes*60+240:
                        child.terminate();child.wait(timeout=15);raise RuntimeError('Trainer did not honor graceful stop within4minutes')
                    time.sleep(5)
                if child.returncode:raise RuntimeError(f'Trainer exited {child.returncode}')
            result=json.loads((training/'result.json').read_text());assert result['optimizer_steps']>0,'No real PPO updates'
            checkpoint=training/json.loads((training/'latest.json').read_text())['checkpoint'];frozen=run/'frozen-checkpoint';shutil.copytree(checkpoint,frozen)
            atomic(run/'freeze.json',{'source_checkpoint':str(checkpoint),'frozen_checkpoint':str(frozen),'elapsed_seconds':time.monotonic()-started,'optimizer_steps':result['optimizer_steps'],'weights_sha256':digest(frozen/'model.bpk'),'training':False})
            checkpoint_pt=run/'pilot-checkpoint.pt';command([exe,'export',frozen,checkpoint_pt],'export','exporting')
            import torch
            from policy_history_audit import bundle_parity
            parity=bundle_parity(torch.load(checkpoint_pt,map_location='cpu',weights_only=False)['state_dict'],checkpoint_pt.with_suffix('.tensors'))
            atomic(run/'export-parity.json',parity)
            if not parity['bit_equal']:raise RuntimeError('Burn/executor export weights differ')
            evaluation=dict(config,engine=str(training/'environment/engine.json'),seed=args.gpu_seed,initial_checkpoint=str(frozen/'model.bpk'))
            atomic(run/'gpu-evaluation-config.json',evaluation)
            command([exe,'evaluate',run/'gpu-evaluation-config.json',run/'gpu-evaluation','--argmax',f'--reset-origin={args.reset_origin}'],'gpu-evaluation','evaluating_gpu')
            native_args=['python',ROOT/'gpu/scripts/evaluate_dota.py','--checkpoint',checkpoint_pt,'--before',before,'--run',run/'dota-evaluation','--matches','30','--workers',str(args.native_workers),'--seed',str(args.native_seed),'--inference-device','cpu','--selection','argmax','--reset-origin',str(args.reset_origin),'--record-prefix','sf_core_pilot002']
            if args.deadline_utc:native_args+=['--deadline-utc',args.deadline_utc]
            command(native_args,'dota-evaluation','evaluating_dota')
            command(['python',ROOT/'gpu/scripts/behavior_report.py','--evaluation',run/'dota-evaluation/evaluation.json','--output',run/'dota-behavior-report.json'],'native-behavior','reporting_behavior')
            if any(digest(Path(p))!=value for p,value in identity.items()):raise RuntimeError('Immutable baseline changed')
            update('complete',child_pid=None,training=False,gpu_report=str(run/'gpu-evaluation/evaluation.json'),native_report=str(run/'dota-evaluation/evaluation.json'))
        except BaseException as error:
            update('failed',training=False,error=repr(error));raise
if __name__=='__main__':main()
