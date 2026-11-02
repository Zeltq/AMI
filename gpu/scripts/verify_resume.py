"""Integration check: real PPO continuation, pinned environment, corruption and stop."""
import argparse,json,shutil,subprocess,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',default='gpu/target/release/sf-duel-gpu.exe');p.add_argument('--config',default='gpu/config-100.json');p.add_argument('--run',required=True);a=p.parse_args()
    binary=Path(a.binary).resolve();root=Path(a.run).resolve();root.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(a.config).read_text());source=root/'source-engine.json'
    shutil.copyfile(config['engine'],source);config.update(engine=str(source),matches=4,ppo_epochs=1,batch_sequences=4)
    configuration=root/'config.json';configuration.write_text(json.dumps(config,indent=2))
    commands=[]
    def execute(args,label,success=True):
        command=[str(binary),*map(str,args)];commands.append(command)
        with (root/f'{label}.out.log').open('w')as out,(root/f'{label}.err.log').open('w')as err:
            result=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,timeout=600)
        if (result.returncode==0)!=success:raise RuntimeError(f'{label}: unexpected exit {result.returncode}; inspect logs')
    continuous=root/'continuous';split=root/'split'
    execute(['train',configuration,continuous,'--cycles','3','--draft'],'continuous')
    execute(['train',configuration,split,'--cycles','1','--draft'],'first')
    # A checkpoint must use the environment saved with it, even when the
    # original development table has subsequently changed or disappeared.
    source.write_text('{"invalidated_original_table":true}')
    execute(['train',configuration,split,'--cycles','2','--resume','--draft'],'resumed')
    def latest(run):return run/json.loads((run/'latest.json').read_text())['checkpoint']
    one,two=latest(continuous),latest(split)
    states={file.name:file.read_bytes()==(two/file.name).read_bytes()for file in one.glob('*.bin')}
    if not states or not all(states.values()):raise RuntimeError(f'Continuation state differs: {states}')
    updates=[json.loads((checkpoint/'state.json').read_text())['optimizer_updates']for checkpoint in (one,two)]
    if updates[0]!=updates[1] or updates[0]<=0:raise RuntimeError(f'Optimizer did not continue: {updates}')
    for label,checkpoint in (('continuous',one),('split',two)):
        execute(['export',checkpoint,root/f'{label}.pt'],f'export-{label}')
    first=root/'continuous.tensors';second=root/'split.tensors';manifest=json.loads((first/'manifest.json').read_text());errors={}
    for name,entry in manifest['tensors'].items():
        x=np.fromfile(first/entry['file'],dtype='<f4');y=np.fromfile(second/entry['file'],dtype='<f4')
        if not np.isfinite(x).all()or not np.isfinite(y).all():raise RuntimeError('Nonfinite model weight')
        errors[name]=float(np.max(np.abs(x-y)))
    maximum=max(errors.values())
    if maximum>1e-6:raise RuntimeError(f'Resumed PPO weight difference {maximum}')
    corrupt=root/'corrupt';shutil.copytree(split,corrupt)
    file=latest(corrupt)/'world_rng.bin';data=bytearray(file.read_bytes());data[0]^=1;file.write_bytes(data)
    execute(['train',configuration,corrupt,'--cycles','1','--resume','--draft'],'corrupt',False)
    if 'corrupt world_rng'not in (root/'corrupt.err.log').read_text():raise RuntimeError('Wrong reason for corrupt-checkpoint rejection')
    # Only this test owns the process. Always request a normal rollout-boundary
    # stop, including when checking concurrent-writer rejection fails.
    live=root/'stop';shutil.copytree(split,live)
    command=[str(binary),'train',str(configuration),str(live),'--resume','--draft'];commands.append(command)
    with (root/'stop.out.log').open('w')as out,(root/'stop.err.log').open('w')as err:
        process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err)
        try:
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                if process.poll()is not None:raise RuntimeError('Stop-test process exited early')
                try:status=json.loads((live/'status.json').read_text())
                except (FileNotFoundError,json.JSONDecodeError):status={}
                if status.get('pid')==process.pid:break
                time.sleep(.1)
            else:raise RuntimeError('Stop-test process never became ready')
            execute(['train',configuration,live,'--resume','--draft'],'duplicate',False)
            if 'run already locked'not in(root/'duplicate.err.log').read_text():raise RuntimeError('Wrong concurrent-writer rejection')
        finally:
            execute(['stop',live],'request-stop');process.wait(timeout=120)
        if process.returncode:raise RuntimeError('Graceful stop failed')
    execute(['train',configuration,live,'--cycles','1','--resume','--draft'],'after-stop')
    report={'passed':True,'matches':4,'continuous_cycles':3,'resumed_cycles':[1,2],'optimizer_updates':updates,
            'state_buffers_bit_equal':states,'model_max_absolute_error':maximum,'weight_errors':errors,
            'source_environment_invalidated_before_resume':True,'corruption_rejected':True,'duplicate_writer_rejected':True,
            'graceful_stop_and_resume':True,'commands':commands}
    (root/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
