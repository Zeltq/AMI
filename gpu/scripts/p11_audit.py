"""Input immutability and P11 result checks, without storing model bytes in git."""
import subprocess
from pathlib import Path
from p11_run import ROOT,RUN,REPORT,START,POOL,read,save,sha
from p09b_results import lines
def prepare():
    files=[]
    for weights in POOL:
        files.extend(Path(weights).glob('*.f32'));files.append(Path(weights)/'manifest.json')
    files.extend(START/k for k in ('model.bpk','optimizer.bpk','progress.json'))
    source=subprocess.check_output(['git','ls-files','gpu/src'],cwd=ROOT,text=True).splitlines()
    files.extend(ROOT/p for p in source if p not in ('gpu/src/ggd_ppo.rs','gpu/src/ggd_pool.rs','gpu/src/main.rs'))
    save(REPORT/'protected.json',{str(p):sha(p) for p in files})
def finish():
    initial=read(REPORT/'protected.json');changed=[p for p,h in initial.items() if sha(p)!=h];assert not changed,changed
    checks=[]
    for folder in RUN.glob('train-*'):
        cfg=read(folder/'config.json');rows=lines(folder/'history.jsonl')
        for r in rows:
            assert 0<=r['learning']['clip_fraction']<=1
            # The coefficient belongs to the beginning of a block.
            before=r['training_seconds']-r['collection_seconds']-r['optimization_seconds']
            expected=.03+(.01-.03)*min(1,max(0,before/(cfg['schedule_minutes']*60)))
            assert abs(r['learning']['kl_coefficient']-expected)<1e-6
        checks.append(dict(run=folder.name,blocks=len(rows),clip_and_anchor_checks=len(rows)))
    save(REPORT/'audit.json',dict(protected_files=len(initial),unchanged=len(initial),changed=changed,training=checks,scope='Original weights, Adam, critic, teachers, physics and observation sources; PPO clipping and continuous anchor schedule. Evaluation reward and outcomes checked by p11_report.'))
if __name__=='__main__':
    import sys
    prepare() if sys.argv[1]=='prepare' else finish()
