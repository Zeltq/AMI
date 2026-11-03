"""Sequential bounded verification; no model training, no duplicate pipeline."""
import json,msvcrt,os,subprocess,sys
from pathlib import Path

root=Path('C:/AMI/runs/p07')
lock=(root/'verification.lock').open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
os.environ['SF_DUEL_DATA_ROOT']='C:/AMI'
binary='C:/AMI/runs/p07/target/release/sf-duel-gpu.exe'
cargo=['--release','--locked','--manifest-path','gpu/Cargo.toml','--target-dir','C:/AMI/runs/p07/target']
stages=[
 ('rust-narrow-final',['cargo','test',*cargo,'p07_','--','--test-threads=1']),
 ('rust-full',['cargo','test',*cargo,'--','--test-threads=1']),
 ('build-final',['cargo','build',*cargo]),
 ('public-parity-final',[binary,'check-public-spells','C:/AMI/runs/p06/dota-calibration-004/engine.json','C:/AMI/runs/p07/spell-fixture.json','C:/AMI-wt/P07/reports/P07/gpu-parity.json']),
 ('regression-004',[binary,'evaluate-teachers','C:/AMI/runs/p06/dota-calibration-004/engine.json','C:/AMI/runs/p07/regression-004','--matches','200','--seed','20261103']),
 ('evaluation-005',[binary,'evaluate-teachers','C:/AMI/runs/p07/dota-calibration-005/engine.json','C:/AMI/runs/p07/evaluation-005','--matches','200','--seed','20261103']),
 ('collect-v4',[binary,'collect-teacher','C:/AMI-wt/P07/reports/P07/collect.json','C:/AMI/runs/p07/collect-v4']),
]
Path('reports/P07/verification-plan.json').write_text(json.dumps([dict(stage=n,command=subprocess.list2cmdline(c)) for n,c in stages],indent=2),encoding='utf-8')
for name,command in stages:
    print(name,flush=True)
    subprocess.run([sys.executable,'gpu/scripts/p07_run.py',name,*command],check=True)
