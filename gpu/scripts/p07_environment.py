"""Create environment 005 as a copy of 004 with the measured uphill flag."""
import hashlib,json,shutil
from pathlib import Path

source=Path('C:/AMI/runs/p06/dota-calibration-004')
target=Path('C:/AMI/runs/p07/dota-calibration-005')
target.mkdir(parents=True,exist_ok=False)
old=json.loads((source/'engine.json').read_text(encoding='utf-8'))
new=dict(old,calibration=target.as_posix(),uphill_miss=True)
copied={}
for key in ('geometry','navigation','movement_geometry','fow_blockers','static_vision'):
    name=old.get(key)
    if name:
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dest)
        copied[key]=dict(file=name,bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
(target/'engine.json').write_text(json.dumps(new,indent=2),encoding='utf-8')
report=dict(source=source.as_posix(),target=target.as_posix(),changed_keys=[k for k in new if old.get(k)!=new[k]],copied=copied,
            old_engine_sha256=hashlib.sha256((source/'engine.json').read_bytes()).hexdigest(),new_engine_sha256=hashlib.sha256((target/'engine.json').read_bytes()).hexdigest())
Path('reports/P07/environment.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
collect=dict(engine=(target/'engine.json').as_posix(),games=2,parallel_matches=2,seconds=600,seed=20261103,observation_profile='ggd-v4',teacher_version='v10')
Path('reports/P07/collect.json').write_text(json.dumps(collect,indent=2),encoding='utf-8')
print(json.dumps(report))
