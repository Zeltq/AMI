"""Byte comparison of legacy v2/v3 collection before and after P07."""
import gzip,hashlib,json,subprocess,sys
from pathlib import Path

root=Path('C:/AMI/runs/p07');out=Path('reports/P07');rows=[]
for profile in ('ggd-v2','ggd-v3'):
    config=dict(engine='C:/AMI/runs/p06/dota-calibration-004/engine.json',games=2,parallel_matches=2,seconds=600,
                seed=20261103,observation_profile=profile,teacher_version='v9',max_blocks=8)
    path=out/f'compat-{profile}.json';path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    blocks=[]
    for version,binary in [('before',root/'baseline.exe'),('after',root/'target/release/sf-duel-gpu.exe')]:
        name=f'compat-{profile}-{version}';target=root/name
        subprocess.run([sys.executable,'gpu/scripts/p07_run.py',name,str(binary),'collect-teacher',str(path.resolve()),str(target)],check=True)
        files=sorted(target.glob('*.ggd.gz'));assert len(files)==8,(name,len(files))
        blocks.append([gzip.decompress(f.read_bytes()) for f in files])
    for i,(before,after) in enumerate(zip(*blocks)):
        assert before==after,(profile,i)
        rows.append(dict(profile=profile,block=i,bytes=len(before),equal=True,sha256=hashlib.sha256(before).hexdigest()))
report=dict(profiles=2,blocks=rows,equal_blocks=len(rows),total_blocks=16,equal_bytes=sum(r['bytes'] for r in rows),
            scope='First 1024 decisions per actor, 2 matches per profile, environment 004; decompressed GGDDemo2 header, FP16 observations and all 13 FP32 signal columns')
(out/'legacy-bytes.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
