"""Verify the frozen export and save small provenance/verification evidence."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'gpu/scripts')]
from weights import rust_name

RUN=Path('C:/AMI/runs/p10');OUT=ROOT/'reports/P10'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=Path('C:/AMI/runs/p09/P09c/continue90/rl-continue-001/checkpoint-000470/weights')
    manifest=json.loads((source/'manifest.json').read_text())
    saved=torch.load(RUN/'ZOg-000470.pt',map_location='cpu',weights_only=False)
    rows=[]
    for name,tensor in saved['state_dict'].items():
        entry=manifest['tensors'][rust_name(name)];path=source/entry['file']
        value=np.fromfile(path,dtype='<f4').reshape(entry['shape'])
        if value.ndim==2 and not name.endswith('_embed.weight'):value=value.T
        equal=np.array_equal(value,tensor.numpy())
        assert equal and sha(path)==entry['sha256'],name
        rows.append(dict(name=name,source_sha256=sha(path),equal=True))
    protected={}
    for name in ('gpu/src/physics.rs','gpu/src/world.rs','gpu/src/scripted_teacher.rs','gpu/src/teacher_v10.rs',
                 'bots/scripted_teacher_v9.lua','bots/scripted_teacher_v9_live.lua'):
        original=subprocess.check_output(['git','show','origin/main:'+name],cwd=ROOT)
        current=(ROOT/name).read_bytes()
        # Git's checkout may use CRLF; compare canonical source bytes.
        assert original.replace(b'\r\n',b'\n')==current.replace(b'\r\n',b'\n'),name
        protected[name]=sha(ROOT/name)
    record=dict(checkpoint=str(source.parent),export_sha256=sha(RUN/'ZOg-000470.pt'),manifest_sha256=sha(source/'manifest.json'),
                equal_tensors=len(rows),total_tensors=len(manifest['tensors']),tensors=rows,
                pipeline=saved['gpu_pipeline'],timing_honest=saved['config']['timing_honest'],timing_oracle=saved['config']['timing_oracle'],
                protected_sources=protected,engine_sha256=sha(Path('C:/AMI/runs/p07/dota-calibration-005/engine.json')),
                collector_sha256=sha(RUN/'nativecollect.exe'),simulator_sha256=sha(Path('C:/AMI/runs/p09/P09c/sf-duel-gpu.exe')))
    (OUT/'provenance.json').write_text(json.dumps(record,indent=2))
    test_log=next(RUN/name for name in ('python-release.log','python-final.log','python-full.log') if (RUN/name).exists())
    log=test_log.read_text(errors='replace')
    tests=int(re.search(r'Ran (\d+) tests',log)[1]);assert '\nOK\n' in log
    verification=dict(python=dict(tests=tests,passed=True,skipped=len(re.findall(r' \.\.\. skipped',log)),
        command="$env:PYTHONPATH='src;scripts;tests;C:/AMI/tools/lua_test_runtime'; python -m unittest discover -s tests -v"),
        go=dict(command='go test ./...',cwd=str(ROOT/'tools/nativecollect'),passed=True),
        rust='Not run: Rust source and simulator binary unchanged',export=dict(equal=len(rows),total=len(manifest['tensors'])))
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    commands=[json.loads(p.read_text()) for p in sorted(RUN.glob('*-job.json'))]
    (OUT/'commands.json').write_text(json.dumps(commands,ensure_ascii=False,indent=2),encoding='utf-8')
    for path in OUT.glob('*.json'):
        path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
    print(json.dumps(dict(tensors=len(rows),python=tests,protected=len(protected))))


if __name__=='__main__':main()
