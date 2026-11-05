"""Verify the P12 CPU export bitwise and inspect original P10 bot recovery."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'gpu/scripts')]
from weights import rust_name
from install_play_bot import inventory
OUT=ROOT/'reports/P12'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    source=Path('C:/AMI/runs/p11/train-001/checkpoint-000644/weights')
    export=Path('C:/AMI/runs/p12/ZOg-000644.pt')
    manifest=json.loads((source/'manifest.json').read_text())
    saved=torch.load(export,map_location='cpu',weights_only=False)
    rows=[]
    for name,tensor in saved['state_dict'].items():
        entry=manifest['tensors'][rust_name(name)];path=source/entry['file']
        value=np.fromfile(path,dtype='<f4').reshape(entry['shape'])
        if value.ndim==2 and not name.endswith('_embed.weight'):value=value.T
        equal=value.tobytes()==tensor.numpy().tobytes()
        assert equal and sha(path)==entry['sha256'],name
        rows.append(dict(name=name,source_sha256=sha(path),bitwise_equal=equal))
    assert len(rows)==len(manifest['tensors'])
    result=dict(checkpoint=str(source.parent),export=str(export),export_sha256=sha(export),
                manifest_sha256=sha(source/'manifest.json'),equal_tensors=len(rows),total_tensors=len(rows),
                tensors=rows,pipeline=saved['gpu_pipeline'],config=saved['config'])
    (OUT/'provenance.json').write_text(json.dumps(result,indent=2)+'\n')
    original=json.loads(Path('C:/AMI/runs/p10/installation.json').read_text())
    assert inventory(Path(original['backup']))==original['before']
    current=inventory(Path(original['path']))
    changes={key:dict(recorded=original['installed_files'].get(key),current=current.get(key))
             for key in sorted(current.keys()|original['installed_files'].keys())
             if current.get(key)!=original['installed_files'].get(key)}
    record=dict(original_backup=original['backup'],original_verified=True,
                original_entries=len(original['before']),original_files=sum(v!='directory' for v in original['before'].values()),
                current_entries=len(current),differences_from_p10=changes,current_inventory=current)
    path=OUT/'installation-before.json'
    if not path.exists():path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(equal=len(rows),total=len(rows),original_entries=len(original['before']),differences=list(changes))))
if __name__=='__main__':main()
