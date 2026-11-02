"""Restore the existing executor's PyTorch policy from a Burn tensor export."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.model import Policy
from weights import rust_name

def verified_manifest(root):
    """Pin legacy Burn exports without hashes; reject damaged hashed exports."""
    path=root/'manifest.json';manifest=json.loads(path.read_text());changed=False
    for name,entry in manifest['tensors'].items():
        raw=(root/entry['file']).read_bytes()
        if len(raw)!=4*int(np.prod(entry['shape'])):raise ValueError(f'Wrong byte count for {name}')
        checksum=hashlib.sha256(raw).hexdigest()
        if entry.get('sha256'):
            if entry['sha256']!=checksum:raise ValueError(f'Weight bundle checksum mismatch: {name}')
        else:entry['sha256']=checksum;changed=True
    if changed:
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(path)
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('bundle');p.add_argument('output');a=p.parse_args();root=Path(a.bundle)
    torch.set_num_threads(2);manifest=verified_manifest(root);model=Policy();state={}
    for name,expected in model.state_dict().items():
        key=rust_name(name);entry=manifest['tensors'][key]
        value=np.fromfile(root/entry['file'],dtype='<f4').reshape(entry['shape'])
        if value.ndim==2 and not name.endswith('_embed.weight'):value=value.T
        if value.shape!=tuple(expected.shape):raise ValueError(f'Wrong shape for {name}')
        if not np.isfinite(value).all():raise ValueError(f'Nonfinite weight {name}')
        state[name]=torch.from_numpy(value.copy())
    if len(state)!=len(manifest['tensors']):raise ValueError('Unused Burn tensors')
    model.load_state_dict(state,strict=True)
    config={'teacher_policy_sha256':hashlib.sha256((ROOT/'tools/nativecollect/teacher.go').read_bytes()).hexdigest()}
    torch.save({'state_dict':state,'config':config,'gpu_pipeline':{'recurrent_reset_interval':128,'observation_storage':'f16',
        'observation_spec':1,'architecture':'existing entity/CNN/LSTM-256 policy',
        'provenance':'GPU self-play. The teacher hash pins evaluation integration; it does not claim teacher training.'}},a.output)
    print(json.dumps({'output':a.output,'parameters':sum(v.numel() for v in state.values()),'sha256':hashlib.sha256(Path(a.output).read_bytes()).hexdigest()}))
if __name__=='__main__':main()
