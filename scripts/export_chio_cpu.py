"""Export a verified Burn CHiO bundle to a CPU checkpoint; never starts a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from sf_duel.chio_cpu import load_bundle
from play_vs_chio import below_normal


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();below_normal();model,manifest=load_bundle(a.bundle)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(spec_version=5,state_dict=model.state_dict(),config=manifest),a.output)
    restored,_=load_bundle(a.output)
    equal=sum(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    result=dict(equal=equal,total=len(model.state_dict()),source=str(a.bundle),output=str(a.output),device='cpu',
                sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
    a.output.with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    if equal!=len(model.state_dict()):raise RuntimeError('CPU checkpoint roundtrip mismatch')


if __name__=='__main__':main()
