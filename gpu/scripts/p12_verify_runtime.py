"""Check both CPU model choices and restore P10 originals in an isolated copy."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from install_play_bot import inventory,restore
from play_vs_model import resolve_model
from sf_duel.model import Policy
from sf_duel.native_policy_server import policy_contract
from sf_duel.observations import warmup_batch


def main():
    run=Path('C:/AMI/runs/p12/runtime-verification')
    run.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    models=[]
    for name in ('000644','000470'):
        path=resolve_model(name)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert policy_contract(saved)==(128,'f16')
        assert saved['config']['timing_honest'] and not saved['config']['timing_oracle']
        policy=Policy(combat_inputs=True,timing_inputs=True,enemy_inputs=True).eval()
        policy.load_state_dict(saved['state_dict'],strict=True)
        with torch.inference_mode():
            policy.decide(warmup_batch(1,'cpu',True,True,True),need_entropy=False)
        models.append(dict(name=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           strict_load=True,cpu_decision=True))
    saved=json.loads(Path('C:/AMI/runs/p12/installation.json').read_text())
    target=run/'bots';shutil.copytree(saved['path'],target)
    saved.update(path=str(target),resolved=str(target.resolve()),junction=False)
    (run/'installation.json').write_text(json.dumps(saved,indent=2))
    result=restore(run)
    assert inventory(target)==saved['before'] and result['restore_verified']
    record=dict(models=models,restore=dict(scope='isolated copy; installed Steam bots left ready for play',
        original_backup=saved['backup'],equal_entries=len(saved['before']),total_entries=len(saved['before']),
        equal_files=sum(v!='directory' for v in saved['before'].values()),verified=True))
    (ROOT/'reports/P12/runtime-verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
