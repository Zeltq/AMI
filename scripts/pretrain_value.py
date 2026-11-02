"""Frozen-actor value-head pretraining with whole-game validation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from sf_duel.model import Policy
from sf_duel.selfplay.config import Config
from sf_duel.selfplay.ppo import load_dataset,warm_value_head
from train_selfplay import compact_dataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False,parents=True);cfg=Config.read(a.config);compact_dataset(a.dataset)
    meta,episodes=load_dataset(a.dataset);assert meta.get('critic_pretraining_only')
    assert hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()==meta['policy_sha256']
    train=[e for m,e in zip(meta['episodes'],episodes) if m['split']=='train'];val=[e for m,e in zip(meta['episodes'],episodes) if m['split']=='validation']
    assert {m['match'] for m in meta['episodes'] if m['split']=='train'}.isdisjoint({m['match'] for m in meta['episodes'] if m['split']=='validation'})
    torch.set_num_threads(2);torch.manual_seed(cfg.seed);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=Policy().cuda();model.load_state_dict(saved['state_dict']);start=time.monotonic()
    def progress(row):
        row={**row,'seconds':time.monotonic()-start};temp=out/'status.tmp';temp.write_text(json.dumps(row,indent=2));temp.replace(out/'status.json');print(json.dumps(row),flush=True)
    result=warm_value_head(model,train,cfg,validation_episodes=val,progress=progress);result['wall_seconds']=time.monotonic()-start
    if result['validation_passed']:
        torch.save({'state_dict':model.state_dict(),'config':{**saved['config'],'value_head_trained':True},'spec_version':saved['spec_version'],'critic_pretraining':result,'source_sha256':meta['policy_sha256']},out/'checkpoint.pt')
    (out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if not result['validation_passed']:raise RuntimeError('Critic did not beat constant baseline on held-out games')


if __name__=='__main__':main()
