"""Validate a PPO dataset by default. Updating weights requires --train."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import torch
from sf_duel.model import Policy
from sf_duel.selfplay.config import Config
from sf_duel.selfplay.ppo import load_dataset, verify_behavior, optimize, value_metrics, warm_value_head


def compact_dataset(path):
    """Compress closed .npy arrays on NTFS before opening memory maps in WSL."""
    root=Path(path).resolve()
    if os.name=='nt':executable='compact.exe';native=str(root)
    elif str(root).startswith('/mnt/') and Path('/mnt/c/Windows/System32/compact.exe').is_file():
        executable='/mnt/c/Windows/System32/compact.exe';native=root.parts[2].upper()+':\\'+'\\'.join(root.parts[3:])
    else:return {'supported':False}
    started=time.monotonic()
    result=subprocess.run([executable,'/C','/I','/Q','/S:'+native,'*.npy'],capture_output=True)
    report={'supported':True,'returncode':result.returncode,'wall_seconds':time.monotonic()-started,'lossless':True}
    (root/'ntfs-storage.json').write_text(json.dumps(report,indent=2));print(json.dumps({'dataset_storage':report}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--train',action='store_true')
    p.add_argument('--skip-value-warmup',action='store_true');p.add_argument('--device',default='cuda')
    p.add_argument('--value-initializer');p.add_argument('--critic-gate',action='store_true');a=p.parse_args()
    holds=[parent/'training-hold.json' for parent in Path(a.dataset).resolve().parents]
    active=[p for p in holds if a.train and p.exists()]
    if active:print(json.dumps({'phase':'waiting_for_transport_verification','holds':[str(p) for p in active]}),flush=True)
    while any(p.exists() for p in active):time.sleep(2)
    cfg=Config.read(a.config);compact_dataset(a.dataset);meta,episodes=load_dataset(a.dataset)
    digest=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    if meta['policy_sha256']!=digest:raise ValueError('Dataset was collected by a different policy')
    torch.set_num_threads(2);torch.manual_seed(cfg.seed)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=Policy().to(a.device)
    if a.value_initializer:
        value_saved=torch.load(a.value_initializer,map_location='cpu',weights_only=False)
        if not value_saved['critic_pretraining']['validation_passed']:raise ValueError('Unvalidated value initializer')
        for key,value in saved['state_dict'].items():
            if not key.startswith('value_head.'):torch.testing.assert_close(value,value_saved['state_dict'][key],rtol=0,atol=0)
        saved['state_dict']={k:value_saved['state_dict'][k] if k.startswith('value_head.') else v for k,v in saved['state_dict'].items()}
        saved['config']['value_head_trained']=True;saved['critic_pretraining']=value_saved['critic_pretraining']
    if a.skip_value_warmup and not saved['config'].get('value_head_trained',False):
        raise ValueError('Cannot skip critic warmup for an untrained value head')
    model.load_state_dict(saved['state_dict'],strict=True);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    if not a.train:
        start=time.monotonic();result=verify_behavior(model,episodes,cfg);result['weights_updated']=False
        result['wall_seconds']=time.monotonic()-start
        result['frames']=sum(len(e['labels']) for e in episodes)
        (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
    start=time.monotonic()
    def progress(value):
        temp=out/'status.tmp';temp.write_text(json.dumps({**value,'minutes':(time.monotonic()-start)/60}));temp.replace(out/'status.json')
        print(json.dumps(value),flush=True)
    critic_check=None;fresh_warmup=None
    if a.critic_gate:
        progress({'phase':'checking_critic_on_fresh_games'})
        constant=saved.get('critic_pretraining',{}).get('training_target_mean',0.)
        critic_check=value_metrics(model,episodes,cfg,constant)
        progress({'phase':'critic_fresh_check','metrics':critic_check})
        if not critic_check['passed']:
            matches=sorted({e['match'] for e in meta['episodes']});validation=set(matches[::4])
            train=[e for m,e in zip(meta['episodes'],episodes) if m['match'] not in validation]
            val=[e for m,e in zip(meta['episodes'],episodes) if m['match'] in validation]
            fresh_warmup=warm_value_head(model,train,cfg,validation_episodes=val,progress=progress)
            if not fresh_warmup['validation_passed']:
                (out/'critic-gate-failed.json').write_text(json.dumps({'initial':critic_check,'warmup':fresh_warmup},indent=2))
                raise RuntimeError('Critic failed fresh-game holdout; actor PPO blocked')
    prior_warmup=saved.get('result',{}).get('value_warmup') or {}
    repair=bool(prior_warmup and prior_warmup.get('value_mse_after',0)>prior_warmup.get('value_mse_before',float('inf')))
    if repair:progress({'phase':'critic_warmup_repair_scheduled','reason':'Previous episode-ordered warmup increased MSE; use globally mixed frozen features with unchanged LR and epochs'})
    optimizer_state=saved.get('optimizer') if saved.get('optimizer_kind')=='ppo' else None
    result,optimizer=optimize(model,episodes,cfg,warmup=not a.skip_value_warmup or repair,progress=progress,optimizer_state=optimizer_state)
    result['critic_fresh_check']=critic_check;result['fresh_value_warmup']=fresh_warmup
    result['matches_used']=len({meta['episodes'][i]['match'] for i in result['episodes_used']})
    result['critic_warmup_repair']=repair
    result['wall_seconds']=time.monotonic()-start
    if result['publishable']:
        torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'optimizer_kind':'ppo','spec_version':1,
                    'config':{**saved['config'],'value_head_trained':True,'selfplay':asdict(cfg)},
                    'behavior_sha256':digest,'result':result},out/'checkpoint.pt')
    (out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if not result['publishable']:raise RuntimeError('Full-dataset KL exceeded limit; checkpoint not published')


if __name__=='__main__':main()
