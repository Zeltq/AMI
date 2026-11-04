"""Freeze the selected policy, cache recurrent features, fit only its value head."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from train_ggd import ROOT,local_path,recurrent
from sf_duel.model import Policy
from sf_duel.observations import packed_tensors
from sf_duel.timing_observations import ablate
from ggd_dataset import read_block,decode
from dagger_data import eligible
from p09b_reward import targets

def save(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def ev_stats(target,prediction):
    target=np.asarray(target,np.float64);prediction=np.asarray(prediction,np.float64);error=target-prediction
    return dict(n=len(target),y=float(target.sum()),yy=float((target*target).sum()),e=float(error.sum()),ee=float((error*error).sum()))
def summarize(rows):
    r={k:sum(x[k] for x in rows) for k in ('n','y','yy','e','ee')};n=max(1,r['n']);var=r['yy']/n-(r['y']/n)**2
    return {**r,'mse':r['ee']/n,'ev':1-(r['ee']/n-(r['e']/n)**2)/var if r['n']>1 and var>1e-10 else None}
def actor_hash(model):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for k,v in model.state_dict().items() if not k.startswith('value_head.'))).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('output');p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--epochs',type=int,default=8);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(20261105);rng=np.random.default_rng(20261105)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    saved=torch.load(local_path(a.checkpoint),map_location='cpu',weights_only=False)
    assert saved['spec_version']==4
    policy=Policy(combat_inputs=True,timing_inputs=True,enemy_inputs=True).cuda();policy.load_state_dict(saved['state_dict']);policy.eval()
    for param in policy.parameters():param.requires_grad_(False)
    before=actor_hash(policy);config=dict(checkpoint=str(a.checkpoint),datasets=a.datasets,epochs=a.epochs,lr=3e-4,batch_frames=8192,seed=20261105,gamma=.99,mana_weight=.5,actor_sha256=before)
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    save(out/'config.json',config)
    cache=[];frames=0
    for dataset_index,dataset in enumerate(a.datasets):
        root=local_path(dataset);target_dir=out/f'targets-{dataset_index}'
        if not (target_dir/'targets.json').exists():targets(root,target_dir)
        target_manifest=json.loads((target_dir/'targets.json').read_text());lookup={x['demonstration']:x for x in target_manifest['files']}
        manifest=json.loads((root/'dataset.json').read_text());folder=out/f'features-{dataset_index}';folder.mkdir(exist_ok=True)
        for block,entry in enumerate(manifest['files']):
            cache_path=folder/(Path(entry['file']).name+'.npz');cache.append(cache_path)
            if cache_path.exists():continue
            h,o,s=read_block(root/entry['file'],entry['sha256'])
            source=h.get('source','teacher-demo')
            ids=[i for i,actor in enumerate(h['actors']) if eligible(actor,'train',source) or eligible(actor,'validation',source)]
            if not ids:
                cache.pop();continue
            target_path=target_dir/lookup[entry['file']]['file']
            with np.load(target_path) as z:
                all_returns=z['returns'];all_post=z['post0'];assert z['actors'].tolist()==h['actors']
            feats=[];returns=[];post=[];validation=[]
            for at in range(0,len(ids),16):
                ix=ids[at:at+16];arrays=decode(h,o[ix],s[ix]);valid=arrays.pop('loss_valid').astype(bool);arrays.pop('labels')
                if not valid.any():continue
                arrays=ablate(arrays,True,False)
                with torch.no_grad():
                    hidden,_=recurrent(policy,packed_tensors(arrays,'cuda'),len(ix),s[ix,:,10])
                    feats.append(hidden.cpu().numpy()[valid].astype(np.float16))
                returns.append(all_returns[ix].reshape(-1)[valid]);post.append(all_post[ix].reshape(-1)[valid])
                validation.append(np.repeat([h['actors'][i]//2%8==6 for i in ix],128)[valid])
            if not feats:cache.pop();continue
            np.savez(cache_path,h=np.concatenate(feats),y=np.concatenate(returns),post=np.concatenate(post),validation=np.concatenate(validation))
            frames+=sum(len(x) for x in returns)
            save(out/'status.json',dict(stage='frozen feature cache',dataset=dataset_index,block=block+1,blocks=len(manifest['files']),new_frames=frames,minutes=(time.monotonic()-start)/60))
    for param in policy.value_head.parameters():param.requires_grad_(True)
    optimizer=torch.optim.Adam(policy.value_head.parameters(),lr=3e-4)
    history=[];best=float('inf');first_epoch=0
    if (out/'last.pt').exists():
        last=torch.load(out/'last.pt',map_location='cpu',weights_only=False);policy.load_state_dict(last['state_dict']);optimizer.load_state_dict(last['critic_optimizer']);history=last['critic_history'];first_epoch=last['critic_epoch'];best=min(r['validation']['all']['mse'] for r in history)
    for epoch in range(first_epoch,a.epochs):
        order=list(cache);rng.shuffle(order);train_n=0;train_loss=0.
        for path in order:
            with np.load(path) as z:h=z['h'];y=z['y'];ids=np.flatnonzero(~z['validation'])
            rng.shuffle(ids)
            for at in range(0,len(ids),8192):
                ix=ids[at:at+8192];x=torch.as_tensor(h[ix].astype(np.float32),device='cuda');target=torch.as_tensor(y[ix],device='cuda')
                pred=policy.value_head(x).squeeze(-1);loss=(pred-target).square().mean()
                assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(policy.value_head.parameters(),.5);optimizer.step()
                train_n+=len(ix);train_loss+=float(loss.detach())*len(ix)
        stats={k:[] for k in ('all','pregame','post0')}
        for path in cache:
            with np.load(path) as z:h=z['h'];y=z['y'];post=z['post'];ids=np.flatnonzero(z['validation'])
            for at in range(0,len(ids),8192):
                ix=ids[at:at+8192]
                with torch.no_grad():pred=policy.value_head(torch.as_tensor(h[ix].astype(np.float32),device='cuda')).squeeze(-1).cpu().numpy()
                for name,mask in [('all',np.ones(len(ix),bool)),('pregame',~post[ix]),('post0',post[ix])]:stats[name].append(ev_stats(y[ix][mask],pred[mask]))
        row=dict(epoch=epoch+1,train_frames=train_n,train_mse=train_loss/max(1,train_n),validation={k:summarize(v) for k,v in stats.items()});history.append(row)
        checkpoint={**saved,'state_dict':policy.state_dict(),'config':{**saved['config'],'model_name':saved['config']['model_name']+'-c'},'critic_optimizer':optimizer.state_dict(),'critic_history':history,'critic_epoch':epoch+1}
        torch.save(checkpoint,out/'last.pt')
        if row['validation']['all']['mse']<best:best=row['validation']['all']['mse'];torch.save(checkpoint,out/'best.pt')
        save(out/'status.json',dict(stage='value fitting',**row,minutes=(time.monotonic()-start)/60));print(json.dumps(row),flush=True)
    assert actor_hash(policy)==before
    result=dict(history=history,actor_sha256_before=before,actor_sha256_after=actor_hash(policy),actor_unchanged=True,selection='minimum validation MSE, all decisions; phase EV reported separately',checkpoint=str(out/'best.pt'),minutes=(time.monotonic()-start)/60)
    save(out/'result.json',result)

if __name__=='__main__':main()
