"""CHiO recurrent imitation / value pretraining on recorded exact rewards."""
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.chio_model import Policy,HEADS
from sf_duel.chio_reward import read_record,SNAP
from ggd_dataset import read_block,decode
from train_ggd import recurrent,local_path,split
from p15_model_fixture import export_weights


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def returns(root,gamma,zero_sum):
    """Full episode Monte Carlo targets; no bootstrap across a terminal."""
    files=sorted(root.glob('chio-reward-*.f32.gz'),reverse=True);carry={};result={}
    for path in files:
        header,data=read_record(path)
        if header['zero_sum']!=zero_sum:raise ValueError('Reward zero_sum mismatch')
        block=int(path.name.split('-')[-1].split('.')[0])
        for row,actor in enumerate(header['actors']):
            target=np.zeros(128,np.float32);value=carry.get(actor,0.)
            for t in range(header['frames']-1,-1,-1):
                if data[row,t,-1]>.5:
                    if data[row,t,SNAP+21]>.5:value=0.
                    value=float(data[row,t,-2])+gamma*value;target[t]=value
            carry[actor]=value;result[(block,actor)]=target
    if not result:raise ValueError('Exact reward records required for critic pretraining')
    return result


def validate_dataset(manifest,cfg,mode,smoke):
    if manifest.get('family')!='CHiO' or manifest.get('observation_profile')!='chio-v1' or manifest.get('decision_ticks')!=6 or manifest.get('offset_ticks')!=6:raise ValueError('CHiO demonstration clock required')
    if manifest.get('reward_mode')!='lastorder-fb-v1' or any(abs(manifest.get(k,0)-v)>1e-6 for k,v in [('gamma',.997),('gae_lambda',.95)]):raise ValueError('Dataset reward contract mismatch')
    if manifest.get('zero_sum',manifest['config'].get('zero_sum',True))!=cfg['zero_sum']:raise ValueError('Dataset zero_sum mismatch')
    if manifest['partial'] and not smoke:raise ValueError('Partial datasets allowed only for smoke')
    if mode=='critic' and manifest['completed_matches']!=manifest['config']['games']:raise ValueError('Critic targets require complete terminal episodes')


def minibatches(sources,stage,batch,seed,targets):
    rng=random.Random(seed);files=[(i,e) for i,(_,m) in enumerate(sources) for e in m['files']];rng.shuffle(files)
    for source,entry in files:
        ids=[j for j,a in enumerate(entry['actors']) if split(a)==stage]
        if not ids:continue
        root,_=sources[source];header,obs,signals=read_block(root/entry['file'],entry['sha256']);rng.shuffle(ids)
        if header['version']!=5:raise ValueError('CHiO version-5 block required')
        block=int(entry['file'].split('-')[1])
        for begin in range(0,len(ids),batch):
            chosen=ids[begin:begin+batch];b=decode(header,obs[chosen],signals[chosen])
            target=None if targets is None else np.stack([targets[source][(block,header['actors'][j])] for j in chosen]).reshape(-1)
            yield b,signals[chosen,:,10].copy(),len(chosen),target


def run(config,output,max_updates=None,initialize=None,updates=None,resume=None,**unused):
    if unused.get('zero_previous') or unused.get('check_resume'):raise ValueError('Legacy ablation/resume-inspection flags are not CHiO modes')
    resume=resume or json.loads(Path(config).read_text()).get('resume')
    cfg=json.loads(Path(config).read_text());allowed={'dataset','replay_datasets','observation_profile','engine','decision_ticks','offset_ticks','gamma','gae_lambda','reward_mode','zero_sum','mode','epochs','updates','batch_sequences','learning_rate','gradient_norm','seed','device','initialize','resume','checkpoint_updates'}
    if set(cfg)-allowed:raise ValueError('Unknown CHiO training keys: '+str(set(cfg)-allowed))
    for key,expected in dict(observation_profile='chio-v1',decision_ticks=6,offset_ticks=6,gamma=.997,gae_lambda=.95,reward_mode='lastorder-fb-v1').items():
        if cfg.get(key)!=expected:raise ValueError('CHiO training contract: '+key)
    if not isinstance(cfg.get('zero_sum'),bool):raise ValueError('Explicit zero_sum required')
    engine=json.loads(local_path(cfg['engine']).read_text())
    if not engine.get('chio_direct') or engine['decision_ticks']!=6 or engine['offset_ticks']!=6:raise ValueError('CHiO environment required')
    mode=cfg.get('mode','imitation')
    if mode not in ('imitation','critic'):raise ValueError('Unsupported CHiO supervised mode')
    sources=[]
    for path in [cfg['dataset']]+cfg.get('replay_datasets',[]):
        root=local_path(path);manifest=json.loads((root/'dataset.json').read_text())
        validate_dataset(manifest,cfg,mode,max_updates is not None)
        sources.append((root,manifest))
    for stage in ('train','validation'):
        if not any(any(split(a)==stage for a in e['actors']) for _,m in sources for e in m['files']):raise ValueError('Separate train/validation matches required')
    device=cfg.get('device','cuda')
    if device=='cpu' and max_updates is None:raise ValueError('CPU permitted only for explicit smoke runs')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CHiO training requires ROCm/CUDA')
    if device not in ('cpu','cuda'):raise ValueError('Unsupported device')
    torch.set_num_threads(2);seed=cfg.get('seed',20261103);torch.manual_seed(seed);model=Policy().to(device)
    init=resume or initialize or cfg.get('initialize')
    if mode=='critic' and not init:raise ValueError('Critic pretraining requires imitation checkpoint')
    if init:
        checkpoint=torch.load(local_path(init),map_location='cpu',weights_only=False)
        if checkpoint['spec_version']!=5:raise ValueError('CHiO checkpoint required')
        model.load_state_dict(checkpoint['state_dict'],strict=True)
    if mode=='critic':
        for name,param in model.named_parameters():param.requires_grad_(name.startswith('value_head.'))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.get('learning_rate',1e-4),weight_decay=1e-5)
    if resume:optimizer.load_state_dict(checkpoint['optimizer'])
    budget=max_updates or updates or cfg.get('updates');epochs=cfg.get('epochs',5);batch=cfg.get('batch_sequences',8);clip=cfg.get('gradient_norm',1.)
    if min(epochs,batch,clip)<=0 or (budget is not None and budget<=0):raise ValueError('Positive training parameters required')
    targets=[returns(root,.997,cfg['zero_sum']) for root,_ in sources] if mode=='critic' else None
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and not resume:raise FileExistsError('Fresh output required')
    provenance={**cfg,'device_actual':str(device),'torch':torch.__version__,'datasets':[{'path':str(root),'sha256':hashlib.sha256((root/'dataset.json').read_bytes()).hexdigest()} for root,_ in sources],
                'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/sf_duel/chio_model.py',ROOT/'gpu/scripts/ggd_dataset.py']}}
    if resume:
        previous=checkpoint['config']
        for key in ('observation_profile','gamma','gae_lambda','zero_sum','mode','dataset','replay_datasets','batch_sequences','learning_rate','seed'):
            if previous.get(key)!=provenance.get(key):raise ValueError('Resume configuration changed: '+key)
    else:save(out/'config.json',provenance)
    started=time.monotonic();steps=checkpoint['optimizer_steps'] if resume else 0;history=checkpoint['history'] if resume else []
    start_epoch=checkpoint.get('epoch',1) if resume else 1;resume_batch=checkpoint.get('batch_index',-1) if resume else -1
    elapsed=history[-1]['wall_seconds'] if history else 0.
    if resume and 'rng' in checkpoint:
        torch.set_rng_state(checkpoint['rng'])
        if device=='cuda' and checkpoint.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    def checkpoint_now(epoch,batch_index):
        state={'spec_version':5,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),
               'optimizer_steps':steps,'config':provenance,'history':history,'epoch':epoch,'batch_index':batch_index,'rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None}
        temporary=out/'checkpoint.pt.tmp';torch.save(state,temporary);temporary.replace(out/'checkpoint.pt')

    initial_value={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith('value_head.')}
    def compute(b,resets,n,target):
        labels=torch.as_tensor(b.pop('labels'),device=device);valid=torch.as_tensor(b.pop('loss_valid'),device=device)
        b={k:torch.as_tensor(v,device=device) for k,v in b.items()}
        if not valid.any():return None,0
        rows=torch.arange(len(labels),device=device)
        if not b['action_mask'][rows[valid],labels[valid,0]].all():raise ValueError('Label outside action mask')
        for kind,col,name in [(2,2,'enemy_target_mask'),(3,3,'ally_target_mask')]:
            use=valid&(labels[:,0]==kind)
            if not b[name][rows[use],labels[use,col]].all():raise ValueError('Label outside target mask')
        if mode=='critic':
            with torch.no_grad():h,keys=recurrent(model,b,n,resets)
            prediction=model.value_head(h).squeeze(-1);target=torch.as_tensor(target,device=device);loss=(prediction[valid]-target[valid]).square().mean()
        else:
            h,keys=recurrent(model,b,n,resets);_,logp,_,_,_=model.heads(b,h,keys,actions={name:labels[:,i] for i,name in enumerate(HEADS)})
            loss=-logp[valid].mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        return loss,int(valid.sum())
    # An explicit update budget is independent of dataset size/epoch count.
    epoch=start_epoch-1
    while epoch<epochs or (budget is not None and steps<budget):
        epoch+=1;model.train();epoch_steps=0
        skipped=0
        for batch_index,(b,resets,n,target) in enumerate(minibatches(sources,'train',batch,seed+epoch,targets)):
            if epoch==start_epoch and batch_index<=resume_batch:skipped+=1;continue
            if budget is not None and steps>=budget:break
            optimizer.zero_grad(set_to_none=True);loss,count=compute(b,resets,n,target)
            if loss is None:continue
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True);optimizer.step();steps+=1;epoch_steps+=1
            row=dict(step=steps,epoch=epoch,loss=float(loss.detach()),gradient_norm=float(norm),frames=count,wall_seconds=elapsed+time.monotonic()-started);history.append(row)
            save(out/'status.json',row);print(json.dumps(row),flush=True)
            if steps%cfg.get('checkpoint_updates',250)==0:checkpoint_now(epoch,batch_index)
            if budget is not None and steps>=budget:break
        if epoch_steps:checkpoint_now(epoch,batch_index)
        if not epoch_steps and not skipped and not (budget is not None and steps>=budget):raise ValueError('No eligible training sequences')
        if budget is not None and steps>=budget:break
    model.eval();validation_sum=0.;validation_frames=0
    with torch.no_grad():
        for b,resets,n,target in minibatches(sources,'validation',batch,seed,targets):
            loss,count=compute(b,resets,n,target)
            if loss is not None:validation_sum+=float(loss)*count;validation_frames+=count
    if not validation_frames:raise ValueError('Empty validation split')
    if mode=='imitation' and any(not torch.equal(initial_value[k],v.cpu()) for k,v in model.state_dict().items() if k in initial_value):raise AssertionError('Imitation changed unused value head')
    export_weights(model.cpu(),out/'weights')
    result=dict(passed=True,mode=mode,updates=steps,finite_updates=len(history),validation_loss=validation_sum/validation_frames,validation_frames=validation_frames,
                wall_seconds=elapsed+time.monotonic()-started,checkpoint=str(out/'checkpoint.pt'),weights=str(out/'weights'),parameters=sum(p.numel() for p in model.parameters()))
    save(out/'history.json',history);save(out/'result.json',result);print(json.dumps(result),flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');p.add_argument('output');p.add_argument('--max-updates',type=int);p.add_argument('--initialize');p.add_argument('--updates',type=int);p.add_argument('--resume')
    run(**vars(p.parse_args()))


if __name__=='__main__':main()
