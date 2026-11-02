"""Complete-episode recurrent PPO. No training starts on import."""
import copy
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from ..model import Policy
from ..bc_data import HEADS, sequence_batch
from .rollouts import EXTRA_KEYS
from .rewards import gae


def model_episode(episode):
    return {k:v for k,v in episode.items() if k not in EXTRA_KEYS}


def tensors(arrays, device):
    return {k:torch.as_tensor(np.array(v,copy=True),device=device) for k,v in arrays.items()}


def starts(episode, cfg):
    return [(learn,max(0,learn-cfg.burn_in)) for learn in range(0,len(episode['labels']),cfg.bptt)]


@torch.no_grad()
def refresh(model, episode, cfg, latents=False):
    """Fresh full history, no zero-state restart at internal chunk boundaries."""
    model.eval();device=next(model.parameters()).device;state=model.initial_state(1)
    boundaries=sorted({0,len(episode['labels']),*[context for _,context in starts(episode,cfg)]})
    result={'states':{},'logp':[],'value':[],'latent':[]}
    for start,end in zip(boundaries,boundaries[1:]):
        result['states'][start]=tuple(v.detach().cpu().clone() for v in state)
        b=tensors({k:v[start:end] for k,v in model_episode(episode).items() if k not in ('labels','loss_valid')},device)
        actions={k:torch.as_tensor(np.array(episode['labels'][start:end,i]),device=device) for i,k in enumerate(HEADS)}
        fused,keys=model.encode_entities(b);output,state=model.memory(fused.unsqueeze(0),state);hidden=output[0]
        _,logp,_,value,_=model.heads(b,hidden,keys,model.value_head(hidden).squeeze(-1),state,actions=actions)
        result['logp'].append(logp.cpu());result['value'].append(value.cpu())
        if latents:result['latent'].append(hidden.cpu())
    for key in ('logp','value'):result[key]=torch.cat(result[key]).numpy()
    if latents:result['latent']=torch.cat(result['latent'])
    return result


def verify_behavior(model, episodes, cfg, tolerance=5e-4):
    worst=0.
    for episode in episodes:
        replay=refresh(model,episode,cfg)
        error=float(np.max(np.abs(replay['logp']-episode['old_log_prob'])))
        if not np.isfinite(error) or error>tolerance:raise ValueError(f'Behavior log-prob replay differs: {error}')
        for start,state in replay['states'].items():
            for key,current in zip(('h','c'),state):
                np.testing.assert_allclose(current.numpy()[0,0],episode[key][start],atol=5e-4,rtol=5e-4)
        worst=max(worst,error)
    return {'max_log_probability_error':worst,'ratio_max_error_bound':float(np.expm1(worst))}


def warm_value_head(model, episodes, cfg, validation_episodes=None, progress=None):
    frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('value_head.')}
    independent=validation_episodes is not None
    validation_episodes=episodes if validation_episodes is None else validation_episodes
    all_episodes=list(episodes)+list(validation_episodes)
    cached=[refresh(model,e,cfg,latents=True) for e in all_episodes]
    before=[r['logp'] for r in cached]
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('value_head.'))
    optimizer=torch.optim.Adam(model.value_head.parameters(),lr=cfg.value_warmup_lr)
    device=next(model.parameters()).device;steps=0
    targets=[gae(e['rewards'],np.zeros(len(e['labels'])),e['terminals'],gamma=cfg.gamma,lam=1)[1] for e in all_episodes]
    def extract(es,rs,ys):
        return (torch.cat([r['latent'][np.flatnonzero(e['loss_valid'])] for e,r in zip(es,rs)]),
                torch.as_tensor(np.concatenate([v[e['loss_valid']] for e,v in zip(es,ys)])))
    n=len(episodes);features,target=extract(episodes,cached[:n],targets[:n])
    validation_features,validation_target=extract(validation_episodes,cached[n:],targets[n:])
    cached.clear()
    def mse(features=features,target=target):
        total=0.;count=0
        with torch.no_grad():
            for start in range(0,len(target),2048):
                predicted=model.value_head(features[start:start+2048].to(device)).squeeze(-1)
                total+=float((predicted-target[start:start+2048].to(device)).square().sum());count+=len(predicted)
        return total/max(1,count)
    before_mse=mse();before_validation=mse(validation_features,validation_target)
    best_mse=before_validation;best=copy.deepcopy(model.value_head.state_dict());history=[];best_epoch=0;stale=0
    try:
        for epoch in range(cfg.value_warmup_epochs):
            # Frozen recurrent features may be shuffled globally: a minibatch
            # must not contain only the winner/loser label of a single episode.
            for ids in torch.randperm(len(target)).split(2048):
                x=features[ids].to(device);y=target[ids].to(device)
                loss=(model.value_head(x).squeeze(-1)-y).square().mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite critic warmup loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.value_head.parameters(),cfg.gradient_norm);optimizer.step();steps+=1
            score=mse(validation_features,validation_target);row={'epoch':epoch+1,'value_mse':mse(),'validation_mse':score};history.append(row)
            if progress:progress({'phase':'critic_pretraining',**row})
            if score<best_mse:
                best_mse=score;best=copy.deepcopy(model.value_head.state_dict());best_epoch=epoch+1;stale=0
            else:stale+=1
            if independent and stale>=3:break
        model.value_head.load_state_dict(best)
        for key,value in frozen.items():torch.testing.assert_close(model.state_dict()[key].cpu(),value,rtol=0,atol=0)
        for old,episode in zip(before,all_episodes):np.testing.assert_allclose(refresh(model,episode,cfg)['logp'],old,rtol=0,atol=1e-6)
    finally:
        for p in model.parameters():p.requires_grad_(True)
    constant=float(target.mean());baseline=float((validation_target-constant).square().mean())
    return {'steps':steps,'actor_unchanged':True,'value_mse_before':before_mse,'value_mse_after':mse(),'history':history,'shuffled_across_episodes':True,
            'validation_mse_before':before_validation,'validation_mse_after':best_mse,'validation_constant_mse':baseline,'best_epoch':best_epoch,
            'validation_is_independent':independent,'validation_passed':independent and best_mse<baseline,'training_target_mean':constant}


@torch.no_grad()
def value_metrics(model, episodes, cfg, constant=0.):
    predictions=[];targets=[]
    for e in episodes:
        predictions.append(refresh(model,e,cfg)['value'][e['loss_valid']])
        targets.append(gae(e['rewards'],np.zeros(len(e['labels'])),e['terminals'],gamma=1,lam=1)[1][e['loss_valid']])
    y=np.concatenate(targets).astype(np.float64);v=np.concatenate(predictions).astype(np.float64)
    mse=float(np.mean((v-y)**2));baseline=float(np.mean((y-constant)**2));variance=float(np.var(y))
    return {'mse':mse,'constant_mse':baseline,'explained_variance':None if variance<1e-12 else 1-float(np.var(y-v))/variance,'count':len(y),'passed':bool(np.isfinite(mse) and mse<baseline)}


def batch_segments(episodes, selection, references, advantages, returns, cfg, device):
    windows=[(i,context) for i,learn,context in selection]
    length=cfg.burn_in+cfg.bptt
    arrays,y,valid=sequence_batch([model_episode(e) for e in episodes],windows,length,0)
    masks=[];old=[];adv=[];ret=[];states=[]
    for i,learn,context in selection:
        e=episodes[i];positions=np.arange(context,context+length);ids=np.minimum(positions,len(e['labels'])-1)
        masks.append((positions>=learn)&(positions<learn+cfg.bptt)&(positions<len(e['labels'])))
        old.append(e['old_log_prob'][ids]);adv.append(advantages[i][ids]);ret.append(returns[i][ids]);states.append(references[i]['states'][context])
    mask=torch.as_tensor(np.concatenate(masks)&valid,device=device)
    state=tuple(torch.cat([pair[k] for pair in states],1).to(device) for k in range(2))
    return tensors(arrays,device),{k:torch.as_tensor(y[:,i],device=device) for i,k in enumerate(HEADS)},mask,state,*[torch.as_tensor(np.concatenate(v),device=device) for v in (old,adv,ret)]


def normalized_advantages(raw,valid=None):
    valid=[np.ones(len(a),bool) for a in raw] if valid is None else valid
    whole=np.concatenate([a[m] for a,m in zip(raw,valid)]).astype(np.float64)
    if not len(whole):raise ValueError('No valid advantages')
    mean=whole.mean();std=max(whole.std(),1e-8)
    return [np.where(m,(a-mean)/std,0).astype(np.float32) for a,m in zip(raw,valid)]


@torch.no_grad()
def measured_kl(model, episodes, cfg):
    total=0.;count=0
    for episode in episodes:
        log_ratio=refresh(model,episode,cfg)['logp'].astype(np.float64)-episode['old_log_prob']
        term=np.expm1(log_ratio)-log_ratio
        if not np.isfinite(term).all():return float('inf')
        total+=term.sum();count+=len(term)
    return float(total/max(1,count))


@torch.no_grad()
def recurrent_drift(model, episode, reference, cfg):
    fresh=refresh(model,episode,cfg)
    return max(float((old-new).abs().max()) for key,pair in reference['states'].items()
               for old,new in zip(pair,fresh['states'][key]))


@torch.no_grad()
def states_at(model, episode, points):
    """Current-weight memory from the real episode start, without policy heads."""
    model.eval();device=next(model.parameters()).device;state=model.initial_state(1);result={};cursor=0
    arrays=model_episode(episode)
    for point in sorted(set(points)):
        while cursor<point:
            end=min(point,cursor+128)
            b=tensors({k:v[cursor:end] for k,v in arrays.items() if k not in ('labels','loss_valid')},device)
            fused,_=model.encode_entities(b);_,state=model.memory(fused.unsqueeze(0),state);cursor=end
        result[point]=tuple(v.detach().cpu().clone() for v in state)
    return result


def optimize(model, episodes, cfg, warmup=True, progress=None, optimizer_state=None):
    """Recurrent PPO, with persisted Adam and coverage of committed updates."""
    cfg.validate()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if progress:progress({'phase':'verifying_behavior'})
    verification=verify_behavior(model,episodes,cfg)
    if progress:progress({'phase':'warming_value_head' if warmup else 'preparing_advantages','verification':verification})
    warming=warm_value_head(model,episodes,cfg) if warmup else None
    advantages=[];returns=[]
    for episode in episodes:
        baseline=refresh(model,episode,cfg)['value']
        a,g=gae(episode['rewards'],baseline,episode['terminals'],gamma=cfg.gamma,lam=cfg.gae_lambda)
        advantages.append(a);returns.append(g)
    advantages=normalized_advantages(advantages,[e['loss_valid'] for e in episodes])
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        for group in optimizer.param_groups:group['lr']=cfg.learning_rate
    optimizer_before=max((int(v['step']) for v in optimizer.state.values() if 'step' in v),default=0)
    rng=random.Random(cfg.seed);device=next(model.parameters()).device
    control=[episodes[i] for i in sorted(rng.sample(range(len(episodes)),min(4,len(episodes))))]
    safe_model=copy.deepcopy(model.state_dict());safe_optimizer=copy.deepcopy(optimizer.state_dict())
    seen=[np.zeros(len(e['labels']),bool) for e in episodes];safe_seen=[v.copy() for v in seen]
    steps=0;safe_steps=0;used=0;safe_used=0;history=[];stopped=False;first_epoch_batches=0
    valid_total=sum(int(e['loss_valid'].sum()) for e in episodes)
    if progress:progress({'phase':'ppo','value_warmup':warming,'optimizer_steps_before':optimizer_before})
    for epoch in range(cfg.ppo_epochs):
        references=None if cfg.fresh_memory else [refresh(model,e,cfg) for e in episodes]
        selections=[(i,learn,context) for i,e in enumerate(episodes) for learn,context in starts(e,cfg) if e['loss_valid'][learn:learn+cfg.bptt].any()]
        batches=[];effective=cfg.batch_sequences*cfg.gradient_accumulation
        for is_first in (True,False):
            group=[s for s in selections if (s[1]==0)==is_first];rng.shuffle(group)
            batches.extend(group[i:i+effective] for i in range(0,len(group),effective))
        rng.shuffle(batches)
        if epoch==0:first_epoch_batches=len(batches)
        for batch_index,selection in enumerate(batches):
            if cfg.fresh_memory:
                references=[{'states':{}} for _ in episodes]
                for i in {s[0] for s in selection}:
                    references[i]['states']=states_at(model,episodes[i],[c for j,l,c in selection if j==i])
            total=sum(int(episodes[i]['loss_valid'][l:l+cfg.bptt].sum()) for i,l,c in selection)
            optimizer.zero_grad(set_to_none=True);metrics_total={'policy_loss':0.,'value_loss':0.,'normalized_entropy':0.}
            for micro_start in range(0,len(selection),cfg.batch_sequences):
                micro=selection[micro_start:micro_start+cfg.batch_sequences]
                b,labels,mask,state,old,adv,ret=batch_segments(episodes,micro,references,advantages,returns,cfg,device)
                n=len(micro);length=cfg.burn_in+cfg.bptt;burn=micro[0][1]-micro[0][2]
                def window(value,start,end):return value.reshape(n,length,*value.shape[1:])[:,start:end].reshape(n*(end-start),*value.shape[1:])
                if burn:
                    with torch.no_grad():state=model.sequence_decide({k:window(v,0,burn) for k,v in b.items()},n,burn,state,actions={k:window(v,0,burn) for k,v in labels.items()})[-1]
                b={k:window(v,burn,burn+cfg.bptt) for k,v in b.items()};labels={k:window(v,burn,burn+cfg.bptt) for k,v in labels.items()}
                mask,old,adv,ret=[window(v,burn,burn+cfg.bptt) for v in (mask,old,adv,ret)]
                model.train();_,logp,_,value,_,metrics=model.sequence_decide(b,n,cfg.bptt,state,actions=labels,return_metrics=True)
                ratio=torch.exp(logp-old);clipped=ratio.clamp(1-cfg.clip,1+cfg.clip)
                actor=-torch.minimum(ratio*adv,clipped*adv)[mask].mean()
                critic=(value[mask]-ret[mask]).square().mean();entropy=metrics['normalized_entropy'][mask].mean()
                weight=int(mask.sum())/total;loss=(actor+cfg.value_coefficient*critic-cfg.entropy_coefficient*entropy)*weight
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
                loss.backward()
                for key,v in [('policy_loss',actor),('value_loss',critic),('normalized_entropy',entropy)]:metrics_total[key]+=float(v.detach())*weight
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.gradient_norm,error_if_nonfinite=True);optimizer.step();steps+=1;used+=total
            for i,learn,context in selection:seen[i][learn:learn+cfg.bptt]|=episodes[i]['loss_valid'][learn:learn+cfg.bptt]
            if steps%cfg.kl_check_batches==0 or batch_index+1==len(batches):
                kl=measured_kl(model,control,cfg)
                row={'epoch':epoch+1,'steps':steps,'kl':kl,**metrics_total,'gradient_norm_before_clip':float(norm),
                     'unique_frame_coverage':sum(int(v.sum()) for v in seen)/valid_total,'effective_batch_sequences':effective}
                history.append(row)
                if progress:progress(row)
                if kl>cfg.stop_kl:
                    model.load_state_dict(safe_model);optimizer.load_state_dict(safe_optimizer);stopped=True;break
                safe_model=copy.deepcopy(model.state_dict());safe_optimizer=copy.deepcopy(optimizer.state_dict());safe_steps=steps;safe_used=used;safe_seen=[v.copy() for v in seen]
        if stopped:break
    final_kl=measured_kl(model,episodes,cfg)
    delta_sq=base_sq=actor_delta_sq=value_delta_sq=0.;maximum=0.
    for key,value in model.state_dict().items():
        delta=value.detach().cpu()-initial[key];delta_sq+=float(delta.square().sum());base_sq+=float(initial[key].square().sum());maximum=max(maximum,float(delta.abs().max()))
        if key.startswith('value_head.'):value_delta_sq+=float(delta.square().sum())
        else:actor_delta_sq+=float(delta.square().sum())
    unique=sum(int(v.sum()) for v in safe_seen)
    return {'verification':verification,'value_warmup':warming,'steps_attempted':steps,'steps_retained':safe_steps,'optimizer_steps_before':optimizer_before,
            'optimizer_steps_after':max((int(v['step']) for v in optimizer.state.values() if 'step' in v),default=0),
            'valid_frames':valid_total,'unique_frames_retained':unique,'frame_coverage':unique/valid_total,'frames_attempted':used,'frames_retained_with_reuse':safe_used,
            'episodes_used':[i for i,v in enumerate(safe_seen) if v.any()],'batches_in_first_epoch':first_epoch_batches,
            'early_stop':stopped,'history':history,'final_kl':final_kl,'publishable':final_kl<=cfg.stop_kl,
            'parameter_delta_l2':delta_sq**.5,'parameter_relative_delta_l2':(delta_sq/max(base_sq,1e-20))**.5,'parameter_max_delta':maximum,
            'actor_parameter_delta_l2':actor_delta_sq**.5,'value_parameter_delta_l2':value_delta_sq**.5},optimizer


def load_dataset(path):
    root=Path(path);meta=json.loads((root/'dataset.json').read_text());episodes=[]
    if not meta.get('complete_episodes') or meta['gamma']!=1:raise ValueError('Unapproved/incomplete dataset')
    for entry in meta['episodes']:
        folder=root/entry['path'];episodes.append({p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in folder.glob('*.npy')})
    if not episodes:raise ValueError('No on-policy learner episodes')
    return meta,episodes
