"""Recurrent GGD imitation on ROCm/CUDA using the existing Policy architecture."""
import argparse,csv,gc,hashlib,json,math,random,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.model import Policy
from sf_duel.observations import packed_tensors
from ggd_dataset import read_block,decode
from weights import rust_name,write_bundle

def local_path(path):
    text=str(path).replace('\\','/')
    return Path('/mnt/'+text[0].lower()+text[2:]) if sys.platform!='win32' and len(text)>1 and text[1]==':' else Path(text)

def split(actor):return 'validation' if actor//2%8==6 else 'test' if actor//2%8==7 else 'train'

def load_resume(path,config,dataset_hash):
    saved=torch.load(resolved(path),map_location='cpu',weights_only=False)
    if saved.get('spec_version')!=2 or saved['config'].get('observation_profile')!='ggd-v2':raise ValueError('Not a GGD checkpoint')
    if saved['config'].get('dataset_sha256')!=dataset_hash:raise ValueError('Resume dataset differs')
    for key in ('seed','batch_sequences','learning_rate','gradient_norm','attack_weight'):
        if saved['config'].get(key)!=config.get(key):raise ValueError('Resume configuration differs: '+key)
    epoch=saved.get('epoch',0);history=saved.get('history',[])
    if len(history)!=epoch or epoch<1 or not history[-1]['validation']['frames'] or not history[-1]['train']['frames']:raise ValueError('Resume requires a completed IL epoch')
    if 'optimizer' not in saved or 'optimizer_steps' not in saved:raise ValueError('Optimizer checkpoint missing')
    if epoch>=config.get('epochs',5):raise ValueError('Checkpoint already reached the requested final epoch')
    model=Policy(combat_inputs=True);model.load_state_dict(saved['state_dict'],strict=True)
    return saved

def resolved(path):
    p=local_path(path);return p if p.is_absolute() else ROOT/p

def native_path(path,executable):
    p=str(path)
    if str(executable).lower().endswith('.exe') and sys.platform!='win32' and p.startswith('/mnt/'):
        return p[5].upper()+':/'+p[7:]
    return p

def evaluate_epoch(model,checkpoint,out,epoch,config,metadata):
    directory=out/f'epoch-{epoch:03d}';directory.mkdir(exist_ok=True)
    tensors={}
    for name,value in model.state_dict().items():
        value=value.detach().float().cpu().numpy()
        if value.ndim==2 and not name.endswith('_embed.weight'):value=value.T
        tensors[rust_name(name)]=value
    write_bundle(directory/'weights',tensors,family='GGD',model_name=metadata['model_name'],epoch=epoch,observation_profile='ggd-v2',teacher_sha256=metadata['teacher_policy_sha256'],source=str(checkpoint),source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),recurrent_reset=128,provenance='Fresh GGD imitation in the GPU simulator')
    executable=resolved(config['executable']);engine=resolved(config['engine']);results={}
    optimizer_memory='model/optimizer retained; inactive GPU allocations released before native evaluation'
    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
    for mode in config.get('modes',['teacher','mirror']):
        target=directory/f'vs-{mode}';args=[str(executable),'evaluate-ggd',native_path(engine,executable),native_path(directory/'weights',executable),native_path(target,executable),'--matches',str(config.get('matches',32)),'--seconds',str(config.get('seconds',600)),'--mode',mode,'--seed',str(config.get('seed',20261103))]
        status={'stage':'gameplay_evaluation','epoch':epoch,'mode':mode,'command':args,'report':str(target/'report.json'),'gpu_memory':optimizer_memory}
        (out/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
        with (directory/f'vs-{mode}.log').open('w') as log:subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((target/'report.json').read_text())
        if not report['complete']:raise RuntimeError('Gameplay evaluation interrupted before its configured horizon')
        results[mode]={'report':str(target/'report.json'),'summary':report['summary'],'wall_seconds':report['wall_seconds']}
        print(json.dumps({'epoch':epoch,'mode':mode,'gameplay':report['summary']}),flush=True)
    return results

def write_epoch_csv(out,history):
    rows=[]
    for epoch in history:
        for mode,gameplay in epoch.get('gameplay',{}).items():
            summary=gameplay['summary']
            for side in summary['sides']:
                row={k:side[k] for k in ('controller','side','games','lh','deny','deaths','raze_hero_hits','raze_creep_hits','razes_hitting_hero','razes_hitting_any_target','stacked_raze_hero_hits','attack_cancellations')}
                row.update(epoch=epoch['epoch'],mode=mode,lane_credited=summary['all_lane_creeps']['numerator'],lane_deaths=summary['all_lane_creeps']['denominator'],accessible_credited=summary['within_attack_distance']['numerator'],accessible_deaths=summary['within_attack_distance']['denominator'])
                for key in ('mid_arrival_game_seconds','mid_arrival_elapsed_seconds','first_move_game_seconds','first_death_game_seconds','longest_movement_stall_seconds','longest_idle_seconds','level','net_worth','health'):
                    for stat in ('count','denominator','mean','minimum','maximum'):row[key+'_'+stat]=side[key][stat]
                for i,name in enumerate(summary['spell_names']):row[name+'_completed']=side['spell_completions'][i]
                for i,name in enumerate(summary['action_names']):row[name+'_commands']=side['actions'][i]
                rows.append(row)
    if rows:
        with (out/'epoch-metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def recurrent(model,b,batch_size,resets):
    """One fast LSTM call normally; split only at actual internal life resets."""
    t=resets.shape[1];fused,keys=model.encode_entities(b);fused=fused.reshape(batch_size,t,-1)
    boundaries=[0]+[i for i in range(1,t) if resets[:,i].any()]+[t]
    state=model.initial_state(batch_size);output=[]
    for begin,end in zip(boundaries,boundaries[1:]):
        keep=torch.as_tensor(1.-resets[:,begin],device=fused.device).reshape(1,batch_size,1)
        state=tuple(v*keep for v in state);h,state=model.memory(fused[:,begin:end],state);output.append(h)
    return torch.cat(output,1).reshape(batch_size*t,-1),keys

def batches(root,manifest,stage,batch_size,seed):
    files=list(manifest['files']);rng=random.Random(seed);rng.shuffle(files)
    for entry in files:
        if not any(split(a)==stage for a in entry['actors']):continue
        header,obs,signals=read_block(root/entry['file'],entry['sha256'])
        ids=[i for i,a in enumerate(header['actors']) if split(a)==stage and (signals[i,:,11]>.5).any()]
        rng.shuffle(ids)
        for begin in range(0,len(ids),batch_size):
            chosen=ids[begin:begin+batch_size];arrays=decode(header,obs[chosen],signals[chosen]);labels=arrays.pop('labels');valid=arrays.pop('loss_valid')
            yield arrays,labels,valid,len(chosen),signals[chosen,:,10].copy()

def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('output');p.add_argument('--max-updates',type=int);p.add_argument('--resume');p.add_argument('--check-resume',action='store_true');a=p.parse_args()
    if a.max_updates is not None and a.max_updates<=0:raise ValueError('Positive smoke update limit required')
    config=json.loads(Path(a.config).read_text());allowed={'dataset','epochs','batch_sequences','learning_rate','seed','gradient_norm','attack_weight','evaluation'}
    if set(config)-allowed:raise ValueError('Unknown IL configuration keys')
    evaluation=config.get('evaluation')
    if evaluation:
        if set(evaluation)-{'executable','engine','matches','seconds','seed','modes'}:raise ValueError('Unknown evaluation configuration keys')
        if not resolved(evaluation['executable']).is_file() or not resolved(evaluation['engine']).is_file():raise FileNotFoundError('Evaluation executable/engine missing')
        if evaluation.get('matches',32)<=0 or evaluation.get('matches',32)%2 or not 0<evaluation.get('seconds',600)<=600:raise ValueError('Invalid evaluation match count/horizon')
        if any(m not in ('teacher','mirror') for m in evaluation.get('modes',['teacher','mirror'])):raise ValueError('Unknown gameplay mode')
    torch.set_num_threads(2);seed=config.get('seed',20261102);torch.manual_seed(seed)
    root=local_path(config['dataset']);root=root if root.is_absolute() else ROOT/root;manifest=json.loads((root/'dataset.json').read_text())
    if manifest['family']!='GGD' or manifest['observation_profile']!='ggd-v2':raise ValueError('Wrong demonstration contract')
    if manifest['partial'] and not a.max_updates:raise ValueError('Partial dataset is allowed only in a smoke run')
    dataset_hash=hashlib.sha256((root/'dataset.json').read_bytes()).hexdigest()
    resumed=load_resume(a.resume,config,dataset_hash) if a.resume else None
    if a.check_resume:
        if resumed is None:raise ValueError('--check-resume requires --resume')
        print(json.dumps({'passed':True,'check':'CPU checkpoint inspection; no training started','completed_epochs':resumed['epoch'],'next_epoch':resumed['epoch']+1,'target_epochs':config.get('epochs',5),'optimizer_steps':resumed['optimizer_steps'],'optimizer_states':len(resumed['optimizer']['state']),'dataset_sha256':dataset_hash}));return
    if not torch.cuda.is_available():raise RuntimeError('GGD IL requires a ROCm/CUDA GPU; CPU fallback is disabled')
    for stage in ('train','validation'):
        if not any(any(split(v)==stage for v in e['actors']) for e in manifest['files']):raise ValueError('Separate train and validation matches required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():raise FileExistsError('Choose a fresh training directory')
    epochs=config.get('epochs',5);batch_size=config.get('batch_sequences',8);lr=config.get('learning_rate',1e-4);clip=config.get('gradient_norm',1.);attack_weight=config.get('attack_weight',4.)
    if min(epochs,batch_size,lr,clip)<=0 or attack_weight<1:raise ValueError('Invalid IL parameters')
    name='GGD0-smoke' if a.max_updates else 'GGD0'
    model=Policy(combat_inputs=True).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    if resumed:
        model.load_state_dict(resumed['state_dict'],strict=True);optimizer.load_state_dict(resumed['optimizer'])
    initial_actor=model.action_head.weight.detach().clone();initial_value=[v.detach().clone() for v in model.value_head.parameters()]
    metadata={**config,'family':'GGD','model_name':name,'teacher_policy_sha256':manifest['teacher_sha256'],'observation_profile':'ggd-v2','initialization':'fresh random weights; no MIA checkpoint or optimizer loaded','device':torch.cuda.get_device_name(0),'torch':torch.__version__,'hip':torch.version.hip,'parameters':sum(p.numel() for p in model.parameters()),'dataset_sha256':hashlib.sha256((root/'dataset.json').read_bytes()).hexdigest()}
    if resumed:metadata.update(resume_checkpoint=str(a.resume),resumed_after_epoch=resumed['epoch'],initialization=resumed['config']['initialization'])
    (out/'config.json').write_text(json.dumps(metadata,indent=2))
    started=time.monotonic();updates=resumed['optimizer_steps'] if resumed else 0;history=resumed['history'] if resumed else [];best=min((r['validation']['nll'] for r in history),default=math.inf);stopped=False;first_gradient=None;step_times=[]
    if resumed:
        best_epoch=min(history,key=lambda r:r['validation']['nll'])['epoch'];source=resolved(a.resume).parent/f'epoch-{best_epoch:03d}.pt'
        if best_epoch==resumed['epoch']:torch.save(resumed,out/'best.pt')
        elif source.exists():shutil.copy2(source,out/'best.pt')
        else:raise FileNotFoundError('Historical best checkpoint missing: '+str(source))
    torch.cuda.synchronize()
    for epoch in range(resumed['epoch'] if resumed else 0,epochs):
        row={'epoch':epoch+1}
        for stage in ('train','validation'):
            training=stage=='train';model.train(training);total_nll=0.;frames=0;correct=0;count=0;tp=0;pred_attacks=0;label_attacks=0;targets_correct=0
            for arrays,labels,valid,n,resets in batches(root,manifest,stage,batch_size,seed+epoch):
                if training and a.max_updates and updates>=a.max_updates:stopped=True;break
                if (out/'stop.json').exists():stopped=True;break
                tick=time.perf_counter();b=packed_tensors(arrays,'cuda');y=torch.as_tensor(labels,device='cuda');mask=torch.as_tensor(valid,device='cuda')
                with torch.set_grad_enabled(training):
                    h,keys=recurrent(model,b,n,resets)
                    action={k:y[:,i] for i,k in enumerate(('action','buy','item','target','x','y'))}
                    result=model.heads(b,h,keys,torch.zeros(len(y),device='cuda'),None,actions=action,return_metrics=True)
                    logp=result[1];prediction=result[-1];weights=mask*(1.+(attack_weight-1.)*(y[:,0]==3))
                    loss=-(logp*weights).sum()/weights.sum().clamp_min(1)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite IL loss')
                    if training:
                        optimizer.zero_grad(set_to_none=True);loss.backward()
                        if first_gradient is None:
                            first_gradient=float(model.action_head.weight.grad.abs().sum())
                            if not first_gradient>0:raise RuntimeError('Action head has no supervised gradient')
                        torch.nn.utils.clip_grad_norm_(model.parameters(),clip);optimizer.step();updates+=1
                with torch.no_grad():
                    pa=prediction['action']==3;la=y[:,0]==3
                    stats=torch.stack((-(logp*mask).sum(),mask.sum(),((prediction['action']==y[:,0])&mask).sum(),(pa&la&mask).sum(),(pa&mask).sum(),(la&mask).sum(),((prediction['target']==y[:,3])&la&mask).sum())).cpu().tolist()
                total_nll+=stats[0];frames+=int(stats[1]);correct+=int(stats[2]);tp+=int(stats[3]);pred_attacks+=int(stats[4]);label_attacks+=int(stats[5]);targets_correct+=int(stats[6]);count+=1
                elapsed=time.perf_counter()-tick
                if training:step_times.append(elapsed)
                progress={'family':'GGD','model_name':name,'stage':stage,'epoch':epoch+1,'updates':updates,'batches':count,'frames':frames,'nll':total_nll/max(1,frames),'action_accuracy':correct/max(1,frames),'attack_precision':{'numerator':tp,'denominator':pred_attacks},'attack_recall':{'numerator':tp,'denominator':label_attacks},'conditional_attack_target_accuracy':{'numerator':targets_correct,'denominator':label_attacks},'last_batch_seconds':elapsed,'wall_seconds':time.monotonic()-started}
                (out/'status.json').write_text(json.dumps(progress,indent=2))
                if count%8==0 or a.max_updates:print(json.dumps(progress),flush=True)
                if not training and a.max_updates and count>=2:break
            if training and a.max_updates and updates>=a.max_updates:stopped=True
            row[stage]={'nll':total_nll/max(1,frames),'action_accuracy':correct/max(1,frames),'action_correct':correct,'frames':frames,'batches':count,'attack_precision':{'numerator':tp,'denominator':pred_attacks},'attack_recall':{'numerator':tp,'denominator':label_attacks},'conditional_attack_target_accuracy':{'numerator':targets_correct,'denominator':label_attacks}}
        history.append(row)
        checkpoint={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'config':metadata,'epoch':epoch+1,'history':history,'spec_version':2,'gpu_pipeline':{'recurrent_reset_interval':128,'observation_storage':'f16','observation_spec':2,'observation_profile':'ggd-v2'},'optimizer_steps':updates}
        torch.save(checkpoint,out/f'epoch-{epoch+1:03d}.pt')
        if evaluation:row['gameplay']=evaluate_epoch(model,out/f'epoch-{epoch+1:03d}.pt',out,epoch+1,evaluation,metadata)
        torch.save(checkpoint,out/f'epoch-{epoch+1:03d}.pt')
        torch.save(checkpoint,out/'last.pt')
        if row['validation']['nll']<best:best=row['validation']['nll'];torch.save(checkpoint,out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
        write_epoch_csv(out,history)
        (out/f'epoch-{epoch+1:03d}' ).mkdir(exist_ok=True);(out/f'epoch-{epoch+1:03d}'/'report.json').write_text(json.dumps(row,indent=2))
        if stopped:break
    actor_delta=float((model.action_head.weight.detach()-initial_actor).norm());value_delta=max(float((p.detach()-v).abs().max()) for p,v in zip(model.value_head.parameters(),initial_value))
    if not actor_delta>0 or value_delta!=0:raise RuntimeError(f'Invalid parameter updates: actor={actor_delta}, value={value_delta}')
    result={'family':'GGD','model_name':name,'complete':not stopped and not a.max_updates,'initialization':metadata['initialization'],'device':metadata['device'],'runtime':'PyTorch ROCm' if torch.version.hip else 'PyTorch CUDA','updates':updates,'epochs':len(history),'wall_seconds':time.monotonic()-started,'first_action_head_gradient_l1':first_gradient,'actor_delta':actor_delta,'value_head_delta':value_delta,'best_validation_nll':best,'mean_batch_seconds_after_first':sum(step_times[1:])/max(1,len(step_times)-1),'batch_sequences':batch_size,'sequence_length':128,'dataset_sha256':metadata['dataset_sha256'],'teacher_sha256':manifest['teacher_sha256'],'checkpoint':str(out/'best.pt'),'test_split_used':False,'history':history}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
