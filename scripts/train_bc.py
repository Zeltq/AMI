"""Recurrent behavior cloning on a whole-match holdout. No PPO/self-play."""
import argparse,json,math,random,time,sys,shutil
from pathlib import Path
import numpy as np
import torch
from sf_duel.model import Policy
from sf_duel.bc_data import HEADS,sequence_batch
from sf_duel.bc_metrics import ImitationMetrics,write_progress

def linux_path(value):
 value=str(value).replace('\\','/')
 return '/mnt/'+value[0].lower()+value[2:] if len(value)>1 and value[1]==':' else value

def minibatches(episodes,batch_size,length,burn,rng):
 ids=list(range(len(episodes)));rng.shuffle(ids)
 for g in range(0,len(ids),4):
  selected=ids[g:g+4];loaded=[]
  for i in selected:
   with np.load(episodes[i],allow_pickle=False) as source:loaded.append({k:source[k] for k in source.files})
  chunks=[(i,start) for i,e in enumerate(loaded) for start in range(0,len(e['labels']),length-burn)]
  rng.shuffle(chunks)
  for j in range(0,len(chunks),batch_size):
   selection=chunks[j:j+batch_size];b,y,valid=sequence_batch(loaded,selection,length,burn)
   if valid.any():yield b,y,valid,len(selection)

def main():
 p=argparse.ArgumentParser();p.add_argument('cache');p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--length',type=int,default=80);p.add_argument('--burn-in',type=int,default=16);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--minutes',type=float,default=60);p.add_argument('--smoke-steps',type=int,default=0);p.add_argument('--resume');a=p.parse_args()
 cache=Path(a.cache);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'config.json').exists():raise FileExistsError('Training output already contains a run; use a new output directory')
 manifest=json.loads((cache/'dataset.json').read_text());episodes={'train':[],'validation':[]}
 for match in manifest['matches']:
  for team in (2,3):episodes[match['split']].append(cache/f"match-{match['index']:04d}-team-{team}.npz")
 if not all(episodes.values()):raise ValueError('Both train and validation matches required')
 torch.set_num_threads(2);torch.manual_seed(17);rng=random.Random(17)
 model=Policy().to('cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5)
 started=time.monotonic();best=math.inf;history=[];steps=0;initial_epoch=0;prior_elapsed=0
 if a.resume:
  saved=torch.load(a.resume,map_location='cpu',weights_only=False)
  if saved['spec_version']!=1 or saved['config']['teacher_policy_sha256']!=manifest['teacher_policy_sha256']:raise ValueError('Checkpoint contract/teacher mismatch')
  if saved['config']['train_matches']!=len(episodes['train'])//2 or saved['config']['validation_matches']!=len(episodes['validation'])//2:raise ValueError('Resume dataset split changed')
  model.load_state_dict(saved['state_dict'],strict=True);optimizer.load_state_dict(saved['optimizer'])
  history=saved['history'];initial_epoch=saved['epoch'];prior_elapsed=history[-1]['elapsed_seconds']
  steps=saved.get('optimizer_steps',max(int(v['step']) for v in optimizer.state.values() if 'step' in v))
  best=history[-1]['validation']['loss'];shutil.copy2(a.resume,out/'best.pt')
  if 'random_state' in saved:rng.setstate(saved['random_state'])
  if 'torch_rng_state' in saved:torch.set_rng_state(saved['torch_rng_state'])
 initial_steps=steps
 config={**vars(a),'device':torch.cuda.get_device_name(),'parameters':sum(p.numel() for p in model.parameters()),'train_matches':len(episodes['train'])//2,'validation_matches':len(episodes['validation'])//2,'teacher_policy_sha256':manifest['teacher_policy_sha256'],'objective':'masked joint action negative log likelihood','value_head_trained':False,'initial_epoch':initial_epoch,'initial_optimizer_steps':initial_steps}
 (out/'config.json').write_text(json.dumps(config,indent=2))
 for epoch in range(initial_epoch+1,initial_epoch+a.epochs+1):
  row={'epoch':epoch}
  for split in ('train','validation'):
   training=split=='train';model.train(training);loss_sum=0.;count=0;metrics=ImitationMetrics();batches=0
   for arrays,y,valid,n in minibatches(episodes[split],a.batch_size,a.length,a.burn_in,rng):
    step_started=time.perf_counter()
    b={k:torch.as_tensor(v,device='cuda',dtype=torch.float32 if v.dtype.kind=='f' else None) for k,v in arrays.items()}
    labels={k:torch.as_tensor(y[:,i],device='cuda') for i,k in enumerate(HEADS)};mask=torch.as_tensor(valid,device='cuda');size=int(valid.sum())
    with torch.set_grad_enabled(training):
     chosen,logp,_,_,_,pred=model.sequence_decide(b,n,a.length,actions=labels,return_metrics=True)
     loss=-(logp*mask).sum()/mask.sum()
     if not torch.isfinite(loss):raise RuntimeError('Nonfinite BC loss')
     if training:
      optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();steps+=1
    loss_sum+=float(loss.detach())*size;count+=size;batches+=1
    metrics.update(pred,labels,mask)
    if batches%25==0 or a.smoke_steps:
     progress={'epoch':epoch,'split':split,'batches':batches,'optimizer_steps':steps,'loss':loss_sum/count,'minutes':round((time.monotonic()-started)/60,2),'last_step_seconds':time.perf_counter()-step_started,'metrics':metrics.summary(),'last_validation':history[-1]['validation'] if history else None}
     write_progress(out,progress,history);print(json.dumps(progress),flush=True)
    if a.smoke_steps and steps-initial_steps>=a.smoke_steps:break
    if training and time.monotonic()-started>a.minutes*60:break
   row[split]={'loss':loss_sum/max(1,count),'frames':count,'metrics':metrics.summary()}
  row['session_elapsed_seconds']=time.monotonic()-started
  row['elapsed_seconds']=prior_elapsed+row['session_elapsed_seconds']
  row['validation_relative_improvement']=None if not history else (history[-1]['validation']['loss']-row['validation']['loss'])/max(history[-1]['validation']['loss'],1e-9)
  row['generalization_gap']=row['validation']['loss']-row['train']['loss']
  history.append(row)
  write_progress(out,{'epoch':epoch,'split':'epoch_complete','optimizer_steps':steps,'loss':row['validation']['loss'],'minutes':round(row['session_elapsed_seconds']/60,2),'metrics':row['validation']['metrics']},history)
  checkpoint={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'history':history,'config':config,'spec_version':1,'optimizer_steps':steps,'random_state':rng.getstate(),'torch_rng_state':torch.get_rng_state()}
  torch.save(checkpoint,out/'last.pt')
  if row['validation']['loss']<best:best=row['validation']['loss'];torch.save(checkpoint,out/'best.pt')
  (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
  if a.smoke_steps or time.monotonic()-started>a.minutes*60:break
 (out/'complete.json').write_text(json.dumps({'best_validation_loss':best,'epochs':len(history),'additional_epochs':len(history)-initial_epoch,'optimizer_steps':steps,'additional_optimizer_steps':steps-initial_steps,'wall_seconds':time.monotonic()-started,'checkpoint':str(out/'best.pt')},indent=2))
if __name__=='__main__':main()
