"""Read existing trajectories and audit frozen-feature critic fitting; no PPO or publication."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from sf_duel.model import Policy
from sf_duel.selfplay.config import Config
from sf_duel.selfplay.ppo import load_dataset, refresh
from sf_duel.selfplay.rewards import gae


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run)
    out=root/'critic-feature-diagnosis';out.mkdir(exist_ok=False);cfg=Config.read(root/'config.json')
    torch.set_num_threads(2);torch.manual_seed(cfg.seed);saved=torch.load(root/'pilot/behavior.pt',map_location='cpu',weights_only=False)
    model=Policy().cuda();model.load_state_dict(saved['state_dict']);frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('value_head.')}
    meta,episodes=load_dataset(root/'critic-expanded-view');data=[];started=time.monotonic()
    def write(name,d):
        temp=out/(name+'.tmp');temp.write_text(json.dumps(d,indent=2));temp.replace(out/(name+'.json'))
    for i,(entry,e) in enumerate(zip(meta['episodes'],episodes)):
        h=refresh(model,e,cfg,latents=True)['latent'];mask=np.asarray(e['loss_valid']);y=gae(e['rewards'],np.zeros(len(mask)),e['terminals'],gamma=1,lam=1)[1]
        data.append((entry,h[mask],torch.as_tensor(y[mask]),np.asarray(e['time'])[mask]))
        write('status',{'phase':'extracting_frozen_features','episodes':i+1,'total':len(episodes)})
    train=[d for d in data if d[0]['split']=='train'];val=[d for d in data if d[0]['split']=='validation']
    x=torch.cat([d[1] for d in train]);y=torch.cat([d[2] for d in train]);constant=float(y.mean())
    def evaluate(rows):
        groups={};games={}
        with torch.no_grad():
            for entry,h,target,t in rows:
                pred=torch.cat([model.value_head(v.cuda()).squeeze(-1).cpu() for v in h.split(2048)])
                err=(pred-target).square().numpy();base=(target-constant).square().numpy()
                masks={'all':np.ones(len(t),bool),'pregame':t<0,'first_2min':(t>=0)&(t<120),'middle':(t>=120)&(t<420),'late':t>=420,'team_'+str(entry['team']):np.ones(len(t),bool)}
                for name,mask in masks.items():
                    g=groups.setdefault(name,[0.,0.,0]);g[0]+=float(err[mask].sum());g[1]+=float(base[mask].sum());g[2]+=int(mask.sum())
                g=games.setdefault(entry['match'],[0.,0.,0]);g[0]+=float(err.sum());g[1]+=float(base.sum());g[2]+=len(err)
        def unpack(d):return {k:{'mse':v[0]/v[2],'baseline':v[1]/v[2],'count':v[2]} for k,v in d.items() if v[2]}
        return {'groups':unpack(groups),'games':unpack(games)}
    initial=evaluate(val);history=[];best=initial['groups']['all']['mse'];best_step=0
    for name,v in model.named_parameters():v.requires_grad_(name.startswith('value_head.'))
    optimizer=torch.optim.Adam(model.value_head.parameters(),lr=cfg.value_warmup_lr)
    # Same initialization, order, LR, minibatch and gradient cap as production.
    # Observe the first epoch more closely; do not export/select a policy here.
    batches=list(torch.randperm(len(y)).split(2048))
    for step,ids in enumerate(batches,1):
        loss=(model.value_head(x[ids].cuda()).squeeze(-1)-y[ids].cuda()).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.value_head.parameters(),cfg.gradient_norm);optimizer.step()
        if step in (1,5,10,25,50,100,150,200,len(batches)):
            result=evaluate(val);score=result['groups']['all']['mse'];row={'step':step,'validation':result};history.append(row)
            if score<best:best=score;best_step=step
            write('status',{'phase':'diagnostic_head_fit','step':step,'total':len(batches),'mse':score,'best_mse':best,'best_step':best_step})
    for k,v in frozen.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
    write('complete',{'initial':initial,'history':history,'best_mse':best,'best_step':best_step,'training':evaluate(train),'actor_unchanged':True,'checkpoint_published':False,'seconds':time.monotonic()-started})


if __name__=='__main__':main()
