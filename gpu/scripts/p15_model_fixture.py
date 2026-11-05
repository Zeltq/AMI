"""Deterministic FP16 inputs and 128 recurrent CHiO steps for Burn parity."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.chio_model import Policy,HEADS
from weights import write_bundle,rust_name


def batch(model,n=128,seed=20261103):
    rng=np.random.default_rng(seed);pc,pt,pp=12,4,6
    shapes={'heroes':(2,96),'creeps':(pc,96),'towers':(4,96),'trees':(pt,8),'projectiles':(pp,12),
            'skills':(4,12),'items':(6,16),'enemy_items':(6,16),'terrain':(4,8,8),'global_features':(6,), 'previous':(55,),
            'hero_combat':(2,model.hero[0].in_features-96),'creep_combat':(pc,model.unit[0].in_features-96),
            'tower_combat':(4,model.unit[0].in_features-96),'projectile_combat':(pp,model.projectile[0].in_features-12)}
    b={name:torch.from_numpy(rng.normal(0,.2,(n,*shape)).astype(np.float16).astype(np.float32)) for name,shape in shapes.items()}
    for name,size in [('hero_valid',2),('creep_valid',pc),('creep_ally',pc),('tree_valid',pt),('projectile_valid',pp),
                      ('action_mask',13),('buy_mask',7),('item_mask',8),('attack_mask',2+pc+4+pt+1),('tree_mask',2+pc+4+pt+1),
                      ('tp_mask',2+pc+4+pt+1),('self_mask',2+pc+4+pt+1)]:b[name]=torch.ones((n,size),dtype=torch.bool)
    b['creep_ally'][:,1::2]=False;b['creep_valid'][:,-2:]=False
    b['enemy_target_mask']=b['creep_valid']&~b['creep_ally'];b['ally_target_mask']=b['creep_valid']&b['creep_ally']
    labels=torch.zeros((n,4),dtype=torch.long)
    for i in range(n):
        kind=i%13;labels[i,0]=kind
        if kind==0:labels[i,1]=i%36
        if kind==2:labels[i,2]=1
        if kind>=6 and kind<=9:b['action_mask'][i]=False;b['action_mask'][i,1]=True;b['action_mask'][i,kind]=True
    return b,labels


def export_weights(model,path):
    arrays={}
    for name,value in model.state_dict().items():
        value=value.detach().cpu().float().numpy()
        if value.ndim==2 and not name.endswith('_embed.weight'):value=value.T
        arrays[rust_name(name)]=value
    write_bundle(path,arrays,family='CHiO',model_name='CHiO',observation_profile='chio-v1',action_types=13,directions=36,
                 decision_ticks=6,offset_ticks=6,recurrent_reset=128,timing_honest=True,timing_oracle=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(20261103)
    model=Policy().eval();b,labels=batch(model);export_weights(model,a.output/'weights')
    resets=np.zeros(128,dtype=np.float32);resets[[0,63]]=1
    with torch.no_grad():
        fused,keys=model.encode_entities(b);memory=None;hs=[]
        for i in range(128):
            if resets[i]:memory=None
            _,memory=model.memory(fused[i:i+1,None,:],memory);hs.append(memory[0][0])
        h=torch.cat(hs);value=model.value_head(h).squeeze(-1)
        _,logp,_,_,_,logits=model.heads(b,h,keys,value,actions={name:labels[:,i] for i,name in enumerate(HEADS)},return_logits=True)
    arrays={k:v.numpy() for k,v in b.items()};arrays.update(labels=labels.numpy(),resets=resets,expected_h=h.numpy(),expected_value=value.numpy(),expected_logp=logp.numpy())
    arrays.update({f'expected_{name}_logits':value.numpy() for name,value in zip(HEADS,logits)})
    write_bundle(a.output/'fixture',arrays,source='synthetic FP16 public input fixture, fixed seed, life reset at step 63')
    torch.save({'state_dict':model.state_dict(),'config':dict(observation_profile='chio-v1',model_name='CHiO'),'spec_version':5},a.output/'initial.pt')
    result=dict(parameters=sum(p.numel() for p in model.parameters()),tensors=len(model.state_dict()),steps=128,life_reset_step=63)
    (a.output/'fixture.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result))


if __name__=='__main__':main()
