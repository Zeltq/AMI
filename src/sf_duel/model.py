"""Entity encoders, one shared LSTM and conditional policy/value outputs."""
import torch
from torch import nn
from torch.distributions import Categorical
from .spec import *

def mlp(i,h,o):
    return nn.Sequential(nn.Linear(i,h),nn.ReLU(),nn.Linear(h,o),nn.ReLU())

def pool(x,mask):
    weights=mask.unsqueeze(-1)
    mean=(x*weights).sum(1)/weights.sum(1).clamp_min(1)
    maximum=x.masked_fill(~weights,-torch.inf).amax(1)
    maximum=torch.where(mask.any(1,keepdim=True),maximum,torch.zeros_like(maximum))
    return torch.cat((mean,maximum),-1)

def distribution(logits, mask):
    if mask.device.type == 'cpu' and not bool(mask.any(-1).all()):
        raise ValueError('Categorical distribution has no legal action')
    return Categorical(logits=logits.masked_fill(~mask,torch.finfo(logits.dtype).min), validate_args=False)

class Policy(nn.Module):
    def __init__(self,hidden_size=256):
        super().__init__()
        self.hidden_size=hidden_size
        self.hero=mlp(2*len(UNIT_FIELDS),128,128)
        self.unit=mlp(2*len(UNIT_FIELDS),64,64)
        self.tree=mlp(2*len(TREE_FIELDS),32,32)
        self.projectile=mlp(2*len(PROJECTILE_FIELDS),32,32)
        self.skill=mlp(2*len(SKILL_FIELDS),32,32)
        self.item=mlp(2*len(ITEM_FIELDS),32,32)
        self.terrain=nn.Sequential(nn.Conv2d(4,16,3,padding=1),nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((2,2)),nn.Flatten(),
            nn.Linear(64,64),nn.ReLU())
        previous_size=len(Act)+4+len(ITEMS)+len(BUYS)
        combined=256+256+256+64+64+4*32+2*len(ITEMS)*32+64+2*len(GLOBAL_FIELDS)+previous_size
        self.combine=nn.Sequential(nn.Linear(combined,256),nn.ReLU())
        self.memory=nn.LSTM(256,hidden_size,batch_first=True)
        self.action_embed=nn.Embedding(len(Act),16)
        self.item_embed=nn.Embedding(len(ITEMS),16)
        self.action_head=nn.Linear(hidden_size,len(Act))
        self.buy_head=nn.Linear(hidden_size,len(BUYS))
        self.item_head=nn.Linear(hidden_size+16,len(ITEMS))
        self.x_head=nn.Linear(hidden_size+32,len(COORDINATES))
        self.y_head=nn.Linear(hidden_size+32,len(COORDINATES))
        self.query=nn.Linear(hidden_size+32,64)
        self.hero_key=nn.Linear(128,64)
        self.tree_key=nn.Linear(32,64)
        self.base_key=nn.Parameter(torch.zeros(1,1,64))
        self.value_head=nn.Sequential(nn.Linear(hidden_size,128),nn.ReLU(),nn.Linear(128,1))
        # Small final logits avoid accidental almost-deterministic random policies.
        for layer in (self.action_head,self.buy_head,self.item_head,self.x_head,self.y_head,self.query):
            nn.init.normal_(layer.weight, std=.01)
            nn.init.zeros_(layer.bias)

    def initial_state(self,n,device=None):
        device=device or next(self.parameters()).device
        return tuple(torch.zeros(1,n,self.hidden_size,device=device) for _ in range(2))

    def encode_entities(self,b):
        heroes=self.hero(b['heroes'])*b['hero_valid'].unsqueeze(-1)
        creeps=self.unit(b['creeps'])
        towers=self.unit(b['towers'])
        trees=self.tree(b['trees'])
        projectiles=self.projectile(b['projectiles'])
        parts=[heroes.flatten(1),towers.flatten(1),
            pool(creeps,b['creep_valid']&b['creep_ally']),
            pool(creeps,b['creep_valid']&~b['creep_ally']),
            pool(trees,b['tree_valid']),pool(projectiles,b['projectile_valid']),
            self.skill(b['skills']).flatten(1),self.item(b['items']).flatten(1),
            (self.item(b['enemy_items'])*b['hero_valid'][:,1,None,None]).flatten(1),
            self.terrain(b['terrain']),b['global_features'],b['previous']]
        fused=self.combine(torch.cat(parts,-1))
        keys=torch.cat((self.hero_key(heroes),creeps,towers,self.tree_key(trees),
                         self.base_key.expand(len(fused),-1,-1)),1)
        return fused,keys

    def forward(self,b,state=None):
        fused,keys=self.encode_entities(b)
        output,new_state=self.memory(fused.unsqueeze(1),state)
        h=output[:,0]
        return h,keys,self.value_head(h).squeeze(-1),new_state

    def sequence_decide(self,b,batch_size,length,state=None,actions=None,deterministic=False,return_metrics=False):
        fused,keys=self.encode_entities(b)
        output,new_state=self.memory(fused.reshape(batch_size,length,-1),state)
        h=output.reshape(batch_size*length,-1)
        return self.heads(b,h,keys,self.value_head(h).squeeze(-1),new_state,actions,deterministic,return_metrics)

    def decide(self,b,state=None,actions=None,deterministic=False,need_entropy=True):
        h,keys,value,state=self.forward(b,state)
        return self.heads(b,h,keys,value,state,actions,deterministic,need_entropy=need_entropy)

    def heads(self,b,h,keys,value,state,actions=None,deterministic=False,return_metrics=False,need_entropy=True):
        chosen={} if actions is None else dict(actions)
        def choose(name,dist):
            if actions is None:
                chosen[name]=dist.logits.argmax(-1) if deterministic else dist.sample()
            return chosen[name]
        action_dist=distribution(self.action_head(h),b['action_mask'])
        action=choose('action',action_dist)
        buy_dist=distribution(self.buy_head(h),b['buy_mask'])
        choose('buy',buy_dist)
        is_item=action==Act.ITEM
        item_mask=torch.where(is_item[:,None],b['item_mask'],torch.ones_like(b['item_mask']))
        item_dist=distribution(self.item_head(torch.cat((h,self.action_embed(action)),-1)),item_mask)
        item=choose('item',item_dist)
        item=torch.where(is_item,item,torch.zeros_like(item));chosen['item']=item
        condition=torch.cat((h,self.action_embed(action),self.item_embed(item)),-1)
        target_mask=torch.where((action==Act.ATTACK)[:,None],b['attack_mask'],b['self_mask'])
        target_mask=torch.where((is_item&(item==3))[:,None],b['tp_mask'],target_mask)
        target_mask=torch.where((is_item&(item==6))[:,None],b['tree_mask'],target_mask)
        target_logits=torch.einsum('bu,bnu->bn',self.query(condition),keys)/8
        target_dist=distribution(target_logits,target_mask)
        choose('target',target_dist)
        x_dist=Categorical(logits=self.x_head(condition),validate_args=False)
        y_dist=Categorical(logits=self.y_head(condition),validate_args=False)
        choose('x',x_dist);choose('y',y_dist)
        spatial=(action==Act.MOVE)|(action==Act.MOVE_DIRECT)|(is_item&(item==3))
        targeted=(action==Act.ATTACK)|(is_item&((item==3)|(item==6)))
        log_prob=action_dist.log_prob(action)+buy_dist.log_prob(chosen['buy'])
        log_prob=log_prob+is_item*item_dist.log_prob(item)+targeted*target_dist.log_prob(chosen['target'])
        log_prob=log_prob+spatial*(x_dist.log_prob(chosen['x'])+y_dist.log_prob(chosen['y']))
        entropy=torch.zeros_like(value)
        if need_entropy or return_metrics:
            entropy=action_dist.entropy()+buy_dist.entropy()+is_item*item_dist.entropy()
            entropy=entropy+targeted*target_dist.entropy()+spatial*(x_dist.entropy()+y_dist.entropy())
        chosen['x']=torch.where(spatial,chosen['x'],torch.full_like(chosen['x'],64))
        chosen['y']=torch.where(spatial,chosen['y'],torch.full_like(chosen['y'],64))
        chosen['target']=torch.where(targeted,chosen['target'],torch.zeros_like(chosen['target']))
        result=(chosen,log_prob,entropy,value,state)
        if return_metrics:
            predictions={name:dist.logits.argmax(-1) for name,dist in [('action',action_dist),('buy',buy_dist),('item',item_dist),('target',target_dist),('x',x_dist),('y',y_dist)]}
            def normalized(dist, mask):
                count=mask.sum(-1)
                return torch.where(count>1,dist.entropy()/count.clamp_min(2).float().log(),torch.zeros_like(value))
            # A scale-controlled conditional-head regularizer, not joint entropy.
            regularizer=normalized(action_dist,b['action_mask'])+normalized(buy_dist,b['buy_mask'])
            regularizer=regularizer+is_item*normalized(item_dist,item_mask)+targeted*normalized(target_dist,target_mask)
            regularizer=regularizer+spatial*(x_dist.entropy()+y_dist.entropy())/torch.log(h.new_tensor(float(len(COORDINATES))))
            predictions['normalized_entropy']=regularizer/(2+is_item.float()+targeted.float()+2*spatial.float())
            return (*result,predictions)
        return result
