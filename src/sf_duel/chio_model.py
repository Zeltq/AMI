"""CHiO encoders and conditional action distribution; no purchase head."""
import torch
from torch import nn
from .model import mlp,pool,distribution
from .spec import UNIT_FIELDS,COMBAT_UNIT_FIELDS,TREE_FIELDS,PROJECTILE_FIELDS,COMBAT_PROJECTILE_FIELDS,SKILL_FIELDS,ITEM_FIELDS
from .timing_observations import TIMING_FIELDS
from .enemy_observations import FIELDS as ENEMY_FIELDS

HEADS=('action','direction','enemy_target','ally_target')
PREVIOUS=55


class Policy(nn.Module):
    def __init__(self):
        super().__init__();self.hidden_size=256
        unit=len(UNIT_FIELDS)+len(COMBAT_UNIT_FIELDS)+len(TIMING_FIELDS)
        self.hero=mlp(2*(unit+len(ENEMY_FIELDS)+25),128,128);self.unit=mlp(2*(unit+17),64,64)
        self.tree=mlp(2*len(TREE_FIELDS),32,32);self.projectile=mlp(2*(len(PROJECTILE_FIELDS)+len(COMBAT_PROJECTILE_FIELDS)),32,32)
        self.skill=mlp(2*len(SKILL_FIELDS),32,32);self.item=mlp(2*len(ITEM_FIELDS),32,32)
        self.terrain=nn.Sequential(nn.Conv2d(4,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.ReLU(),
                                   nn.AdaptiveAvgPool2d((2,2)),nn.Flatten(),nn.Linear(64,64),nn.ReLU())
        self.combine=nn.Sequential(nn.Linear(1636-128-30+PREVIOUS,256),nn.ReLU())
        self.memory=nn.LSTM(256,256,batch_first=True);self.action_embed=nn.Embedding(13,16)
        self.action_head=nn.Linear(256,13);self.direction_head=nn.Linear(272,36)
        self.enemy_query=nn.Linear(272,64);self.ally_query=nn.Linear(272,64)
        self.value_head=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
        for layer in (self.action_head,self.direction_head,self.enemy_query,self.ally_query):
            nn.init.normal_(layer.weight,std=.01);nn.init.zeros_(layer.bias)

    def initial_state(self,n,device=None):
        return tuple(torch.zeros(1,n,256,device=device or next(self.parameters()).device) for _ in range(2))

    def encode_entities(self,b):
        heroes=self.hero(torch.cat((b['heroes'],b['hero_combat']),-1))*b['hero_valid'].unsqueeze(-1)
        creeps=self.unit(torch.cat((b['creeps'],b['creep_combat']),-1));towers=self.unit(torch.cat((b['towers'],b['tower_combat']),-1))
        trees=self.tree(b['trees']);projectiles=self.projectile(torch.cat((b['projectiles'],b['projectile_combat']),-1))
        parts=[heroes.flatten(1),towers.flatten(1),pool(creeps,b['creep_valid']&b['creep_ally']),pool(creeps,b['creep_valid']&~b['creep_ally']),
               pool(trees,b['tree_valid']),pool(projectiles,b['projectile_valid']),self.skill(b['skills']).flatten(1),
               self.item(b['items']).flatten(1),(self.item(b['enemy_items'])*b['hero_valid'][:,1,None,None]).flatten(1),
               self.terrain(b['terrain']),b['global_features'],b['previous']]
        return self.combine(torch.cat(parts,-1)),creeps

    def forward(self,b,state=None):
        fused,keys=self.encode_entities(b);output,state=self.memory(fused[:,None,:],state);h=output[:,0]
        return h,keys,self.value_head(h).squeeze(-1),state

    def heads(self,b,h,keys,value=None,state=None,actions=None,deterministic=False,return_logits=False):
        chosen={} if actions is None else dict(actions)
        def choose(name,dist):
            if actions is None:chosen[name]=dist.logits.argmax(-1) if deterministic else dist.sample()
            return chosen[name]
        raw_action=self.action_head(h);action=distribution(raw_action,b['action_mask']);kind=choose('action',action)
        condition=torch.cat((h,self.action_embed(kind)),-1)
        raw_direction=self.direction_head(condition);direction=distribution(raw_direction,torch.ones_like(raw_direction,dtype=torch.bool));choose('direction',direction)
        moving=kind==0;enemy=kind==2;ally=kind==3
        raw_enemy=torch.einsum('bu,bnu->bn',self.enemy_query(condition),keys)/8
        raw_ally=torch.einsum('bu,bnu->bn',self.ally_query(condition),keys)/8
        enemy_mask=torch.where(enemy[:,None],b['enemy_target_mask'],torch.ones_like(b['enemy_target_mask']))
        ally_mask=torch.where(ally[:,None],b['ally_target_mask'],torch.ones_like(b['ally_target_mask']))
        enemy_dist=distribution(raw_enemy,enemy_mask);ally_dist=distribution(raw_ally,ally_mask)
        choose('enemy_target',enemy_dist);choose('ally_target',ally_dist)
        logp=action.log_prob(kind)+moving*direction.log_prob(chosen['direction'])+enemy*enemy_dist.log_prob(chosen['enemy_target'])+ally*ally_dist.log_prob(chosen['ally_target'])
        def normalized(dist,mask):
            count=mask.sum(-1);return torch.where(count>1,dist.entropy()/count.clamp_min(2).float().log(),torch.zeros_like(logp))
        entropy=(normalized(action,b['action_mask'])+moving*normalized(direction,torch.ones_like(raw_direction,dtype=torch.bool))+
                 enemy*normalized(enemy_dist,enemy_mask)+ally*normalized(ally_dist,ally_mask))/(1+moving.float()+enemy.float()+ally.float())
        for name,used in [('direction',moving),('enemy_target',enemy),('ally_target',ally)]:chosen[name]=torch.where(used,chosen[name],torch.zeros_like(chosen[name]))
        result=(chosen,logp,entropy,self.value_head(h).squeeze(-1) if value is None else value,state)
        return (*result,(raw_action,raw_direction,raw_enemy,raw_ally)) if return_logits else result

    def decide(self,b,state=None,actions=None,deterministic=False,need_entropy=True):
        h,keys,value,state=self.forward(b,state)
        return self.heads(b,h,keys,value,state,actions,deterministic)
