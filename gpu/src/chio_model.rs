//! ZOg-sized encoders/CNN/LSTM with the four conditional CHiO heads.
use burn::{module::Module,nn::{Linear,LinearConfig,Embedding,EmbeddingConfig,conv::{Conv2d,Conv2dConfig},PaddingConfig2d},
    tensor::{Tensor,Int,Bool,Device,activation}};
use crate::{contract::*,model::{Mlp,Memory,Observation,pool},chio_actions::Masks,chio_observe as obs};

#[derive(Module,Debug)]
pub struct Policy {
    pub hero:Mlp,pub unit:Mlp,pub tree:Mlp,pub projectile:Mlp,pub skill:Mlp,pub item:Mlp,
    pub terrain_first:Conv2d,pub terrain_second:Conv2d,pub terrain_linear:Linear,
    pub combine:Linear,pub memory:Memory,pub action_embed:Embedding,
    pub action_head:Linear,pub direction_head:Linear,pub enemy_query:Linear,pub ally_query:Linear,
    pub value_first:Linear,pub value_second:Linear,
}
pub struct Encoded {pub fused:Tensor<2>,pub keys:Tensor<3>}
pub struct Heads {pub actions:Tensor<2,Int>,pub logp:Tensor<1>,pub entropy:Tensor<1>,pub value:Tensor<1>,pub logits:[Tensor<2>;4]}

impl Policy {
    pub fn new(d:&Device)->Self {
        let unit=UNIT_COUNT+COMBAT_UNIT_COUNT+crate::timing_observe::FIELDS;
        let linear=|i,o|LinearConfig::new(i,o).init(d);
        Self{hero:Mlp::new(2*(unit+crate::enemy_observe::FIELDS+obs::HERO_FIELDS),128,128,d),
            unit:Mlp::new(2*(unit+obs::UNIT_FIELDS),64,64,d),tree:Mlp::new(2*TREE_COUNT,32,32,d),
            projectile:Mlp::new(2*(PROJECTILE_COUNT+COMBAT_PROJECTILE_COUNT),32,32,d),
            skill:Mlp::new(2*SKILL_COUNT,32,32,d),item:Mlp::new(2*ITEM_COUNT,32,32,d),
            terrain_first:Conv2dConfig::new([4,16],[3,3]).with_padding(PaddingConfig2d::Same).init(d),
            terrain_second:Conv2dConfig::new([16,16],[3,3]).with_padding(PaddingConfig2d::Same).init(d),
            terrain_linear:linear(64,64),combine:linear(1636-128-30+obs::PREVIOUS,256),memory:Memory::new(d),
            action_embed:EmbeddingConfig::new(13,16).init(d),action_head:linear(256,13),direction_head:linear(272,36),
            enemy_query:linear(272,64),ally_query:linear(272,64),value_first:linear(256,128),value_second:linear(128,1)}
    }
    pub fn encode(&self,b:&Observation)->Encoded {
        let n=b.heroes.dims()[0];assert_eq!(b.items.dims()[1],6);assert_eq!(b.previous.dims()[1],obs::PREVIOUS);
        let heroes=self.hero.forward(Tensor::cat(vec![b.heroes.clone(),b.hero_combat.clone()],2))*b.hero_valid.clone().float().unsqueeze_dim::<3>(2);
        let creeps=self.unit.forward(Tensor::cat(vec![b.creeps.clone(),b.creep_combat.clone()],2));
        let towers=self.unit.forward(Tensor::cat(vec![b.towers.clone(),b.tower_combat.clone()],2));
        let trees=self.tree.forward(b.trees.clone());let projectiles=self.projectile.forward(Tensor::cat(vec![b.projectiles.clone(),b.projectile_combat.clone()],2));
        let terrain=activation::relu(self.terrain_second.forward(activation::relu(self.terrain_first.forward(b.terrain.clone()))));
        let terrain=terrain.reshape([n,16,2,4,2,4]).mean_dim(5).mean_dim(3).reshape([n,64]);
        let enemy=self.item.forward(b.enemy_items.clone())*b.hero_valid.clone().slice([0..n,1..2]).float().reshape([n,1,1]);
        let parts=vec![heroes.reshape([n,256]),towers.reshape([n,256]),
            pool(creeps.clone(),b.creep_valid.clone().bool_and(b.creep_ally.clone())),
            pool(creeps.clone(),b.creep_valid.clone().bool_and(b.creep_ally.clone().bool_not())),
            pool(trees,b.tree_valid.clone()),pool(projectiles,b.projectile_valid.clone()),
            self.skill.forward(b.skills.clone()).reshape([n,128]),self.item.forward(b.items.clone()).reshape([n,192]),enemy.reshape([n,192]),
            activation::relu(self.terrain_linear.forward(terrain)),b.global_features.clone(),b.previous.clone()];
        Encoded{fused:activation::relu(self.combine.forward(Tensor::cat(parts,1))),keys:creeps}
    }
    pub fn value(&self,h:Tensor<2>)->Tensor<1>{self.value_second.forward(activation::relu(self.value_first.forward(h))).squeeze_dim(1)}
    pub fn heads(&self,masks:&Masks,h:Tensor<2>,keys:Tensor<3>,forced:Option<Tensor<2,Int>>,sampler:&crate::sampling::Sampler)->Heads {
        use crate::sampling::distribution;
        let n=h.dims()[0];let targets=keys.dims()[1];let label=|i|forced.as_ref().map(|x|x.clone().slice([0..n,i..i+1]).squeeze_dim(1));
        let action=distribution(self.action_head.forward(h.clone()),masks.types.clone(),label(0),sampler);
        let moving=action.choice.clone().equal_elem(0);let enemy=action.choice.clone().equal_elem(2);let ally=action.choice.clone().equal_elem(3);
        let embedding=self.action_embed.forward(action.choice.clone().unsqueeze_dim::<2>(1)).squeeze_dim(1);
        let condition=Tensor::cat(vec![h.clone(),embedding],1);
        let mut direction=distribution(self.direction_head.forward(condition.clone()),Tensor::<2,Bool>::ones([n,36],&h.device()),label(1),sampler);
        let enemy_mask=Tensor::<2,Bool>::ones([n,targets],&h.device()).mask_where(enemy.clone().unsqueeze_dim::<2>(1),masks.enemy.clone());
        let ally_mask=Tensor::<2,Bool>::ones([n,targets],&h.device()).mask_where(ally.clone().unsqueeze_dim::<2>(1),masks.ally.clone());
        let enemy_logits=(self.enemy_query.forward(condition.clone()).reshape([n,1,64])*keys.clone()).sum_dim(2).reshape([n,targets])/8.;
        let ally_logits=(self.ally_query.forward(condition).reshape([n,1,64])*keys).sum_dim(2).reshape([n,targets])/8.;
        let mut enemy_head=distribution(enemy_logits,enemy_mask,label(2),sampler);let mut ally_head=distribution(ally_logits,ally_mask,label(3),sampler);
        let mw=moving.clone().float();let ew=enemy.clone().float();let aw=ally.clone().float();
        let logp=action.logp+mw.clone()*direction.logp+ew.clone()*enemy_head.logp+aw.clone()*ally_head.logp;
        let entropy=(action.normalized+mw.clone()*direction.normalized+ew.clone()*enemy_head.normalized+aw.clone()*ally_head.normalized)/(1.+mw+ew+aw);
        direction.choice=direction.choice.mask_fill(moving.bool_not(),0);enemy_head.choice=enemy_head.choice.mask_fill(enemy.bool_not(),0);ally_head.choice=ally_head.choice.mask_fill(ally.bool_not(),0);
        Heads{actions:Tensor::stack(vec![action.choice,direction.choice,enemy_head.choice,ally_head.choice],1),logp,entropy,value:self.value(h),
            logits:[action.logits,direction.logits,enemy_head.logits,ally_head.logits]}
    }
}
