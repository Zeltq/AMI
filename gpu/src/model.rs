use burn::{module::{Module, Param}, nn::{Linear, LinearConfig, Embedding, EmbeddingConfig,
    conv::{Conv2d, Conv2dConfig}, PaddingConfig2d}, tensor::{Tensor, Bool, Int, Device, activation}};
use crate::contract::*;

#[derive(Clone, Debug)]
pub struct Observation {
    pub combat_profile:bool,
    pub heroes: Tensor<3>, pub creeps: Tensor<3>, pub towers: Tensor<3>,
    pub hero_combat:Tensor<3>,pub creep_combat:Tensor<3>,pub tower_combat:Tensor<3>,pub projectile_combat:Tensor<3>,
    pub trees: Tensor<3>, pub projectiles: Tensor<3>, pub skills: Tensor<3>,
    pub items: Tensor<3>, pub enemy_items: Tensor<3>, pub terrain: Tensor<4>,
    pub global_features: Tensor<2>, pub previous: Tensor<2>,
    pub hero_valid: Tensor<2, Bool>, pub creep_valid: Tensor<2, Bool>,
    pub creep_ally: Tensor<2, Bool>, pub tree_valid: Tensor<2, Bool>,
    pub projectile_valid: Tensor<2, Bool>,
    pub action_mask: Tensor<2, Bool>, pub buy_mask: Tensor<2, Bool>,
    pub item_mask: Tensor<2, Bool>, pub attack_mask: Tensor<2, Bool>,
    pub tree_mask: Tensor<2, Bool>, pub tp_mask: Tensor<2, Bool>, pub self_mask: Tensor<2, Bool>,
}

#[derive(Module, Debug)]
pub struct Mlp { pub first: Linear, pub second: Linear }
impl Mlp {
    fn new(i: usize, h: usize, o: usize, d: &Device) -> Self {
        Self { first: LinearConfig::new(i,h).init(d), second: LinearConfig::new(h,o).init(d) }
    }
    fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        activation::relu(self.second.forward(activation::relu(self.first.forward(x))))
    }
}

// PyTorch gate order i,f,g,o is retained. Two fused projections avoid eight
// small matrix multiplications per step and preserve both bias vectors.
#[derive(Module, Debug)]
pub struct Memory { pub input: Linear, pub hidden: Linear }
impl Memory {
    fn new(d: &Device) -> Self {
        Self { input: LinearConfig::new(256,1024).init(d), hidden: LinearConfig::new(256,1024).init(d) }
    }
    pub fn step(&self, x: Tensor<2>, h: Tensor<2>, c: Tensor<2>) -> (Tensor<2>,Tensor<2>) {
        self.projected(self.input.forward(x),h,c)
    }
    pub fn projected(&self,input:Tensor<2>,h:Tensor<2>,c:Tensor<2>)->(Tensor<2>,Tensor<2>){
        crate::cell::cell(input+self.hidden.forward(h),c)
    }
    #[allow(dead_code)]
    pub fn reference(&self, x: Tensor<2>, h: Tensor<2>, c: Tensor<2>) -> (Tensor<2>,Tensor<2>) {
        let b=x.dims()[0];
        let gates=self.input.forward(x)+self.hidden.forward(h);
        let i=activation::sigmoid(gates.clone().slice([0..b,0..256]));
        let f=activation::sigmoid(gates.clone().slice([0..b,256..512]));
        let g=gates.clone().slice([0..b,512..768]).tanh();
        let o=activation::sigmoid(gates.slice([0..b,768..1024]));
        let c=f*c+i*g; let h=o*c.clone().tanh(); (h,c)
    }
}

#[derive(Module, Debug)]
pub struct Policy {
    pub hero: Mlp, pub unit: Mlp, pub tree: Mlp, pub projectile: Mlp,
    pub skill: Mlp, pub item: Mlp,
    pub terrain_first: Conv2d, pub terrain_second: Conv2d, pub terrain_linear: Linear,
    pub combine: Linear, pub memory: Memory,
    pub action_embed: Embedding, pub item_embed: Embedding,
    pub action_head: Linear, pub buy_head: Linear, pub item_head: Linear,
    pub x_head: Linear, pub y_head: Linear, pub query: Linear,
    pub hero_key: Linear, pub tree_key: Linear, pub base_key: Param<Tensor<3>>,
    pub value_first: Linear, pub value_second: Linear,
}

pub struct Encoded { pub fused: Tensor<2>, pub keys: Tensor<3> }
pub struct Heads {
    pub actions: Tensor<2,Int>, pub logp: Tensor<1>, pub entropy: Tensor<1>, pub value: Tensor<1>,
    pub logits:[Tensor<2>;6],
}

pub fn pool(x: Tensor<3>, mask: Tensor<2,Bool>) -> Tensor<2> {
    let w=mask.clone().float().unsqueeze_dim::<3>(2);
    let mean=(x.clone()*w.clone()).sum_dim(1)/w.sum_dim(1).clamp_min(1.);
    let maximum=x.mask_fill(mask.clone().bool_not().unsqueeze_dim::<3>(2), f32::MIN).max_dim(1)
        .mask_fill(mask.any_dim(1).bool_not().unsqueeze_dim::<3>(2),0.);
    Tensor::cat(vec![mean.squeeze_dim(1),maximum.squeeze_dim(1)],1)
}

impl Policy {
    /// GGD adds inputs to the existing encoders; every downstream layer keeps
    /// its original size. Legacy MIA construction/import remains unchanged.
    pub fn new_ggd(d:&Device)->Self {
        let mut policy=Self::new(d);
        policy.hero.first=LinearConfig::new(2*(UNIT_COUNT+COMBAT_UNIT_COUNT),128).init(d);
        policy.unit.first=LinearConfig::new(2*(UNIT_COUNT+COMBAT_UNIT_COUNT),64).init(d);
        policy.projectile.first=LinearConfig::new(2*(PROJECTILE_COUNT+COMBAT_PROJECTILE_COUNT),32).init(d);
        policy
    }
    pub fn is_ggd(&self)->bool {self.hero.first.weight.val().dims()[0]==2*(UNIT_COUNT+COMBAT_UNIT_COUNT)}
    pub fn new(d: &Device) -> Self {
        let linear=|i,o| LinearConfig::new(i,o).init(d);
        Self {
            hero:Mlp::new(2*UNIT_COUNT,128,128,d), unit:Mlp::new(2*UNIT_COUNT,64,64,d),
            tree:Mlp::new(2*TREE_COUNT,32,32,d), projectile:Mlp::new(2*PROJECTILE_COUNT,32,32,d),
            skill:Mlp::new(2*SKILL_COUNT,32,32,d), item:Mlp::new(2*ITEM_COUNT,32,32,d),
            terrain_first:Conv2dConfig::new([4,16],[3,3]).with_padding(PaddingConfig2d::Same).init(d),
            terrain_second:Conv2dConfig::new([16,16],[3,3]).with_padding(PaddingConfig2d::Same).init(d),
            terrain_linear:linear(64,64), combine:linear(1636,256), memory:Memory::new(d),
            action_embed:EmbeddingConfig::new(11,16).init(d), item_embed:EmbeddingConfig::new(8,16).init(d),
            action_head:linear(256,11),buy_head:linear(256,7),item_head:linear(272,8),
            x_head:linear(288,129),y_head:linear(288,129),query:linear(288,64),
            hero_key:linear(128,64),tree_key:linear(32,64),base_key:Param::from_tensor(Tensor::zeros([1,1,64],d)),
            value_first:linear(256,128),value_second:linear(128,1),
        }
    }

    pub fn encode(&self,b:&Observation) -> Encoded {
        let n=b.heroes.dims()[0];
        let ggd=self.is_ggd();
        assert!(!ggd||b.combat_profile,"GGD requires ggd-v2 observations; legacy MIA inputs are incomplete");
        let extend=|base:Tensor<3>,extra:Tensor<3>|if ggd{Tensor::cat(vec![base,extra],2)}else{base};
        let heroes=self.hero.forward(extend(b.heroes.clone(),b.hero_combat.clone()))*b.hero_valid.clone().float().unsqueeze_dim::<3>(2);
        let creeps=self.unit.forward(extend(b.creeps.clone(),b.creep_combat.clone()));let towers=self.unit.forward(extend(b.towers.clone(),b.tower_combat.clone()));
        let trees=self.tree.forward(b.trees.clone());let projectiles=self.projectile.forward(extend(b.projectiles.clone(),b.projectile_combat.clone()));
        let terrain=activation::relu(self.terrain_second.forward(activation::relu(self.terrain_first.forward(b.terrain.clone()))));
        // AdaptiveAvgPool(2,2) on an 8x8 tensor is exactly four 4x4 means.
        let terrain=terrain.reshape([n,16,2,4,2,4]).mean_dim(5).mean_dim(3).reshape([n,64]);
        let enemy=self.item.forward(b.enemy_items.clone())*b.hero_valid.clone().slice([0..n,1..2]).float().reshape([n,1,1]);
        let parts=vec![heroes.clone().reshape([n,256]),towers.clone().reshape([n,256]),
            pool(creeps.clone(),b.creep_valid.clone().bool_and(b.creep_ally.clone())),
            pool(creeps.clone(),b.creep_valid.clone().bool_and(b.creep_ally.clone().bool_not())),
            pool(trees.clone(),b.tree_valid.clone()),pool(projectiles,b.projectile_valid.clone()),
            self.skill.forward(b.skills.clone()).reshape([n,128]),self.item.forward(b.items.clone()).reshape([n,256]),
            enemy.reshape([n,256]),activation::relu(self.terrain_linear.forward(terrain)),b.global_features.clone(),b.previous.clone()];
        Encoded { fused:activation::relu(self.combine.forward(Tensor::cat(parts,1))),
            keys:Tensor::cat(vec![self.hero_key.forward(heroes),creeps,towers,self.tree_key.forward(trees),
                self.base_key.val().expand([n,1,64])],1) }
    }

    pub fn value(&self,h:Tensor<2>) -> Tensor<1> {
        self.value_second.forward(activation::relu(self.value_first.forward(h))).squeeze_dim(1)
    }

    pub fn heads(&self,b:&Observation,h:Tensor<2>,keys:Tensor<3>,forced:Option<Tensor<2,Int>>,
        sampler:&crate::sampling::Sampler) -> Heads {
        use crate::sampling::distribution;
        let n=h.dims()[0]; let targets=keys.dims()[1];
        let label=|col:usize|forced.as_ref().map(|x|x.clone().slice([0..n,col..col+1]).squeeze_dim(1));
        let action=distribution(self.action_head.forward(h.clone()),b.action_mask.clone(),label(0),sampler);
        let buy=distribution(self.buy_head.forward(h.clone()),b.buy_mask.clone(),label(1),sampler);
        let is_item=action.choice.clone().equal_elem(10); let item_weights=is_item.clone().float();
        let embedding=self.action_embed.forward(action.choice.clone().unsqueeze_dim::<2>(1)).squeeze_dim(1);
        let item_logits=self.item_head.forward(Tensor::cat(vec![h.clone(),embedding.clone()],1));
        let item_mask=Tensor::<2,Bool>::ones([n,ITEMS],&h.device()).mask_where(is_item.clone().unsqueeze_dim::<2>(1),b.item_mask.clone());
        let mut item=distribution(item_logits,item_mask,label(2),sampler);
        item.choice=item.choice.mask_fill(is_item.clone().bool_not(),0);
        let condition=Tensor::cat(vec![h.clone(),embedding,self.item_embed.forward(item.choice.clone().unsqueeze_dim::<2>(1)).squeeze_dim(1)],1);
        let attack=action.choice.clone().equal_elem(3);
        let tp=is_item.clone().bool_and(item.choice.clone().equal_elem(3));
        let tango=is_item.clone().bool_and(item.choice.clone().equal_elem(6));
        let target_mask=b.self_mask.clone().mask_where(attack.clone().unsqueeze_dim::<2>(1),b.attack_mask.clone())
            .mask_where(tp.clone().unsqueeze_dim::<2>(1),b.tp_mask.clone())
            .mask_where(tango.clone().unsqueeze_dim::<2>(1),b.tree_mask.clone());
        let target_logits=(self.query.forward(condition.clone()).reshape([n,1,64])*keys).sum_dim(2).reshape([n,targets])/8.;
        let mut target=distribution(target_logits,target_mask,label(3),sampler);
        let coordinate_mask=Tensor::<2,Bool>::ones([n,COORDS],&h.device());
        let mut x=distribution(self.x_head.forward(condition.clone()),coordinate_mask.clone(),label(4),sampler);
        let mut y=distribution(self.y_head.forward(condition),coordinate_mask,label(5),sampler);
        let spatial=action.choice.clone().equal_elem(1).bool_or(action.choice.clone().equal_elem(2)).bool_or(tp.clone());
        let targeted=attack.bool_or(tp).bool_or(tango);
        let sw=spatial.clone().float();let tw=targeted.clone().float();
        let logp=action.logp+buy.logp+item_weights.clone()*item.logp+tw.clone()*target.logp+sw.clone()*(x.logp+y.logp);
        let entropy=(action.normalized+buy.normalized+item_weights.clone()*item.normalized+tw.clone()*target.normalized+
            sw.clone()*(x.normalized+y.normalized))/(2.+item_weights+tw+sw*2.);
        x.choice=x.choice.mask_fill(spatial.clone().bool_not(),64);y.choice=y.choice.mask_fill(spatial.bool_not(),64);
        target.choice=target.choice.mask_fill(targeted.bool_not(),0);
        Heads {actions:Tensor::stack(vec![action.choice,buy.choice,item.choice,target.choice,x.choice,y.choice],1),
            logp,entropy,value:self.value(h),logits:[action.logits,buy.logits,item.logits,target.logits,x.logits,y.logits]}
    }
}
