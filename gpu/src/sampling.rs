use burn::tensor::{Tensor, Int, Bool, Device, activation};
use cubecl::{prelude::*, server::Handle};
use crate::runtime::{primitive,tensor,argument};

pub struct Sampler { pub state: Handle, pub client: Client, pub actors: usize, argmax:bool }

#[cube(launch)]
fn greedy(logits:&[f32],output:&mut [f32],width:usize){
    let a=ABSOLUTE_POS;
    if a<output.len(){
        let mut best=0usize;let mut maximum=logits[a*width];
        for i in 1..width {if logits[a*width+i]>maximum {maximum=logits[a*width+i];best=i;}}
        output[a]=best as f32;
    }
}

#[cube(launch)]
fn categorical(logits: &[f32], rng: &mut [u32], output: &mut [f32], width: usize) {
    let actor=ABSOLUTE_POS;
    if actor < output.len() {
        let mut seed=rng[actor];
        seed=seed^(seed<<13); seed=seed^(seed>>17); seed=seed^(seed<<5);
        rng[actor]=seed;
        let uniform=(seed>>8) as f32 / 16777216.0;
        let start=actor*width;
        let mut maximum=logits[start];
        for i in 0..width { maximum=f32::max(maximum,logits[start+i]); }
        let mut total=0.0f32;
        for i in 0..width { total+=(logits[start+i]-maximum).exp(); }
        let threshold=uniform*total;
        let mut cumulative=0.0f32; let mut selected=width-1; let mut found=false;
        for i in 0..width {
            cumulative+=(logits[start+i]-maximum).exp();
            if !found && cumulative>threshold { selected=i; found=true; }
        }
        output[actor]=selected as f32;
    }
}

impl Sampler {
    pub fn new(n:usize,seed:u32,d:&Device) -> Self {
        let p=primitive(Tensor::<1>::zeros([n],d));
        let seeds:Vec<u32>=(0..n).map(|i| seed.wrapping_add((i as u32+1).wrapping_mul(2654435761))|1).collect();
        Self { state:p.client.create_from_slice(&seeds.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>()),client:p.client,actors:n,argmax:false }
    }
    pub fn with_argmax(mut self,enabled:bool)->Self {self.argmax=enabled;self}
    pub fn draw(&self,logits:Tensor<2>) -> Tensor<1,Int> {
        let [n,width]=logits.dims(); assert_eq!(n,self.actors);
        let logits=primitive(logits.without_autodiff());
        let output=primitive(Tensor::<1>::zeros([n], &Tensor::<2>::from_primitive::<burn_cubecl::CubeBackend>(logits.clone()).device()));
        if self.argmax {greedy::launch(&self.client,CubeCount::Static(n.div_ceil(64)as u32,1,1),CubeDim::new_1d(64),argument(&logits),argument(&output),width);}
        else {categorical::launch(&self.client,CubeCount::Static(n.div_ceil(64) as u32,1,1),CubeDim::new_1d(64),
            argument(&logits),unsafe {BufferArg::from_raw_parts(self.state.clone(),1)},argument(&output),width);}
        tensor::<1>(output).int()
    }
}

pub struct Distribution {
    pub choice:Tensor<1,Int>, pub logp:Tensor<1>, pub normalized:Tensor<1>,
    pub logits:Tensor<2>,
}

pub fn distribution(logits:Tensor<2>,mask:Tensor<2,Bool>,forced:Option<Tensor<1,Int>>,sampler:&Sampler) -> Distribution {
    let masked=logits.clone().mask_fill(mask.clone().bool_not(),-1e30);
    let log_probs=activation::log_softmax(masked.clone(),1);
    let choice=forced.unwrap_or_else(||sampler.draw(masked));
    let logp=log_probs.clone().gather(1,choice.clone().unsqueeze_dim::<2>(1)).squeeze_dim(1);
    let entropy=-(log_probs.clone().exp()*log_probs).sum_dim(1).squeeze_dim(1);
    let count=mask.float().sum_dim(1).squeeze_dim(1);
    let normalized=(entropy.clone()/count.clone().clamp_min(2.).log()).mask_fill(count.lower_equal_elem(1.),0.);
    Distribution {choice,logp,normalized,logits}
}


