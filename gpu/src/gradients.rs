//! Global norm clipping, matching torch.nn.utils.clip_grad_norm_.
use burn::{module::{Module,ModuleVisitor,Param},optim::GradientsParams,tensor::Tensor};

pub fn clip<M:Module>(model:&M,grads:&mut GradientsParams,limit:f32)->Tensor<1>{
    struct Norm<'a>{grads:&'a GradientsParams,terms:Vec<Tensor<1>>}
    impl ModuleVisitor for Norm<'_>{
        fn visit_float<const D:usize>(&mut self,p:&Param<Tensor<D>>){
            if let Some(g)=self.grads.get::<D>(p.id){self.terms.push(g.square().sum());}
        }
    }
    let mut visitor=Norm{grads,terms:vec![]};model.visit(&mut visitor);
    assert!(!visitor.terms.is_empty(),"no model gradients");
    let norm=Tensor::cat(visitor.terms,0).sum().sqrt();
    let scale=(limit/(norm.clone()+1e-6)).clamp_max(1.0);
    struct Apply<'a>{grads:&'a mut GradientsParams,scale:Tensor<1>}
    impl ModuleVisitor for Apply<'_>{
        fn visit_float<const D:usize>(&mut self,p:&Param<Tensor<D>>){
            if let Some(g)=self.grads.remove::<D>(p.id){self.grads.register(p.id,g*self.scale.clone().reshape([1;D]));}
        }
    }
    model.visit(&mut Apply{grads,scale});norm
}
