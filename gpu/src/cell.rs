//! Fused FP32 LSTM point operations; Burn still owns autodiff and all parameters.
use burn::{backend::{backend_extension,Dispatch,tensor::FloatTensor},tensor::Tensor};
use burn_cubecl::{CubeBackend,tensor::CubeTensor,kernel::into_contiguous};
use cubecl::prelude::*;

#[backend_extension(Autodiff,Cube)]
pub trait CellOps:burn::backend::Backend {
    fn sf_lstm_cell(gates:FloatTensor<Self>,previous:FloatTensor<Self>)->FloatTensor<Self>;
    fn sf_lstm_derivative(gates:FloatTensor<Self>,previous:FloatTensor<Self>,gradient:FloatTensor<Self>)->(FloatTensor<Self>,FloatTensor<Self>);
}

#[cube(launch)]
fn forward(gates:&[f32],previous:&[f32],output:&mut [f32],n:usize){
    let k=ABSOLUTE_POS;
    if k<n*256 {
        let b=k/256;let j=k%256;let base=b*1024+j;
        let i=1.0f32/(1.0f32+(-gates[base]).exp());let f=1.0f32/(1.0f32+(-gates[base+256]).exp());
        let g=gates[base+512].tanh();let o=1.0f32/(1.0f32+(-gates[base+768]).exp());
        let c=f*previous[k]+i*g;
        output[b*512+j]=o*c.tanh();output[b*512+256+j]=c;
    }
}
#[cube(launch)]
fn backward(gates:&[f32],previous:&[f32],gradient:&[f32],dg:&mut [f32],dc:&mut [f32],n:usize){
    let k=ABSOLUTE_POS;
    if k<n*256 {
        let b=k/256;let j=k%256;let base=b*1024+j;
        let i=1.0f32/(1.0f32+(-gates[base]).exp());let f=1.0f32/(1.0f32+(-gates[base+256]).exp());
        let g=gates[base+512].tanh();let o=1.0f32/(1.0f32+(-gates[base+768]).exp());
        let c=f*previous[k]+i*g;let tc=c.tanh();
        let dh=gradient[b*512+j];let carry=gradient[b*512+256+j]+dh*o*(1.0f32-tc*tc);
        dg[base]=carry*g*i*(1.0f32-i);dg[base+256]=carry*previous[k]*f*(1.0f32-f);
        dg[base+512]=carry*i*(1.0f32-g*g);dg[base+768]=dh*tc*o*(1.0f32-o);dc[k]=carry*f;
    }
}

fn output_like(source:&CubeTensor,n:usize,width:usize)->CubeTensor {
    CubeTensor::new_contiguous(source.client.clone(),source.device.clone(),[n,width].into(),source.client.empty(n*width*4),source.dtype)
}
impl CellOps for CubeBackend {
    fn sf_lstm_cell(gates:CubeTensor,previous:CubeTensor)->CubeTensor {
        let gates=into_contiguous(gates);let previous=into_contiguous(previous);let n=gates.meta.shape()[0];
        assert_eq!(gates.dtype,burn::tensor::DType::F32);let out=output_like(&gates,n,512);
        forward::launch(&gates.client,CubeCount::Static(n as u32,1,1),CubeDim::new_1d(256),
            crate::runtime::argument(&gates),crate::runtime::argument(&previous),crate::runtime::argument(&out),n);out
    }
    fn sf_lstm_derivative(gates:CubeTensor,previous:CubeTensor,gradient:CubeTensor)->(CubeTensor,CubeTensor){
        let gates=into_contiguous(gates);let previous=into_contiguous(previous);let gradient=into_contiguous(gradient);
        let n=gates.meta.shape()[0];let dg=output_like(&gates,n,1024);let dc=output_like(&gates,n,256);
        backward::launch(&gates.client,CubeCount::Static(n as u32,1,1),CubeDim::new_1d(256),
            crate::runtime::argument(&gates),crate::runtime::argument(&previous),crate::runtime::argument(&gradient),
            crate::runtime::argument(&dg),crate::runtime::argument(&dc),n);(dg,dc)
    }
}

impl<B:CellOps,C:burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy> CellOps for burn::backend::autodiff::Autodiff<B,C>{
    fn sf_lstm_cell(gates:FloatTensor<Self>,previous:FloatTensor<Self>)->FloatTensor<Self>{
        use burn::backend::autodiff::{ops::{Backward,Ops,OpsKind},grads::Gradients,checkpoint::base::Checkpointer};
        #[derive(Debug)] struct Derivative;
        impl<B:CellOps> Backward<B,2> for Derivative {
            type State=(FloatTensor<B>,FloatTensor<B>);
            fn backward(self,ops:Ops<Self::State,2>,gradients:&mut Gradients,_:&mut Checkpointer){
                let upstream=gradients.consume::<B>(&ops.node);
                let (dg,dc)=B::sf_lstm_derivative(ops.state.0,ops.state.1,upstream);
                if let Some(parent)=ops.parents[0].as_ref(){gradients.register::<B>(parent.id,dg);}
                if let Some(parent)=ops.parents[1].as_ref(){gradients.register::<B>(parent.id,dc);}
            }
        }
        match Derivative.prepare::<C>([gates.node(),previous.node()]).compute_bound().stateful(){
            OpsKind::Tracked(prep)=>{
                let saved=(gates.primitive().clone(),previous.primitive().clone());
                let out=B::sf_lstm_cell(gates.into_primitive(),previous.into_primitive());prep.finish(saved,out)
            }
            OpsKind::UnTracked(prep)=>prep.finish(B::sf_lstm_cell(gates.into_primitive(),previous.into_primitive())),
        }
    }
    fn sf_lstm_derivative(_:FloatTensor<Self>,_:FloatTensor<Self>,_:FloatTensor<Self>)->(FloatTensor<Self>,FloatTensor<Self>){
        panic!("higher-order LSTM derivatives are not supported")
    }
}

pub fn cell(gates:Tensor<2>,previous:Tensor<2>)->(Tensor<2>,Tensor<2>){
    let n=gates.dims()[0];
    let out=Tensor::<2>::from_dispatch(Dispatch::sf_lstm_cell(gates.into_dispatch(),previous.into_dispatch()));
    (out.clone().slice([0..n,0..256]),out.slice([0..n,256..512]))
}
