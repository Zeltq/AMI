use burn::{tensor::{Device, DeviceKind, Tensor}};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::prelude::*;

pub fn device() -> Device {
    #[cfg(feature = "hip")]
    { Device::rocm(0) }
    #[cfg(not(feature = "hip"))]
    { Device::vulkan(DeviceKind::DiscreteGpu(0)) }
}

pub fn file_sha256(path:&std::path::Path)->anyhow::Result<String>{
    use sha2::{Digest,Sha256};use std::io::Read;
    let mut file=std::fs::File::open(path)?;let mut hash=Sha256::new();let mut bytes=[0u8;65536];
    loop {let n=file.read(&mut bytes)?;if n==0 {break;}hash.update(&bytes[..n]);}Ok(format!("{:x}",hash.finalize()))
}
pub fn bytes_sha256(bytes:&[u8])->String{use sha2::{Digest,Sha256};format!("{:x}",Sha256::digest(bytes))}

pub fn profile_value(profile:burn::tensor::ProfileDuration)->serde_json::Value{
    let method=profile.timing_method().to_string();
    let seconds=burn_std::future::block_on(profile.resolve()).map(|ticks|ticks.duration().as_secs_f64());
    serde_json::json!({"method":method,"seconds":seconds})
}

pub fn primitive<const D: usize>(t: Tensor<D>) -> CubeTensor {
    burn_cubecl::kernel::into_contiguous(t.try_into_primitive::<CubeBackend>().expect("plain CubeCL tensor required"))
}

pub fn tensor<const D: usize>(t: CubeTensor) -> Tensor<D> {
    Tensor::from_primitive::<CubeBackend>(t)
}

// The simulator only consumes detached, contiguous FP32 tensors on this client.
// No host staging or CPU fallback exists in this boundary.
pub fn argument(t: &CubeTensor) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(t.handle.clone(), 1) }
}

#[cube(launch)]
fn increment(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0;
    }
}

pub fn qualification() -> anyhow::Result<serde_json::Value> {
    use burn::{nn::LinearConfig, optim::{AdamConfig, GradientsParams}};
    let device = device();
    let data = Tensor::<2>::ones([200, 256], &device);
    let source = primitive(data);
    let target = primitive(Tensor::<2>::zeros([200, 256], &device));
    increment::launch(&source.client, CubeCount::Static(200, 1, 1), CubeDim::new_1d(256),
        argument(&source), argument(&target));
    let observed = tensor::<2>(target);
    let kernel_mean: f32 = observed.clone().mean().into_scalar::<f32>();
    anyhow::ensure!((kernel_mean - 2.).abs() < 1e-6, "CubeCL/Burn interoperability failed");
    let mut model = LinearConfig::new(256, 11).init(&device.clone().autodiff());
    let before = model.weight.val().detach().sum().into_scalar::<f32>();
    let mut optimizer = AdamConfig::new().with_epsilon(1e-8).init();
    let input = observed.autodiff();
    let loss = model.forward(input).square().mean();
    let loss_value = loss.clone().detach().into_scalar::<f32>();
    let gradients = GradientsParams::from_grads(loss.backward(), &model);
    model = optimizer.step(1e-4, model, gradients);
    device.sync()?;
    let after = model.weight.val().detach().sum().into_scalar::<f32>();
    anyhow::ensure!(loss_value.is_finite() && before != after, "optimizer did not update weights");
    Ok(serde_json::json!({"kernel_mean":kernel_mean,"loss":loss_value,
        "weight_sum_before":before,"weight_sum_after":after,"device":format!("{device:?}"),
        "gpu_kernel_inference_backward_adam":true}))
}

