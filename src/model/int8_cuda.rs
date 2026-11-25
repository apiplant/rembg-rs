//! CUDA INT8 ConvRot linear for Bria's Fibo2 transformer.
//!
//! Comfy-Kitchen uses the same contract: rotate BF16 activations by the
//! checkpoint's regular Hadamard transform, row-quantize to INT8, use an INT8
//! tensor-core GEMM, then apply row and weight scales to the INT32 result.

use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{Arc, Mutex, OnceLock},
};

use anyhow::{Context, Result, bail};
use candle_core::{
    CudaStorage, CustomOp3, DType, Layout, Shape, Tensor,
    cuda_backend::{CudaDType, cudarc},
};
use cudarc::{
    cublas::{CudaBlas, result as cublas_result, sys as cublas_sys},
    driver::{
        CudaContext, CudaModule, CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
    },
    nvrtc,
};
use half::bf16;

const CUDA_SOURCE: &str = include_str!("int8_linear.cu");
const BLOCK: u32 = 256;

static MODULES: OnceLock<Mutex<HashMap<usize, Arc<CudaModule>>>> = OnceLock::new();
static BLAS_HANDLES: OnceLock<Mutex<HashMap<usize, CudaBlas>>> = OnceLock::new();

/// Computes an INT8 ConvRot linear on CUDA and adds its optional BF16 bias.
pub(super) fn linear(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    bias: Option<&Tensor>,
    group: usize,
) -> Result<Tensor> {
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::U8
        || weight_scale.dtype() != DType::F32
    {
        bail!(
            "CUDA INT8 linear expects BF16 input, U8 weight bytes, and F32 scales; got {:?}, {:?}, {:?}",
            input.dtype(),
            weight.dtype(),
            weight_scale.dtype()
        );
    }
    let input = input.force_contiguous()?;
    let weight = weight.force_contiguous()?;
    let weight_scale = weight_scale.force_contiguous()?;
    let (outputs, inputs) = weight.dims2()?;
    if input.dims().last() != Some(&inputs) {
        bail!("INT8 linear input width does not match its weight matrix");
    }
    if weight_scale.elem_count() != outputs {
        bail!(
            "INT8 linear has {} weight scales for {outputs} output rows",
            weight_scale.elem_count()
        );
    }
    if group > 0 && (group != 256 || inputs % group != 0) {
        bail!("CUDA ConvRot currently requires a 256-wide group dividing {inputs}");
    }
    let bias = bias.map(Tensor::force_contiguous).transpose()?;
    let op = Int8LinearOp {
        outputs,
        inputs,
        group,
    };
    let output = input.apply_op3_no_bwd(&weight, &weight_scale, &op)?;
    match bias {
        Some(bias) => Ok(output.broadcast_add(&bias.to_dtype(output.dtype())?)?),
        None => Ok(output),
    }
}

struct Int8LinearOp {
    outputs: usize,
    inputs: usize,
    group: usize,
}

impl CustomOp3 for Int8LinearOp {
    fn name(&self) -> &'static str {
        "fibo2_int8_convrot_linear"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "CUDA INT8 linear received CPU tensors".into(),
        ))
    }

    fn cuda_fwd(
        &self,
        x_storage: &CudaStorage,
        x_layout: &Layout,
        weight_storage: &CudaStorage,
        weight_layout: &Layout,
        scale_storage: &CudaStorage,
        scale_layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        self.forward_cuda(
            x_storage,
            x_layout,
            weight_storage,
            weight_layout,
            scale_storage,
            scale_layout,
        )
        .map_err(|error| candle_core::Error::Msg(format!("{}: {error:#}", self.name())))
    }

    fn metal_fwd(
        &self,
        _: &candle_core::MetalStorage,
        _: &Layout,
        _: &candle_core::MetalStorage,
        _: &Layout,
        _: &candle_core::MetalStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::MetalStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "CUDA INT8 linear received Metal tensors".into(),
        ))
    }
}

impl Int8LinearOp {
    fn forward_cuda(
        &self,
        x_storage: &CudaStorage,
        x_layout: &Layout,
        weight_storage: &CudaStorage,
        weight_layout: &Layout,
        scale_storage: &CudaStorage,
        scale_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        anyhow::ensure!(
            x_layout.is_contiguous() && x_layout.start_offset() == 0,
            "input must be contiguous"
        );
        anyhow::ensure!(
            weight_layout.is_contiguous() && weight_layout.start_offset() == 0,
            "weight must be contiguous"
        );
        anyhow::ensure!(
            scale_layout.is_contiguous() && scale_layout.start_offset() == 0,
            "scales must be contiguous"
        );

        let x = bf16::as_cuda_slice(x_storage).context("reading BF16 CUDA input")?;
        let weight =
            u8::as_cuda_slice(weight_storage).context("reading packed INT8 CUDA weight")?;
        let weight_scale =
            f32::as_cuda_slice(scale_storage).context("reading F32 CUDA weight scales")?;
        let input_elements = x_layout.shape().elem_count();
        let rows = input_elements / self.inputs;
        anyhow::ensure!(rows > 0, "INT8 linear input has no rows");
        let mut output_dims = x_layout.shape().dims().to_vec();
        *output_dims
            .last_mut()
            .context("linear input has no dimensions")? = self.outputs;
        let output_shape = Shape::from(output_dims);
        let output_elements = rows * self.outputs;

        let cuda = &x_storage.device;
        let stream = cuda.cuda_stream();
        let context = stream.context();
        let module = module_for(context).context("loading INT8 CUDA kernels")?;
        let activation = if self.group > 0 {
            quantize_convrot(x, self.inputs, rows, self.group, cuda, &stream, &module)?
        } else {
            quantize_rows(x, self.inputs, rows, cuda, &stream, &module)?
        };

        let mut accum = unsafe { cuda.alloc::<i32>(output_elements) }
            .context("allocating INT8 GEMM accumulator")?;
        let (weight_ptr, _weight_read) = weight.device_ptr(&stream);
        let (activation_ptr, _activation_read) = activation.quantized.device_ptr(&stream);
        {
            let (accum_ptr, _accum_write) = accum.device_ptr_mut(&stream);
            run_int8_gemm(
                context,
                &stream,
                weight_ptr,
                activation_ptr,
                accum_ptr,
                rows,
                self.outputs,
                self.inputs,
            )?;
        }

        let mut output = unsafe { cuda.alloc::<bf16>(output_elements) }
            .context("allocating BF16 INT8 linear output")?;
        let dequant = module
            .load_function("dequantize_int8_rows")
            .context("loading INT8 dequantization kernel")?;
        let cfg = LaunchConfig::for_num_elems(output_elements as u32);
        let rows_i32 = rows as i32;
        let outputs_i32 = self.outputs as i32;
        let has_bias = 0i32;
        let mut builder = stream.launch_builder(&dequant);
        builder
            .arg(&accum)
            .arg(&activation.scales)
            .arg(weight_scale)
            .arg(&0u64)
            .arg(&mut output)
            .arg(&rows_i32)
            .arg(&outputs_i32)
            .arg(&has_bias);
        unsafe { builder.launch(cfg) }.context("launching INT8 dequantization kernel")?;

        Ok((
            <bf16 as CudaDType>::wrap_cuda_slice(output, cuda.clone()),
            output_shape,
        ))
    }
}

struct Activation {
    quantized: CudaSlice<i8>,
    scales: CudaSlice<f32>,
}

fn quantize_rows(
    input: &CudaSlice<bf16>,
    inputs: usize,
    rows: usize,
    cuda: &candle_core::CudaDevice,
    stream: &Arc<cudarc::driver::CudaStream>,
    module: &Arc<CudaModule>,
) -> Result<Activation> {
    let mut quantized = unsafe { cuda.alloc::<i8>(rows * inputs) }
        .context("allocating row-wise INT8 activations")?;
    let mut scales =
        unsafe { cuda.alloc::<f32>(rows) }.context("allocating row-wise activation scales")?;
    let kernel = module
        .load_function("quantize_bf16_rows")
        .context("loading BF16 row quantization kernel")?;
    let inputs_i32 = inputs as i32;
    let mut builder = stream.launch_builder(&kernel);
    builder
        .arg(input)
        .arg(&mut quantized)
        .arg(&mut scales)
        .arg(&inputs_i32);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .context("launching BF16 row quantization kernel")?;
    Ok(Activation { quantized, scales })
}

fn quantize_convrot(
    input: &CudaSlice<bf16>,
    inputs: usize,
    rows: usize,
    group: usize,
    cuda: &candle_core::CudaDevice,
    stream: &Arc<cudarc::driver::CudaStream>,
    module: &Arc<CudaModule>,
) -> Result<Activation> {
    let groups = inputs / group;
    let mut rotated = unsafe { cuda.alloc::<f32>(rows * inputs) }
        .context("allocating ConvRot activation scratch")?;
    let mut group_max =
        unsafe { cuda.alloc::<f32>(rows * groups) }.context("allocating ConvRot group maxima")?;
    let mut quantized = unsafe { cuda.alloc::<i8>(rows * inputs) }
        .context("allocating ConvRot INT8 activations")?;
    let mut scales =
        unsafe { cuda.alloc::<f32>(rows) }.context("allocating ConvRot activation scales")?;

    let rotate = module
        .load_function("convrot_groups")
        .context("loading ConvRot group kernel")?;
    let inputs_i32 = inputs as i32;
    let groups_i32 = groups as i32;
    let mut rotate_builder = stream.launch_builder(&rotate);
    rotate_builder
        .arg(input)
        .arg(&mut rotated)
        .arg(&mut group_max)
        .arg(&inputs_i32)
        .arg(&groups_i32);
    unsafe {
        rotate_builder.launch(LaunchConfig {
            grid_dim: ((rows * groups) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .context("launching ConvRot group rotation kernel")?;

    let quantize = module
        .load_function("quantize_rotated_rows")
        .context("loading ConvRot row quantization kernel")?;
    let mut quantize_builder = stream.launch_builder(&quantize);
    quantize_builder
        .arg(&rotated)
        .arg(&group_max)
        .arg(&mut quantized)
        .arg(&mut scales)
        .arg(&inputs_i32)
        .arg(&groups_i32);
    unsafe {
        quantize_builder.launch(LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .context("launching ConvRot row quantization kernel")?;

    Ok(Activation { quantized, scales })
}

fn module_for(context: &Arc<CudaContext>) -> Result<Arc<CudaModule>> {
    let modules = MODULES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut modules = modules
        .lock()
        .map_err(|_| anyhow::anyhow!("CUDA module cache poisoned"))?;
    if let Some(module) = modules.get(&context.ordinal()) {
        return Ok(module.clone());
    }
    let ptx =
        nvrtc::compile_ptx(CUDA_SOURCE).context("compiling ConvRot CUDA kernels with NVRTC")?;
    let module = context
        .load_module(ptx)
        .context("loading ConvRot CUDA PTX module")?;
    modules.insert(context.ordinal(), module.clone());
    Ok(module)
}

#[allow(clippy::too_many_arguments)]
fn run_int8_gemm(
    context: &Arc<CudaContext>,
    stream: &Arc<cudarc::driver::CudaStream>,
    weight: cudarc::driver::sys::CUdeviceptr,
    activation: cudarc::driver::sys::CUdeviceptr,
    output: cudarc::driver::sys::CUdeviceptr,
    rows: usize,
    outputs: usize,
    inputs: usize,
) -> Result<()> {
    let handles = BLAS_HANDLES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut handles = handles
        .lock()
        .map_err(|_| anyhow::anyhow!("cuBLAS handle cache poisoned"))?;
    if !handles.contains_key(&context.ordinal()) {
        let handle = CudaBlas::new(stream.clone()).context("creating cuBLAS INT8 handle")?;
        handles.insert(context.ordinal(), handle);
    }
    let handle = handles
        .get(&context.ordinal())
        .context("cuBLAS handle missing")?;
    let alpha = 1i32;
    let beta = 0i32;
    // Row-major [N,K] weights and [M,K] activations are interpreted as
    // column-major [K,N] and [K,M], producing column-major [N,M] output.
    unsafe {
        cublas_result::gemm_ex(
            *handle.handle(),
            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            outputs as i32,
            rows as i32,
            inputs as i32,
            (&alpha as *const i32).cast::<c_void>(),
            weight as usize as *const c_void,
            cublas_sys::cudaDataType::CUDA_R_8I,
            inputs as i32,
            activation as usize as *const c_void,
            cublas_sys::cudaDataType::CUDA_R_8I,
            inputs as i32,
            (&beta as *const i32).cast::<c_void>(),
            output as usize as *mut c_void,
            cublas_sys::cudaDataType::CUDA_R_32I,
            outputs as i32,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
    .context("running cuBLAS INT8 tensor-core GEMM")?;
    Ok(())
}
