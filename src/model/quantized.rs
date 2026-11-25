//! Loader and Candle linear operation for Bria's `int8_tensorwise` ConvRot weights.
//!
//! ComfyUI stores signed INT8 bytes, one FP32 scale per output row, and a JSON
//! marker beside each quantized linear. This module loads those tensors without
//! expanding the whole checkpoint. CPU and Metal paths dequantize before the
//! matmul; the CUDA Fibo2 path consumes packed weights with INT8 GEMM.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{
    DType, Device, Shape, Tensor,
    safetensors::{Load, MmapedSafetensors},
};
use candle_nn::{Init, VarBuilder, var_builder::SimpleBackend};
use serde::Deserialize;

pub struct Checkpoint {
    tensors: MmapedSafetensors,
}

impl Checkpoint {
    /// Opens one or more sharded safetensors files as memory maps.
    pub fn open<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        // SAFETY: Candle owns each mmap for the lifetime of `MmapedSafetensors`.
        let tensors =
            unsafe { MmapedSafetensors::multi(paths) }.context("opening safetensors checkpoint")?;
        Ok(Self { tensors })
    }

    pub fn tensor(&self, name: &str, device: &Device) -> Result<Tensor> {
        let view = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        view.load(device)
            .with_context(|| format!("loading tensor {name}"))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.get(name).is_ok()
    }

    /// Creates a Candle `VarBuilder` that transparently expands Bria's
    /// quantized matrices as each model layer is constructed.
    pub fn into_var_builder(self, dtype: DType, device: &Device) -> VarBuilder<'static> {
        VarBuilder::from_backend(
            Box::new(QuantizedBackend { checkpoint: self }),
            dtype,
            device.clone(),
        )
    }

    pub fn quantized_linear(&self, prefix: &str, device: &Device) -> Result<Linear> {
        let weight_name = format!("{prefix}.weight");
        let view = self
            .tensors
            .get(&weight_name)
            .with_context(|| format!("missing {weight_name}"))?;
        let bias = self
            .tensors
            .get(&format!("{prefix}.bias"))
            .ok()
            .map(|v| v.load(device))
            .transpose()?;

        if view.dtype() != safetensors::Dtype::I8 {
            return Ok(Linear::Dense {
                weight: view.load(device)?,
                bias,
            });
        }

        let shape = view.shape().to_vec();
        if shape.len() != 2 {
            bail!("{weight_name} must be a matrix, got {shape:?}");
        }
        let quant = Tensor::from_raw_buffer(view.data(), DType::U8, &shape, device)?;
        let scale_name = format!("{prefix}.weight_scale");
        let scale_view = self
            .tensors
            .get(&scale_name)
            .with_context(|| format!("missing {scale_name}"))?;
        let scale = scale_view.load(device)?.to_dtype(DType::F32)?;
        let marker_name = format!("{prefix}.comfy_quant");
        let marker = self
            .tensors
            .get(&marker_name)
            .with_context(|| format!("missing {marker_name}"))?;
        let metadata: QuantMarker = serde_json::from_slice(marker.data())
            .with_context(|| format!("invalid quantization marker {marker_name}"))?;
        if metadata.format != "int8_tensorwise" {
            bail!(
                "unsupported quantization format {} in {marker_name}",
                metadata.format
            );
        }
        let group = if metadata.convrot {
            metadata.convrot_groupsize.unwrap_or(0)
        } else {
            0
        };
        if group > 0 && shape[1] % group != 0 {
            bail!(
                "ConvRot group {group} does not divide {weight_name} input width {}",
                shape[1]
            );
        }
        Ok(Linear::Int8ConvRot {
            weight: quant,
            scale,
            bias,
            group,
        })
    }
}

struct QuantizedBackend {
    checkpoint: Checkpoint,
}

impl QuantizedBackend {
    fn load(&self, name: &str, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        let result = (|| -> Result<Tensor> {
            let weight_prefix = name.strip_suffix(".weight");
            if let Some(prefix) = weight_prefix
                && self.checkpoint.contains(&format!("{prefix}.comfy_quant"))
            {
                let linear = self
                    .checkpoint
                    .quantized_linear(prefix, device)?
                    .dequantize(dtype)?;
                if let Linear::Dense { weight, .. } = linear {
                    return Ok(weight);
                }
            }
            Ok(self.checkpoint.tensor(name, device)?.to_dtype(dtype)?)
        })();
        result.map_err(|error| candle_core::Error::Msg(error.to_string()))
    }
}

impl SimpleBackend for QuantizedBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _hints: Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.load(name, dtype, device)?;
        if tensor.dims() != shape.dims() {
            return Err(candle_core::Error::Msg(format!(
                "tensor {name} has shape {:?}, requested {:?}",
                tensor.dims(),
                shape.dims()
            )));
        }
        Ok(tensor)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        self.load(name, dtype, device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.checkpoint.contains(name)
    }
}

#[derive(Deserialize)]
struct QuantMarker {
    format: String,
    #[serde(default)]
    convrot: bool,
    convrot_groupsize: Option<usize>,
}

pub enum Linear {
    Dense {
        weight: Tensor,
        bias: Option<Tensor>,
    },
    Int8ConvRot {
        weight: Tensor,
        scale: Tensor,
        bias: Option<Tensor>,
        group: usize,
    },
}

impl Linear {
    /// Keeps ConvRot matrices packed for the CUDA INT8 path and casts only
    /// non-quantized tensors such as biases to the activation dtype.
    pub fn keep_quantized(self, dtype: DType) -> Result<Self> {
        match self {
            Self::Dense { weight, bias } => Ok(Self::Dense {
                weight: weight.to_dtype(dtype)?,
                bias: bias.map(|v| v.to_dtype(dtype)).transpose()?,
            }),
            Self::Int8ConvRot {
                weight,
                scale,
                bias,
                group,
            } => Ok(Self::Int8ConvRot {
                weight,
                scale,
                bias: bias.map(|v| v.to_dtype(dtype)).transpose()?,
                group,
            }),
        }
    }

    /// Expands one quantized layer into the requested compute dtype.
    /// Call this while loading the model to retain BF16 weights in the target
    /// device memory and avoid repeating the ConvRot inverse each denoising step.
    pub fn dequantize(self, dtype: DType) -> Result<Self> {
        match self {
            Self::Dense { weight, bias } => Ok(Self::Dense {
                weight: weight.to_dtype(dtype)?,
                bias,
            }),
            Self::Int8ConvRot {
                weight,
                scale,
                bias,
                group,
            } => {
                let (out, _input) = weight.dims2()?;
                let unsigned = weight.to_dtype(DType::F32)?;
                let negative = unsigned.ge(128f64)?.to_dtype(DType::F32)?;
                let signed = unsigned.broadcast_sub(&negative.affine(256.0, 0.0)?)?;
                let mut dequant = signed.broadcast_mul(&scale.reshape((out, 1))?)?;
                if group > 0 {
                    dequant = hadamard_rows(&dequant, group)?;
                }
                Ok(Self::Dense {
                    weight: dequant.to_dtype(dtype)?,
                    bias: bias.map(|v| v.to_dtype(dtype)).transpose()?,
                })
            }
        }
    }

    /// Applies `x Wᵀ + b`. CUDA Fibo2 keeps ConvRot weights packed; other
    /// devices use the dense dequantized fallback.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Self::Int8ConvRot {
            weight,
            scale,
            bias,
            group,
        } = self
            && xs.device().is_cuda()
        {
            return super::int8_cuda::linear(xs, weight, scale, bias.as_ref(), *group)
                .map_err(Into::into);
        }

        let (weight, bias) = match self {
            Self::Dense { weight, bias } => (weight.clone(), bias.as_ref()),
            Self::Int8ConvRot {
                weight,
                scale,
                bias,
                group,
            } => {
                let (out, _input) = weight.dims2()?;
                // Safetensors stores signed INT8 bits. Candle has U8 storage, so
                // correct the upper half after conversion without a CPU copy.
                let unsigned = weight.to_dtype(DType::F32)?;
                let negative = unsigned.ge(128f64)?.to_dtype(DType::F32)?;
                let signed = unsigned.broadcast_sub(&negative.affine(256.0, 0.0)?)?;
                let mut dequant = signed.broadcast_mul(&scale.reshape((out, 1))?)?;
                if *group > 0 {
                    dequant = hadamard_rows(&dequant, *group)?;
                }
                (dequant, bias.as_ref())
            }
        };
        let output = xs.broadcast_matmul(&weight.transpose(0, 1)?.to_dtype(xs.dtype())?)?;
        match bias {
            Some(bias) => Ok(output.broadcast_add(&bias.to_dtype(xs.dtype())?)?),
            None => Ok(output),
        }
    }
}

/// Applies ConvRot's orthonormal regular-Hadamard transform independently to
/// each row group. The matrix is a Kronecker power of the symmetric H4 basis,
/// so applying it again undoes the checkpoint's weight rotation.
fn hadamard_rows(input: &Tensor, group: usize) -> Result<Tensor> {
    let (rows, width) = input.dims2()?;
    let mut power_check = group;
    while power_check > 1 && power_check % 4 == 0 {
        power_check /= 4;
    }
    if group < 4 || power_check != 1 || width % group != 0 {
        bail!("invalid Hadamard group {group} for row width {width}");
    }
    let groups = width / group;
    let mut value = input.reshape((rows * groups, group))?;
    let h4_stages = group.ilog(4) as usize;
    let mut stride = 1;
    for _ in 0..h4_stages {
        let outer = group / (stride * 4);
        let basis = value.reshape((rows * groups * outer, 4, stride))?;
        let x0 = basis.narrow(1, 0, 1)?.squeeze(1)?;
        let x1 = basis.narrow(1, 1, 1)?.squeeze(1)?;
        let x2 = basis.narrow(1, 2, 1)?.squeeze(1)?;
        let x3 = basis.narrow(1, 3, 1)?.squeeze(1)?;
        let y0 = x0
            .broadcast_add(&x1)?
            .broadcast_add(&x2)?
            .broadcast_sub(&x3)?;
        let y1 = x0
            .broadcast_add(&x1)?
            .broadcast_sub(&x2)?
            .broadcast_add(&x3)?;
        let y2 = x0
            .broadcast_sub(&x1)?
            .broadcast_add(&x2)?
            .broadcast_add(&x3)?;
        let y3 = x0
            .neg()?
            .broadcast_add(&x1)?
            .broadcast_add(&x2)?
            .broadcast_add(&x3)?;
        value = Tensor::stack(&[&y0, &y1, &y2, &y3], 1)?.reshape((rows * groups, group))?;
        stride *= 4;
    }
    Ok(value
        .affine(1.0 / (group as f64).sqrt(), 0.0)?
        .reshape((rows, width))?)
}
