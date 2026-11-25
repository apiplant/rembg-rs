//! FLUX.2 VAE used by RMBG-3.0, including the model's 2x2 latent folding and
//! running-statistics normalization.

use std::path::Path;

use super::flux_autoencoder::{Config as FluxConfig, Decoder, Encoder};
use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv2d, Module, VarBuilder, conv2d};

const LATENT_CHANNELS: usize = 32;
const FOLDED_CHANNELS: usize = LATENT_CHANNELS * 4;
const BN_EPS: f64 = 1e-5;

/// FLUX.2 VAE with the exact latent preparation used by Bria's Fibo2 pipeline.
///
/// Inputs and outputs use NCHW RGB tensors in the `[-1, 1]` range. `encode`
/// returns the normalized VAE latents expected by Fibo2, while `decode` accepts
/// those same latents and returns RGB pixels in `[-1, 1]`.
pub struct Flux2Vae {
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: Conv2d,
    post_quant_conv: Conv2d,
    bn_mean: Tensor,
    bn_var: Tensor,
    dtype: DType,
    device: Device,
}

impl Flux2Vae {
    /// Loads the official `flux2-vae.safetensors` checkpoint.
    pub fn load<P: AsRef<Path>>(path: P, device: &Device, dtype: DType) -> Result<Self> {
        // SAFETY: VarBuilder owns the memory map for the model's lifetime.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device) }
            .context("opening FLUX.2 VAE safetensors")?;
        let config = FluxConfig {
            in_channels: 3,
            ch: 128,
            out_ch: 3,
            ch_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            z_channels: LATENT_CHANNELS,
        };
        let conv_cfg = candle_nn::Conv2dConfig::default();
        let encoder = Encoder::new(&config, vb.pp("encoder"))?;
        let decoder = Decoder::new(&config, vb.pp("decoder"))?;
        let quant_conv = conv2d(
            LATENT_CHANNELS * 2,
            LATENT_CHANNELS * 2,
            1,
            conv_cfg,
            vb.pp("encoder.quant_conv"),
        )?;
        let post_quant_conv = conv2d(
            LATENT_CHANNELS,
            LATENT_CHANNELS,
            1,
            conv_cfg,
            vb.pp("decoder.post_quant_conv"),
        )?;
        let bn_mean = vb.get((FOLDED_CHANNELS,), "bn.running_mean")?;
        let bn_var = vb.get((FOLDED_CHANNELS,), "bn.running_var")?;
        Ok(Self {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
            bn_mean,
            bn_var,
            dtype,
            device: device.clone(),
        })
    }

    /// Encodes RGB images into the normalized VAE latents used by Fibo2.
    /// Spatial dimensions must be divisible by 16.
    pub fn encode(&self, pixels: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = pixels.dims4()?;
        if channels != 3 || height % 16 != 0 || width % 16 != 0 {
            bail!("VAE input must be NCHW RGB with height and width divisible by 16");
        }
        let pixels = pixels.to_device(&self.device)?.to_dtype(self.dtype)?;
        let moments = self.quant_conv.forward(&self.encoder.forward(&pixels)?)?;
        // DiagonalGaussianDistribution.mode() is the first half of the moments.
        let mean = moments.chunk(2, 1)?.remove(0);
        let folded = patchify(&mean)?;
        let normalized = self.normalize(&folded)?;
        unpatchify(&normalized).with_context(|| format!("unfolding VAE latents for batch {batch}"))
    }

    /// Decodes normalized Fibo2 latents to RGB images in `[-1, 1]`.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = latents.dims4()?;
        if channels != LATENT_CHANNELS || height % 2 != 0 || width % 2 != 0 {
            bail!("VAE latents must have 32 channels and even height and width");
        }
        let folded = patchify(&latents.to_device(&self.device)?.to_dtype(self.dtype)?)?;
        let folded = self.denormalize(&folded)?;
        let moments = unpatchify(&folded)?;
        let z = self.post_quant_conv.forward(&moments)?;
        self.decoder
            .forward(&z)
            .with_context(|| format!("decoding VAE batch {batch}"))
    }

    fn normalize(&self, latents: &Tensor) -> Result<Tensor> {
        let mean = self
            .bn_mean
            .to_dtype(latents.dtype())?
            .reshape((1, FOLDED_CHANNELS, 1, 1))?;
        let std = self
            .bn_var
            .to_dtype(latents.dtype())?
            .affine(1.0, BN_EPS)?
            .sqrt()?
            .reshape((1, FOLDED_CHANNELS, 1, 1))?;
        Ok(latents.broadcast_sub(&mean)?.broadcast_div(&std)?)
    }

    fn denormalize(&self, latents: &Tensor) -> Result<Tensor> {
        let mean = self
            .bn_mean
            .to_dtype(latents.dtype())?
            .reshape((1, FOLDED_CHANNELS, 1, 1))?;
        let std = self
            .bn_var
            .to_dtype(latents.dtype())?
            .affine(1.0, BN_EPS)?
            .sqrt()?
            .reshape((1, FOLDED_CHANNELS, 1, 1))?;
        Ok(latents.broadcast_mul(&std)?.broadcast_add(&mean)?)
    }
}

fn patchify(latents: &Tensor) -> Result<Tensor> {
    let (batch, channels, height, width) = latents.dims4()?;
    if height % 2 != 0 || width % 2 != 0 {
        bail!("VAE latent dimensions must be even for 2x2 patch folding");
    }
    Ok(latents
        .reshape((batch, channels, height / 2, 2, width / 2, 2))?
        .permute((0, 1, 3, 5, 2, 4))?
        .reshape((batch, channels * 4, height / 2, width / 2))?)
}

fn unpatchify(latents: &Tensor) -> Result<Tensor> {
    let (batch, channels, height, width) = latents.dims4()?;
    if channels % 4 != 0 {
        bail!("folded VAE latent channel count must be divisible by four");
    }
    Ok(latents
        .reshape((batch, channels / 4, 2, 2, height, width))?
        .permute((0, 1, 4, 2, 5, 3))?
        .reshape((batch, channels / 4, height * 2, width * 2))?)
}
