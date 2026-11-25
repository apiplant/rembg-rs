//! End-to-end native RMBG pipeline using Bria's official quantized Comfy bundle.

use std::{path::Path, time::Instant};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, GrayImage, RgbImage, imageops::FilterType};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

use super::qwen3vl::Qwen3Encoder;
use super::{fibo2::Fibo2Transformer, flux2_vae::Flux2Vae, quantized::Checkpoint};

const OUTPUT_AREA: usize = 1024 * 1024;
const VAE_FACTOR: usize = 16;
const QWEN_IMAGE_SIZE: usize = 768;
const DENOISE_STEPS: usize = 4;

pub struct NativePipeline {
    qwen: Qwen3Encoder,
    vae: Flux2Vae,
    transformer: Fibo2Transformer,
    device: Device,
}

impl NativePipeline {
    pub fn load(models_dir: &Path, device_name: &str) -> Result<Self> {
        Self::load_with_vae_device(models_dir, device_name, "auto")
    }

    /// Load the pipeline with an independently selected VAE device. `same`
    /// places it on the denoiser's device; `auto` checks available CUDA memory.
    pub fn load_with_vae_device(
        models_dir: &Path,
        device_name: &str,
        vae_device_name: &str,
    ) -> Result<Self> {
        crate::download::ensure_models(models_dir)?;
        let dit_path = models_dir.join("diffusion_models/rmbg3_int8_convrot.safetensors");
        let qwen_path = models_dir.join("text_encoders/qwen3vl_4b_int8_convrot.safetensors");
        let vae_path = models_dir.join("vae/flux2-vae.safetensors");
        let metadata_dir = models_dir.join("qwen3vl");
        let config_path = metadata_dir.join("config.json");
        let tokenizer_path = metadata_dir.join("tokenizer.json");
        let device = select_device(device_name)?;
        // Qwen and Fibo together exceed 24 GiB. Keep the 4B Qwen encoder in
        // system memory and run the 8B denoiser on the selected accelerator.
        let qwen = Qwen3Encoder::load(
            &qwen_path,
            &config_path,
            &tokenizer_path,
            &Device::Cpu,
            // Candle's CPU matmul backend does not support BF16. Qwen stays
            // on CPU, so use F32 there and cast its feature bundles on the GPU.
            DType::F32,
        )
        .context("loading native Qwen3-VL encoder")?;
        // Keep VAE weights and activations off the denoiser's accelerator by
        // default. The large DiT needs the available VRAM for inference. Users
        // with spare memory can opt into `--vae-device same` or `cuda:N`.
        let vae_device = match vae_device_name {
            "same" => device.clone(),
            "auto" => auto_vae_device(&device)?,
            name => select_device(name)?,
        };
        let vae_dtype = if vae_device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let vae = Flux2Vae::load(&vae_path, &vae_device, vae_dtype)
            .context("loading native FLUX.2 VAE")?;
        let checkpoint = Checkpoint::open(&[dit_path])?;
        let transformer = Fibo2Transformer::load(&checkpoint, &device)
            .context("loading native Fibo2 transformer")?;
        Ok(Self {
            qwen,
            vae,
            transformer,
            device,
        })
    }

    /// Returns the generated grayscale matte at the model's trained crop size.
    pub fn remove_background(&self, source: &DynamicImage) -> Result<GrayImage> {
        let profiling = std::env::var_os("RMBG_PROFILE").is_some();
        let total = Instant::now();
        let stage = Instant::now();
        let (vae_pixels, qwen_image) = preprocess(source, &Device::Cpu)?;
        if profiling {
            eprintln!(
                "rmbg profile preprocess_ms={:.1}",
                stage.elapsed().as_secs_f64() * 1000.0
            );
        }
        let stage = Instant::now();
        let features = self.qwen.encode(&qwen_image)?.to_device(&self.device)?;
        if profiling {
            eprintln!(
                "rmbg profile qwen_encode_ms={:.1}",
                stage.elapsed().as_secs_f64() * 1000.0
            );
        }
        let stage = Instant::now();
        let context = self.vae.encode(&vae_pixels)?.to_device(&self.device)?;
        if profiling {
            eprintln!(
                "rmbg profile vae_encode_ms={:.1}",
                stage.elapsed().as_secs_f64() * 1000.0
            );
        }
        let (_, _, latent_h, latent_w) = context.dims4()?;
        let mut rng = StdRng::seed_from_u64(0);
        let noise = (0..32 * latent_h * latent_w)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect::<Vec<f32>>();
        let latents = Tensor::from_vec(noise, (1, 32, latent_h, latent_w), &self.device)?;
        let sigmas = shifted_sigmas(latent_h * latent_w / 4, DENOISE_STEPS);
        let stage = Instant::now();
        let result = self
            .transformer
            .denoise(&latents, &[context], &features, &sigmas)?;
        if profiling {
            eprintln!(
                "rmbg profile denoise_ms={:.1}",
                stage.elapsed().as_secs_f64() * 1000.0
            );
        }
        // The VAE decoder needs a large workspace at full resolution. Release
        // denoiser activations and return unused async-pool blocks before it
        // allocates, while keeping the transformer weights resident.
        drop(features);
        drop(latents);
        if self.device.is_cuda() {
            trim_cuda_pool(&self.device)?;
        }
        let stage = Instant::now();
        let decoded_result = self.vae.decode(&result);
        if profiling {
            if let Err(error) = &decoded_result {
                eprintln!("rmbg profile vae_decode_error={error:#}");
            }
        }
        let decoded = decoded_result?.to_device(&Device::Cpu)?;
        drop(result);
        if self.device.is_cuda() {
            trim_cuda_pool(&self.device)?;
        }
        let decoded = decoded.squeeze(0)?.to_dtype(DType::F32)?;
        if profiling {
            eprintln!(
                "rmbg profile vae_decode_ms={:.1}",
                stage.elapsed().as_secs_f64() * 1000.0
            );
        }
        let planes = decoded.to_vec3::<f32>()?;
        let height = planes[0].len();
        let width = planes[0][0].len();
        let mut pixels = RgbImage::new(width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                let pixel = std::array::from_fn(|c| {
                    (((planes[c][y][x].clamp(-1.0, 1.0) + 1.0) * 127.5).round()) as u8
                });
                pixels.put_pixel(x as u32, y as u32, image::Rgb(pixel));
            }
        }
        let matte = DynamicImage::ImageRgb8(pixels).to_luma8();
        if profiling {
            eprintln!(
                "rmbg profile total_ms={:.1}",
                total.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(matte)
    }
}

#[cfg(feature = "cuda")]
fn auto_vae_device(device: &Device) -> Result<Device> {
    if device.is_metal() {
        return Ok(device.clone());
    }
    let Device::Cuda(cuda) = device else {
        return Ok(Device::Cpu);
    };
    let (free_bytes, _) = cuda.cuda_stream().context().mem_get_info()?;
    if free_bytes >= 22 * 1024 * 1024 * 1024 {
        Ok(device.clone())
    } else {
        Ok(Device::Cpu)
    }
}

#[cfg(not(feature = "cuda"))]
fn auto_vae_device(device: &Device) -> Result<Device> {
    if device.is_metal() {
        Ok(device.clone())
    } else {
        Ok(Device::Cpu)
    }
}

#[cfg(feature = "cuda")]
fn trim_cuda_pool(device: &Device) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::result;

    let Device::Cuda(cuda) = device else {
        return Ok(());
    };
    let stream = cuda.cuda_stream();
    stream
        .synchronize()
        .context("synchronizing CUDA before VAE decode")?;
    let context = stream.context();
    // SAFETY: The device is owned by Candle's live CUDA context and the default
    // memory pool belongs to that same device.
    let pool = unsafe { result::device::get_default_mem_pool(context.cu_device()) }
        .context("getting CUDA default memory pool")?;
    // SAFETY: All temporary inference tensors dropped above, and the stream was
    // synchronized before returning unused blocks to the driver.
    unsafe { result::mem_pool::trim_to(pool, 0) }.context("trimming unused CUDA memory")?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn trim_cuda_pool(_device: &Device) -> Result<()> {
    Ok(())
}

fn select_device(device: &str) -> Result<Device> {
    if device == "cpu" {
        return Ok(Device::Cpu);
    }
    if device == "cuda" {
        return Device::new_cuda(0).context("CUDA requested; build with `--features cuda`");
    }
    if let Some(index) = device.strip_prefix("cuda:") {
        return Device::new_cuda(index.parse()?)
            .context("CUDA requested; build with `--features cuda`");
    }
    if device == "metal" || device == "mps" {
        return Device::new_metal(0)
            .context("Metal device unavailable; build with `--features metal`");
    }
    bail!("native device must be cpu, cuda, cuda:N, or metal")
}

fn preprocess(source: &DynamicImage, device: &Device) -> Result<(Tensor, DynamicImage)> {
    let rgb = source.to_rgb8();
    let aspect = rgb.width() as f64 / rgb.height() as f64;
    let (width, height) = trained_size(aspect)?;
    let crop = crop_to_aspect(&rgb, width as f64 / height as f64);
    let crop = image::imageops::resize(&crop, width as u32, height as u32, FilterType::Lanczos3);
    let mut values = Vec::with_capacity(3 * width * height);
    for channel in 0..3 {
        for pixel in crop.pixels() {
            values.push(pixel[channel] as f32 / 127.5 - 1.0);
        }
    }
    let vae_pixels = Tensor::from_vec(values, (1, 3, height, width), device)?;
    let qwen_image = thumbnail(&crop, QWEN_IMAGE_SIZE);
    Ok((vae_pixels, DynamicImage::ImageRgb8(qwen_image)))
}

fn trained_size(aspect: f64) -> Result<(usize, usize)> {
    let multiple = VAE_FACTOR;
    let min_side = 4 * multiple;
    let mut best: Option<(i64, usize, usize, usize)> = None;
    for width in (min_side..2 * OUTPUT_AREA / min_side).step_by(multiple) {
        for height in (min_side..2 * OUTPUT_AREA / width).step_by(multiple) {
            let area = width * height;
            if !(0.9 * OUTPUT_AREA as f64..=1.1 * OUTPUT_AREA as f64).contains(&(area as f64)) {
                continue;
            }
            let ratio = width as f64 / height as f64;
            let crop = 1.0 - (ratio / aspect).min(aspect / ratio);
            let candidate = (
                (crop * 1_000_000_000.0).round() as i64,
                area.abs_diff(OUTPUT_AREA),
                width,
                height,
            );
            let better = best.as_ref().is_none_or(|current| {
                candidate.0 < current.0
                    || (candidate.0 == current.0
                        && (candidate.1 < current.1
                            || (candidate.1 == current.1 && candidate.2 > current.2)))
            });
            if better {
                best = Some(candidate);
            }
        }
    }
    let (_, _, width, height) = best.context("could not find RMBG trained image dimensions")?;
    Ok((width, height))
}

fn crop_to_aspect(image: &RgbImage, aspect: f64) -> RgbImage {
    let source_aspect = image.width() as f64 / image.height() as f64;
    if source_aspect > aspect {
        let width = (image.height() as f64 * aspect).round() as u32;
        let left = (image.width() - width) / 2;
        image::imageops::crop_imm(image, left, 0, width, image.height()).to_image()
    } else {
        let height = (image.width() as f64 / aspect).round() as u32;
        let top = (image.height() - height) / 2;
        image::imageops::crop_imm(image, 0, top, image.width(), height).to_image()
    }
}

fn thumbnail(image: &RgbImage, max_side: usize) -> RgbImage {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let largest = width.max(height);
    if largest <= max_side {
        return image.clone();
    }
    let scale = max_side as f64 / largest as f64;
    let width = ((width as f64 * scale).round() as usize).max(1);
    let height = ((height as f64 * scale).round() as usize).max(1);
    image::imageops::resize(image, width as u32, height as u32, FilterType::Lanczos3)
}

fn shifted_sigmas(image_seq_len: usize, steps: usize) -> Vec<f32> {
    let base_seq = 256.0f64;
    let max_seq = 4096.0f64;
    let base_shift = 0.5f64;
    let max_shift = 1.15f64;
    let slope = (max_shift - base_shift) / (max_seq - base_seq);
    let mu = image_seq_len as f64 * slope + base_shift - slope * base_seq;
    let exp_mu = mu.exp();
    let mut sigmas = (0..steps)
        .map(|index| {
            let raw = 1.0 - index as f64 * (1.0 - 1.0 / steps as f64) / (steps - 1) as f64;
            (exp_mu / (exp_mu + (1.0 / raw - 1.0))) as f32
        })
        .collect::<Vec<_>>();
    sigmas.push(0.0);
    sigmas
}
