//! RMBG-3.0 background removal with a native Candle pipeline.

pub mod download;
pub mod model;
pub use model::{
    fibo2::Fibo2Transformer, flux2_vae::Flux2Vae, native::NativePipeline, quantized::Checkpoint,
    qwen3vl::Qwen3Encoder,
};

use anyhow::{Context, Result, bail};
use image::{DynamicImage, ImageFormat, RgbaImage, imageops::FilterType};
use std::{path::PathBuf, sync::Mutex};

const MAX_PIXELS: u64 = 40_000_000;

/// Root used for cached model weights.
pub fn cache_root() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"))
}

/// Default ComfyUI-compatible model directory under this app's user cache.
pub fn default_models_dir() -> PathBuf {
    cache_root().join("rembg-rs/models/comfyui")
}

#[derive(Clone, Debug)]
pub struct Config {
    pub device: String,
    /// Device for the FLUX.2 VAE (`auto`, `cpu`, `same`, or `cuda:N`).
    pub vae_device: String,
    pub models_dir: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            device: if cfg!(feature = "cuda") {
                "cuda"
            } else {
                "cpu"
            }
            .into(),
            vae_device: "auto".into(),
            models_dir: default_models_dir(),
        }
    }
}

pub struct Rmbg {
    native: Mutex<NativePipeline>,
}

impl Rmbg {
    /// Loads the model, downloading missing files into `models_dir` first.
    pub fn new(config: Config) -> Result<Self> {
        Ok(Self {
            native: Mutex::new(NativePipeline::load_with_vae_device(
                &config.models_dir,
                &config.device,
                &config.vae_device,
            )?),
        })
    }

    /// Returns an RGBA PNG at the exact source dimensions.
    pub fn remove_background(&self, input: &[u8]) -> Result<Vec<u8>> {
        let source = image::load_from_memory(input).context("unsupported or invalid image")?;
        if source.width() == 0
            || source.height() == 0
            || u64::from(source.width()) * u64::from(source.height()) > MAX_PIXELS
        {
            bail!("image dimensions must be nonzero and at most 40 million pixels");
        }
        let rgb = source.to_rgb8();
        let matte = self
            .native
            .lock()
            .map_err(|_| anyhow::anyhow!("native model lock poisoned"))?
            .remove_background(&source)?;
        let matte =
            image::imageops::resize(&matte, rgb.width(), rgb.height(), FilterType::Lanczos3);
        let (width, height) = rgb.dimensions();
        let mut rgba_bytes = Vec::with_capacity(width as usize * height as usize * 4);
        for (rgb, alpha) in rgb.as_raw().chunks_exact(3).zip(matte.as_raw()) {
            rgba_bytes.extend_from_slice(rgb);
            rgba_bytes.push(*alpha);
        }
        let rgba = RgbaImage::from_raw(width, height, rgba_bytes)
            .context("could not assemble output RGBA image")?;
        let mut output = std::io::Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(rgba).write_to(&mut output, ImageFormat::Png)?;
        Ok(output.into_inner())
    }
}
