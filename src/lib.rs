//! RMBG-3.0 Rust port: Candle implements the ConvRot linear layers, Fibo2 DiT,
//! and FLUX.2 VAE, with a native image pipeline and optional reference bridges.

pub mod model;
pub use model::{
    fibo2::Fibo2Transformer, flux2_vae::Flux2Vae, native::NativePipeline, quantized::Checkpoint,
    qwen3vl::Qwen3Encoder,
};

use anyhow::{Context, Result, bail};
use base64::{Engine, engine::general_purpose::STANDARD};
use image::{DynamicImage, ImageFormat, RgbaImage, imageops::FilterType};
use serde::{Deserialize, Serialize};
use std::{
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::Mutex,
};

const MAX_PIXELS: u64 = 40_000_000;

/// Root used for model weights and Hugging Face cache data.
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
    pub python: String,
    pub device: String,
    /// Device for the FLUX.2 VAE (`auto`, `cpu`, `same`, or a device like `cuda:0`).
    pub vae_device: String,
    pub cache_dir: PathBuf,
    pub models_dir: PathBuf,
    pub model_id: String,
    pub base_model_id: String,
    pub backend: String,
    pub comfy_url: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            python: "python3".into(),
            device: "cuda".into(),
            // Auto uses CUDA only when the GPU has enough free memory for the
            // quantized denoiser and full-resolution VAE workspace.
            vae_device: "auto".into(),
            cache_dir: cache_root().join("rembg-rs"),
            models_dir: default_models_dir(),
            model_id: "briaai/RMBG-3.0".into(),
            base_model_id: "briaai/Fibo-2.0-edit-turbo".into(),
            backend: "native".into(),
            comfy_url: "http://127.0.0.1:8188".into(),
        }
    }
}

#[derive(Serialize)]
struct Request {
    image: String,
}
#[derive(Deserialize)]
struct Response {
    matte: Option<String>,
    error: Option<String>,
    ready: Option<bool>,
}

struct Worker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}
pub struct Rmbg {
    runtime: Runtime,
}

enum Runtime {
    Python(Mutex<Worker>),
    Native(Mutex<NativePipeline>),
}

impl Rmbg {
    pub fn new(config: Config) -> Result<Self> {
        if config.backend == "native" {
            return Ok(Self {
                runtime: Runtime::Native(Mutex::new(NativePipeline::load_with_vae_device(
                    &config.models_dir,
                    &config.device,
                    &config.vae_device,
                )?)),
            });
        }
        std::fs::create_dir_all(&config.cache_dir)?;
        let script_name = match config.backend.as_str() {
            "diffusers" => "worker.py",
            "comfy" => "comfy_worker.py",
            _ => bail!("backend must be diffusers or comfy"),
        };
        let script = script_path(script_name);
        if !script.is_file() {
            bail!("model worker script missing: {}", script.display());
        }
        let mut child = Command::new(&config.python)
            .arg("-u")
            .arg(script)
            .env("HF_HOME", &config.cache_dir)
            .env("RMBG_DEVICE", &config.device)
            .env("RMBG_MODEL_ID", &config.model_id)
            .env("RMBG_BASE_MODEL_ID", &config.base_model_id)
            .env("RMBG_COMFY_URL", &config.comfy_url)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("starting Python worker; install python/requirements.txt")?;
        let stdin = child.stdin.take().context("worker stdin unavailable")?;
        let stdout = BufReader::new(child.stdout.take().context("worker stdout unavailable")?);
        let mut worker = Worker {
            child,
            stdin,
            stdout,
        };
        let response = read_response(&mut worker)?;
        if response.ready != Some(true) {
            bail!(
                "model worker could not start: {}",
                response.error.unwrap_or_else(|| "unknown error".into())
            );
        }
        Ok(Self {
            runtime: Runtime::Python(Mutex::new(worker)),
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
        let matte = match &self.runtime {
            Runtime::Python(worker) => {
                let mut encoded = std::io::Cursor::new(Vec::new());
                DynamicImage::ImageRgb8(rgb.clone()).write_to(&mut encoded, ImageFormat::Png)?;
                let request = Request {
                    image: STANDARD.encode(encoded.into_inner()),
                };
                let mut worker = worker
                    .lock()
                    .map_err(|_| anyhow::anyhow!("worker lock poisoned"))?;
                serde_json::to_writer(&mut worker.stdin, &request)?;
                worker.stdin.write_all(b"\n")?;
                worker.stdin.flush()?;
                let response = read_response(&mut worker)?;
                if let Some(error) = response.error {
                    bail!("model inference failed: {error}");
                }
                let bytes = STANDARD.decode(response.matte.context("worker omitted matte")?)?;
                image::load_from_memory(&bytes)
                    .context("worker returned invalid matte")?
                    .to_luma8()
            }
            Runtime::Native(native) => native
                .lock()
                .map_err(|_| anyhow::anyhow!("native model lock poisoned"))?
                .remove_background(&source)?,
        };
        // Diffusers' model card specifies Lanczos resizing back to source dimensions.
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

pub fn script_path(name: &str) -> PathBuf {
    let worker_dir = std::env::var_os("RMBG_WORKER_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::current_exe().ok().and_then(|exe| {
                let path = exe.parent()?.join("python");
                path.is_dir().then_some(path)
            })
        })
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python"));
    worker_dir.join(name)
}

impl Drop for Rmbg {
    fn drop(&mut self) {
        if let Runtime::Python(worker) = &mut self.runtime {
            if let Ok(worker) = worker.get_mut() {
                let _ = worker.child.kill();
                let _ = worker.child.wait();
            }
        }
    }
}

fn read_response(worker: &mut Worker) -> Result<Response> {
    let mut line = String::new();
    if worker.stdout.read_line(&mut line)? == 0 {
        bail!("model worker exited unexpectedly");
    }
    serde_json::from_str(&line).context("invalid model worker response")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_location_is_app_specific() {
        assert_eq!(Config::default().cache_dir.file_name().unwrap(), "rembg-rs");
    }
}
