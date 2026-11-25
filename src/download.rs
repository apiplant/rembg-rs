//! Fetch the public quantized model bundle without a Python runtime.

use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::Path,
    time::Duration,
};

use anyhow::{Context, Result, bail};

struct ModelFile {
    repo: &'static str,
    remote: &'static str,
    local: &'static str,
    size: u64,
}

// Sizes are from the public RMBG-3.0-Comfy and Qwen3-VL-4B-Instruct bundles.
// They let a complete local installation start without contacting the Hub.
const FILES: &[ModelFile] = &[
    ModelFile {
        repo: "briaai/RMBG-3.0-Comfy",
        remote: "diffusion_models/rmbg3_int8_convrot.safetensors",
        local: "diffusion_models/rmbg3_int8_convrot.safetensors",
        size: 8_408_562_474,
    },
    ModelFile {
        repo: "briaai/RMBG-3.0-Comfy",
        remote: "text_encoders/qwen3vl_4b_int8_convrot.safetensors",
        local: "text_encoders/qwen3vl_4b_int8_convrot.safetensors",
        size: 4_858_559_240,
    },
    ModelFile {
        repo: "briaai/RMBG-3.0-Comfy",
        remote: "vae/flux2-vae.safetensors",
        local: "vae/flux2-vae.safetensors",
        size: 336_211_292,
    },
    ModelFile {
        repo: "Qwen/Qwen3-VL-4B-Instruct",
        remote: "config.json",
        local: "qwen3vl/config.json",
        size: 1_505,
    },
    ModelFile {
        repo: "Qwen/Qwen3-VL-4B-Instruct",
        remote: "tokenizer.json",
        local: "qwen3vl/tokenizer.json",
        size: 7_032_403,
    },
];

/// Ensure the official INT8 bundle is present, resuming interrupted downloads.
/// A complete bundle requires no network access on subsequent starts.
pub fn ensure_models(models_dir: &Path) -> Result<()> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(120))
        .build();
    for model in FILES {
        let destination = models_dir.join(model.local);
        if destination
            .metadata()
            .is_ok_and(|meta| meta.len() == model.size)
        {
            continue;
        }
        fetch_file(&agent, model, &destination)?;
    }
    Ok(())
}

fn fetch_file(agent: &ureq::Agent, model: &ModelFile, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .context("model file has no parent directory")?;
    fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    let part = destination.with_extension(format!(
        "{}.part",
        destination
            .extension()
            .unwrap_or_default()
            .to_string_lossy()
    ));
    let mut offset = part.metadata().map(|meta| meta.len()).unwrap_or(0);
    if offset > model.size {
        fs::remove_file(&part).with_context(|| format!("removing {}", part.display()))?;
        offset = 0;
    }
    if offset < model.size {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model.repo, model.remote
        );
        eprintln!(
            "Downloading {} ({:.2} GiB){}",
            model.local,
            model.size as f64 / (1024.0 * 1024.0 * 1024.0),
            if offset > 0 { " — resuming" } else { "" }
        );
        let mut request = agent.get(&url);
        if offset > 0 {
            request = request.set("Range", &format!("bytes={offset}-"));
        }
        let response = request
            .call()
            .with_context(|| format!("downloading {url}"))?;
        let append = offset > 0
            && response.status() == 206
            && response
                .header("Content-Range")
                .is_some_and(|range| range.starts_with(&format!("bytes {offset}-")));
        if response.status() != 200 && !append {
            bail!("unexpected HTTP response while downloading {url}");
        }
        let mut output = OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(&part)
            .with_context(|| format!("opening {}", part.display()))?;
        let mut body = response.into_reader();
        let mut buffer = [0u8; 256 * 1024];
        loop {
            let read = body
                .read(&mut buffer)
                .with_context(|| format!("reading {url}"))?;
            if read == 0 {
                break;
            }
            output
                .write_all(&buffer[..read])
                .with_context(|| format!("writing {}", part.display()))?;
        }
        output
            .sync_all()
            .with_context(|| format!("syncing {}", part.display()))?;
    }
    let actual = File::open(&part)?.metadata()?.len();
    if actual != model.size {
        bail!(
            "incomplete download of {}: got {} bytes, expected {} (rerun to resume)",
            model.local,
            actual,
            model.size
        );
    }
    fs::rename(&part, destination)
        .with_context(|| format!("moving {} to {}", part.display(), destination.display()))?;
    eprintln!("Ready {}", model.local);
    Ok(())
}
