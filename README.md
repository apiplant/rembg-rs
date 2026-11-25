# rembg-rs

**RMBG-3.0 background removal in native Rust.** Run the Bria model locally from a command line, a small web UI, or an HTTP endpoint. The native path uses Candle and an INT8 ConvRot transformer; Python and ComfyUI are optional reference backends.

[Project site](https://apiplant.github.io/rembg-rs/) · [Performance notes](docs/performance.md) · [RMBG-3.0 model card](https://huggingface.co/briaai/RMBG-3.0)

> **Model license:** Bria's RMBG-3.0 weights are CC BY-NC 4.0. Commercial use requires a separate Bria license. This repository's Rust code is MIT licensed.

## Quick start

Requirements: Rust stable, about 14 GB of disk space for the model bundle, and a supported CUDA GPU for CUDA inference. The default cache is `~/.cache/rembg-rs/models/comfyui` (or `$XDG_CACHE_HOME/rembg-rs/models/comfyui`). Model download uses Python's `huggingface_hub`; inference through the native backend does not need Python.

```sh
# Install the small dependency used by the model download command
python3 -m pip install huggingface_hub

# Download the official INT8 model bundle once
cargo run --release -- fetch-comfy

# Remove a background (CUDA)
cargo run --release --features cuda-cudnn -- remove input.jpg output.png

# Start the local browser UI and HTTP API
cargo run --release --features cuda-cudnn -- serve --addr 127.0.0.1:8080
```

Open the URL printed by `serve`. Drop an image to get a transparent PNG, then preview it over another image or a solid color. For a CPU run, use `--device cpu`; for CUDA without cuDNN, build with `--features cuda`. The `cuda-cudnn` feature needs the CUDA toolkit, NVCC, and cuDNN installed.

The model bundle is about 13.6 GB. Override its directory with `--models-dir PATH` or `RMBG_MODELS_DIR=PATH`. `fetch-comfy` resumes interrupted Hugging Face downloads and validates the files. The public INT8 bundle does not require an HF token.

## Server and API

```sh
rembg-rs serve --addr 127.0.0.1:8080
```

The server provides the browser UI at `/`, a readiness check at `/health`, and `POST /remove` for raw image bytes. The response is a full-size PNG with the predicted matte in its alpha channel. Defaults allow a 50 MiB upload and one waiting request. Configure these with `--max-upload-mib`, `--queue-size`, `RMBG_MAX_UPLOAD_MIB`, and `RMBG_QUEUE_SIZE`.

The HTTP worker keeps one model instance loaded and processes one image at a time. Requests beyond the configured queue receive HTTP 503 with `Retry-After`. Bind to `127.0.0.1` for local use; bind to a public interface only behind a trusted access layer.

## CUDA memory and performance

The denoiser keeps the official INT8 ConvRot weights packed on the GPU, row-quantizes activations, runs INT8 tensor-core GEMMs, and dequantizes results to BF16. The FLUX.2 VAE's default `auto` placement uses CUDA only when at least 22 GiB is free at startup; otherwise it runs on CPU. Set `--vae-device cpu`, `same`, `cuda`, or `cuda:N` to choose explicitly. Qwen3-VL encoding currently runs on CPU.

Measured on an RTX 4090 with the same 1410×1080 image and model bundle:

| Implementation | Warm median |
| --- | ---: |
| Native Rust CUDA path | **15.967 s** |
| ComfyUI `AILab_RMBG3` native INT8 path | 17.891 s |

The Rust median was 10.8% lower in this run. These are end-to-end request timings, including PNG transfer. See [the full measurements and method](docs/performance.md).

## Optional reference backends

The Diffusers backend needs access to the gated Hugging Face repositories `briaai/RMBG-3.0` and `briaai/Fibo-2.0-edit-turbo`:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r python/requirements.txt
export HF_TOKEN=YOUR_TOKEN
cargo run --release -- --backend diffusers --python .venv/bin/python remove input.jpg output.png
```

The ComfyUI backend uses an existing ComfyUI server with the [AILab_RMBG3 node](https://github.com/1038lab/ComfyUI-RMBG):

```sh
cargo run --release -- --backend comfy remove input.jpg output.png
cargo run --release -- --backend comfy --comfy-url http://127.0.0.1:8188 serve
```

## Library use

The model components are exposed as `rembg_rs::{Checkpoint, Qwen3Encoder, Fibo2Transformer, Flux2Vae, NativePipeline}`. The pipeline can also be used directly:

```rust,ignore
let pipeline = NativePipeline::load(&rembg_rs::default_models_dir(), "cuda")?;
let matte = pipeline.remove_background(&image)?;
```

## License

The code in this repository is MIT licensed. RMBG-3.0 weights are licensed CC BY-NC 4.0; the non-commercial restriction applies to the weights and their use. Obtain a commercial license from Bria before commercial use. See the [model card](https://huggingface.co/briaai/RMBG-3.0) and [Bria licensing](https://bria.ai/license/).
