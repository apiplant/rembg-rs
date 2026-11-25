# CUDA performance notes

Measured 2026-11-25 on an NVIDIA GeForce RTX 4090 (24 GiB). Rust and ComfyUI
used the same RMBG-3.0 INT8 ConvRot transformer, Qwen3-VL INT8 bundle, FLUX.2
VAE, 1410×1080 input image, four Euler steps, and seed 0. The Rust request used
the native release build with CUDA/cuDNN and `--vae-device auto`. ComfyUI used
the current `AILab_RMBG3` node and PyTorch 2.13.0+cu130. Times include the HTTP
request and PNG response; ComfyUI also includes its upload, prompt, and result
download steps.

| Implementation | First request | Warm requests | Warm median |
| --- | ---: | --- | ---: |
| Rust, packed INT8 CUDA path (`auto` VAE) | 18.263 s | 16.637, 15.740, 15.967 s | **15.967 s** |
| ComfyUI, native INT8 path | 19.757 s | 17.700, 17.891, 18.010 s | **17.891 s** |
| Rust, before the CUDA INT8/VAE changes | 65.834 s | 65.878, 67.277, 67.640 s | **67.277 s** |

The packed Rust path is 10.8% faster than the warmed ComfyUI median and 4.21×
faster than the previous Rust path. The initial ComfyUI request includes its
cold graph and model setup.

## Rust stage profile

For the same image, the first request measured 12.18 s for Qwen encoding,
0.41 s for VAE encode, 4.64 s for denoising, and 0.62 s for VAE decode.
Warm requests measured about 10.1 s for Qwen, 0.035 s for VAE encode, 4.8 s
for denoising, and 0.61 s for VAE decode. Qwen remains on CPU and is now the
largest remaining latency cost. The VAE and denoiser run on CUDA in BF16 and
INT8, respectively.

The INT8 path keeps the ConvRot weights packed on the GPU. A CUDA kernel
applies the 256-value regular Hadamard transform and computes group maxima; a
second kernel row-quantizes the rotated activations. cuBLAS performs the INT8
tensor-core GEMM, followed by CUDA dequantization to BF16. This follows the
[Comfy-Kitchen `int8_tensorwise` ConvRot path](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/cuda/ops/int8_linear.cu)
without expanding the 8.4 GB checkpoint into BF16 weights. After each request
the CUDA pool is trimmed; `nvidia-smi` returned to about 9.4 GiB after the
full-resolution run, down from about 21.8 GiB in the previous Rust build.

Four consecutive Rust results were valid 1410×1080 RGBA PNGs. Their alpha
channels were bit-identical for the fixed Rust seed. Pixel-for-pixel parity
with ComfyUI is not expected because Rust's seeded noise stream differs from
PyTorch's RNG stream. On this image, Rust's optimized alpha had mean absolute
error 1.27/255 against the ComfyUI result, correlation 0.9981, and 0.9963 IoU
after thresholding both mattes at 128. Against the previous dense Rust path,
alpha MAE was 0.04/255.

The VAE defaults to automatic placement: CUDA is selected only when at least
22 GiB is free when the model starts; otherwise it stays on CPU. Use
`--vae-device same` to force CUDA or `--vae-device cpu` to force CPU. The CUDA
memory pool is trimmed after denoising and after VAE decode so temporary
activation allocations do not remain reserved between requests.
