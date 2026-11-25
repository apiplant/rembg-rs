//! Native Candle model components for RMBG-3.0 checkpoints.
pub mod fibo2;
pub mod flux2_vae;
mod flux_autoencoder;
#[cfg(feature = "cuda")]
mod int8_cuda;
pub mod native;
pub mod quantized;
pub mod qwen3vl;
