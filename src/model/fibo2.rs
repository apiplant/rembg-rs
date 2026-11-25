//! Candle implementation of the Fibo 2 DiT used by RMBG-3.0.
//!
//! The module consumes VAE latents (`[1, 32, H/8, W/8]`) and the six Qwen
//! feature bundles produced by RMBG's fixed instruction. `Flux2Vae` implements
//! the model's FLUX.2 VAE stage; Qwen encoding and image-pipeline integration
//! remain separate stages.

use anyhow::{Result, bail};
use candle_core::{D, DType, Device, Tensor};

use super::quantized::{Checkpoint, Linear};

const DIM: usize = 3584;
const CHANNELS: usize = 32;
const HEADS: usize = 28;
const HEAD_DIM: usize = DIM / HEADS;
const CAP_DIM: usize = 7680;
const LAYERS: usize = 44;
const INJECTION: [usize; 5] = [6, 14, 22, 30, 38];
const NORM_EPS: f64 = 1e-5;
const PATCH: usize = 2;

fn silu(x: &Tensor) -> Result<Tensor> {
    // Candle's unary SiLU kernel fuses sigmoid and multiply on CUDA.
    Ok(x.silu()?)
}

struct Norm(Tensor);

impl Norm {
    fn load(cp: &Checkpoint, name: &str, dev: &Device) -> Result<Self> {
        Ok(Self(cp.tensor(name, dev)?))
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::rms_norm(
            x,
            &self.0.to_dtype(x.dtype())?,
            NORM_EPS as f32,
        )?)
    }
}

fn linear(cp: &Checkpoint, name: &str, dev: &Device) -> Result<Linear> {
    let linear = cp.quantized_linear(name, dev)?;
    if dev.is_cuda() {
        linear.keep_quantized(DType::BF16)
    } else {
        linear.dequantize(DType::BF16)
    }
}

fn dense_activation(linear: &Linear, x: &Tensor) -> Result<Tensor> {
    let dtype = match linear {
        Linear::Dense { weight, .. } => weight.dtype(),
        Linear::Int8ConvRot { .. } => x.dtype(),
    };
    linear.forward(&x.to_dtype(dtype)?)
}

struct FeedForward {
    w13: Linear,
    w2: Linear,
}

impl FeedForward {
    fn load(cp: &Checkpoint, prefix: &str, dev: &Device) -> Result<Self> {
        Ok(Self {
            w13: linear(cp, &format!("{prefix}.w13"), dev)?,
            w2: linear(cp, &format!("{prefix}.w2"), dev)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fused = self.w13.forward(x)?;
        let halves = fused.chunk(2, D::Minus1)?;
        let gated = silu(&halves[0])?.broadcast_mul(&halves[1])?;
        self.w2.forward(&gated)
    }
}

struct Attention {
    q: Option<Linear>,
    kv: Option<Linear>,
    qkv: Option<Linear>,
    out: Linear,
    q_norm: Norm,
    k_norm: Norm,
}

impl Attention {
    fn self_attention(cp: &Checkpoint, prefix: &str, dev: &Device) -> Result<Self> {
        Ok(Self {
            q: None,
            kv: None,
            qkv: Some(linear(cp, &format!("{prefix}.qkv"), dev)?),
            out: linear(cp, &format!("{prefix}.out"), dev)?,
            q_norm: Norm::load(cp, &format!("{prefix}.q_norm.weight"), dev)?,
            k_norm: Norm::load(cp, &format!("{prefix}.k_norm.weight"), dev)?,
        })
    }

    fn cross_attention(cp: &Checkpoint, prefix: &str, dev: &Device) -> Result<Self> {
        Ok(Self {
            q: Some(linear(cp, &format!("{prefix}.to_q"), dev)?),
            kv: Some(linear(cp, &format!("{prefix}.to_kv"), dev)?),
            qkv: None,
            out: linear(cp, &format!("{prefix}.to_out"), dev)?,
            q_norm: Norm::load(cp, &format!("{prefix}.norm_q.weight"), dev)?,
            k_norm: Norm::load(cp, &format!("{prefix}.norm_k.weight"), dev)?,
        })
    }

    fn forward(&self, x: &Tensor, context: Option<&Tensor>, rope: Option<&Rope>) -> Result<Tensor> {
        let (batch, seq, _) = x.dims3()?;
        let (q, k, v) = if let Some(qkv) = &self.qkv {
            let fused = qkv.forward(x)?;
            let parts = fused.chunk(3, D::Minus1)?;
            (parts[0].clone(), parts[1].clone(), parts[2].clone())
        } else {
            let source = context.unwrap_or(x);
            let q = self.q.as_ref().unwrap().forward(x)?;
            let kv = self.kv.as_ref().unwrap().forward(source)?;
            let parts = kv.chunk(2, D::Minus1)?;
            (q, parts[0].clone(), parts[1].clone())
        };
        let q = q.reshape((batch, seq, HEADS, HEAD_DIM))?;
        let k_seq = k.dim(1)?;
        let q = self.q_norm.forward(&q)?;
        let k = self
            .k_norm
            .forward(&k.reshape((batch, k_seq, HEADS, HEAD_DIM))?)?;
        let v = v.reshape((batch, k_seq, HEADS, HEAD_DIM))?;
        let (q, k) = if let Some(rope) = rope {
            (rope.apply(&q)?, rope.apply(&k)?)
        } else {
            (q, k)
        };
        let attended = attention(&q, &k, &v)?;
        self.out.forward(&attended.reshape((batch, seq, DIM))?)
    }
}

struct TransformerBlock {
    attention: Attention,
    feed_forward: FeedForward,
    attention_norm1: Norm,
    attention_norm2: Norm,
    ffn_norm1: Norm,
    ffn_norm2: Norm,
    adaln: Linear,
    cross_attention: Option<Attention>,
    cross_norm1: Option<Norm>,
    cross_norm2: Option<Norm>,
    cross_gate: Option<Tensor>,
}

impl TransformerBlock {
    fn load(cp: &Checkpoint, prefix: &str, inject: bool, dev: &Device) -> Result<Self> {
        let cross_attention = inject
            .then(|| Attention::cross_attention(cp, &format!("{prefix}.cross_attention"), dev))
            .transpose()?;
        Ok(Self {
            attention: Attention::self_attention(cp, &format!("{prefix}.attention"), dev)?,
            feed_forward: FeedForward::load(cp, &format!("{prefix}.feed_forward"), dev)?,
            attention_norm1: Norm::load(cp, &format!("{prefix}.attention_norm1.weight"), dev)?,
            attention_norm2: Norm::load(cp, &format!("{prefix}.attention_norm2.weight"), dev)?,
            ffn_norm1: Norm::load(cp, &format!("{prefix}.ffn_norm1.weight"), dev)?,
            ffn_norm2: Norm::load(cp, &format!("{prefix}.ffn_norm2.weight"), dev)?,
            adaln: linear(cp, &format!("{prefix}.adaLN_modulation.0"), dev)?,
            cross_attention,
            cross_norm1: inject
                .then(|| Norm::load(cp, &format!("{prefix}.cross_attn_norm1.weight"), dev))
                .transpose()?,
            cross_norm2: inject
                .then(|| Norm::load(cp, &format!("{prefix}.cross_attn_norm2.weight"), dev))
                .transpose()?,
            cross_gate: if inject {
                Some(cp.tensor(&format!("{prefix}.cross_attn_gate"), dev)?)
            } else {
                None
            },
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        noisy: &Tensor,
        clean: &Tensor,
        noise_mask: &Tensor,
        rope: Option<&Rope>,
        text: Option<&Tensor>,
    ) -> Result<Tensor> {
        let noise_mask = noise_mask.to_dtype(x.dtype())?;
        let mod_noisy = self.adaln.forward(noisy)?.chunk(4, D::Minus1)?;
        let mod_clean = self.adaln.forward(clean)?.chunk(4, D::Minus1)?;
        let mut mods = Vec::with_capacity(4);
        for i in 0..4 {
            let a = mod_noisy[i].unsqueeze(1)?;
            let b = mod_clean[i].unsqueeze(1)?;
            let mixed = b.broadcast_add(&a.broadcast_sub(&b)?.broadcast_mul(&noise_mask)?)?;
            mods.push(mixed);
        }
        let scale_attn = mods[0].affine(1.0, 1.0)?;
        let gate_attn = mods[1].tanh()?;
        let scale_ffn = mods[2].affine(1.0, 1.0)?;
        let gate_ffn = mods[3].tanh()?;

        let attn_input = self
            .attention_norm1
            .forward(x)?
            .broadcast_mul(&scale_attn)?;
        let attn = self.attention.forward(&attn_input, None, rope)?;
        let mut hidden =
            x.broadcast_add(&gate_attn.broadcast_mul(&self.attention_norm2.forward(&attn)?)?)?;
        if let (Some(attn), Some(text)) = (&self.cross_attention, text) {
            let cross_input = self.cross_norm1.as_ref().unwrap().forward(&hidden)?;
            let cross = attn.forward(&cross_input, Some(text), None)?;
            let gate = self
                .cross_gate
                .as_ref()
                .unwrap()
                .tanh()?
                .to_dtype(hidden.dtype())?;
            let cross = self.cross_norm2.as_ref().unwrap().forward(&cross)?;
            hidden = hidden.broadcast_add(&cross.broadcast_mul(&gate)?)?;
        }
        let ffn_input = self.ffn_norm1.forward(&hidden)?.broadcast_mul(&scale_ffn)?;
        let ffn = self.feed_forward.forward(&ffn_input)?;
        Ok(hidden.broadcast_add(&gate_ffn.broadcast_mul(&self.ffn_norm2.forward(&ffn)?)?)?)
    }
}

struct PerceiverLayer {
    cross: Attention,
    self_attn: Attention,
    cross_norm: Norm,
    self_norm: Norm,
    ffn_norm: Norm,
    ffn: FeedForward,
}

impl PerceiverLayer {
    fn load(cp: &Checkpoint, prefix: &str, dev: &Device) -> Result<Self> {
        Ok(Self {
            cross: Attention::cross_attention(cp, &format!("{prefix}.cross_attention"), dev)?,
            self_attn: Attention::cross_attention(cp, &format!("{prefix}.self_attention"), dev)?,
            cross_norm: Norm::load(cp, &format!("{prefix}.cross_attn_norm.weight"), dev)?,
            self_norm: Norm::load(cp, &format!("{prefix}.self_attn_norm.weight"), dev)?,
            ffn_norm: Norm::load(cp, &format!("{prefix}.ffn_norm.weight"), dev)?,
            ffn: FeedForward::load(cp, &format!("{prefix}.ffn"), dev)?,
        })
    }
    fn forward(&self, x: &Tensor, text: &Tensor) -> Result<Tensor> {
        let cross = self
            .cross
            .forward(&self.cross_norm.forward(x)?, Some(text), None)?;
        let x = x.broadcast_add(&cross)?;
        let self_attn = self
            .self_attn
            .forward(&self.self_norm.forward(&x)?, None, None)?;
        let x = x.broadcast_add(&self_attn)?;
        let ffn = self.ffn.forward(&self.ffn_norm.forward(&x)?)?;
        Ok(x.broadcast_add(&ffn)?)
    }
}

struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    fn new(ids: &[[usize; 3]], dev: &Device) -> Result<Self> {
        let dims = [32usize, 48, 48];
        let mut cos = Vec::with_capacity(ids.len() * (HEAD_DIM / 2));
        let mut sin = Vec::with_capacity(ids.len() * (HEAD_DIM / 2));
        for id in ids {
            for (axis, axis_dim) in dims.iter().enumerate() {
                for j in (0..*axis_dim).step_by(2) {
                    let freq = 256f64.powf(-(j as f64) / (*axis_dim as f64));
                    let phase = id[axis] as f64 * freq;
                    cos.push(phase.cos() as f32);
                    sin.push(phase.sin() as f32);
                }
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (1, ids.len(), 1, HEAD_DIM / 2), dev)?,
            sin: Tensor::from_vec(sin, (1, ids.len(), 1, HEAD_DIM / 2), dev)?,
        })
    }

    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let b = dims[0];
        let s = dims[1];
        let h = dims[2];
        let paired = x.reshape((b, s, h, HEAD_DIM / 2, 2))?;
        let halves = paired.chunk(2, D::Minus1)?;
        let even = halves[0].squeeze(D::Minus1)?;
        let odd = halves[1].squeeze(D::Minus1)?;
        let cos = self.cos.to_dtype(x.dtype())?;
        let sin = self.sin.to_dtype(x.dtype())?;
        let a = even
            .broadcast_mul(&cos)?
            .broadcast_sub(&odd.broadcast_mul(&sin)?)?;
        let b = even
            .broadcast_mul(&sin)?
            .broadcast_add(&odd.broadcast_mul(&cos)?)?;
        Ok(Tensor::cat(
            &[&a.unsqueeze(D::Minus1)?, &b.unsqueeze(D::Minus1)?],
            D::Minus1,
        )?
        .reshape((dims[0], dims[1], dims[2], HEAD_DIM))?)
    }
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (batch, seq, heads, width) = q.dims4()?;
    if k.dim(0)? != batch || k.dim(2)? != heads || v.dim(0)? != batch || v.dim(2)? != heads {
        bail!("Fibo2 attention expects matching batch and head dimensions");
    }
    if q.device().is_cuda() {
        #[cfg(feature = "cuda")]
        {
            let result =
                candle_flash_attn::flash_attn(q, k, v, 1.0 / (width as f32).sqrt(), false)?;
            return Ok(result);
        }
        #[cfg(not(feature = "cuda"))]
        bail!("CUDA Fibo2 requires building with the `cuda` feature for flash attention");
    }
    if q.device().is_metal() {
        return Ok(candle_nn::ops::sdpa(
            &q.transpose(1, 2)?,
            &k.transpose(1, 2)?,
            &v.transpose(1, 2)?,
            None,
            false,
            1.0 / (width as f32).sqrt(),
            0.0,
        )?
        .transpose(1, 2)?);
    }
    if batch == 1 {
        let lens = Tensor::from_vec(vec![seq as u32], (1,), q.device())?;
        let q = q.reshape((seq, heads, width))?.to_dtype(DType::F16)?;
        let k_seq = k.dim(2)?;
        let k = k.reshape((k_seq, heads, width))?.to_dtype(DType::F16)?;
        let v = v.reshape((k_seq, heads, width))?.to_dtype(DType::F16)?;
        let output = candle_nn::attention::flash_attn_varlen_cpu(
            &q,
            &k,
            &v,
            None,
            &lens,
            &Tensor::from_vec(vec![k_seq as u32], (1,), q.device())?,
            seq,
            k_seq,
            1.0 / (width as f32).sqrt(),
            false,
            None,
            None,
        )?;
        return Ok(output
            .reshape((1, seq, heads, width))?
            .to_dtype(DType::BF16)?);
    }
    bail!("CPU Fibo2 attention currently supports batch size one")
}

fn patchify(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    if h % PATCH != 0 || w % PATCH != 0 {
        bail!("latent dimensions must be divisible by the transformer patch size");
    }
    Ok(x.reshape((b, c, h / PATCH, PATCH, w / PATCH, PATCH))?
        .permute((0, 2, 4, 1, 3, 5))?
        .reshape((b, (h / PATCH) * (w / PATCH), c * PATCH * PATCH))?)
}

fn grid_ids(plane: usize, h: usize, w: usize) -> Vec<[usize; 3]> {
    let mut ids = Vec::with_capacity(h * w);
    for y in 0..h {
        for x in 0..w {
            ids.push([plane, y, x]);
        }
    }
    ids
}

fn timestep_embedding(t: &Tensor, dev: &Device) -> Result<Tensor> {
    let values = t.to_vec1::<f32>()?;
    let mut embedded = Vec::with_capacity(values.len() * 256);
    for value in values {
        for i in 0..128 {
            let f = (-(10000f64.ln()) * i as f64 / 128.0).exp();
            embedded.push((value as f64 * f).cos() as f32);
        }
        for i in 0..128 {
            let f = (-(10000f64.ln()) * i as f64 / 128.0).exp();
            embedded.push((value as f64 * f).sin() as f32);
        }
    }
    Ok(Tensor::from_vec(embedded, (t.dims1()?, 256), dev)?)
}

fn final_layer_norm(x: &Tensor) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(centered.broadcast_div(&(variance.affine(1.0, 1e-6)?.sqrt()?))?)
}

/// Fibo2 diffusion transformer. Encoder output shape is `[1, 6, text_len, 7680]`.
pub struct Fibo2Transformer {
    x_embedder: Linear,
    t0: Linear,
    t2: Linear,
    noise_refiner: Vec<TransformerBlock>,
    layers: Vec<TransformerBlock>,
    gist_queries: Tensor,
    perceiver: Vec<PerceiverLayer>,
    perceiver_norm: Norm,
    final_mod: Linear,
    final_linear: Linear,
}

impl Fibo2Transformer {
    pub fn load(cp: &Checkpoint, dev: &Device) -> Result<Self> {
        let noise_refiner = (0..2)
            .map(|i| TransformerBlock::load(cp, &format!("noise_refiner.{i}"), false, dev))
            .collect::<Result<Vec<_>>>()?;
        let layers = (0..LAYERS)
            .map(|i| {
                TransformerBlock::load(cp, &format!("layers.{i}"), INJECTION.contains(&i), dev)
            })
            .collect::<Result<Vec<_>>>()?;
        let perceiver = (0..2)
            .map(|i| PerceiverLayer::load(cp, &format!("perceiver.layers.{i}"), dev))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            x_embedder: linear(cp, "x_embedder", dev)?,
            t0: linear(cp, "t_embedder.mlp.0", dev)?,
            t2: linear(cp, "t_embedder.mlp.2", dev)?,
            noise_refiner,
            layers,
            gist_queries: cp.tensor("perceiver.latent_queries", dev)?,
            perceiver,
            perceiver_norm: Norm::load(cp, "perceiver.norm_out.weight", dev)?,
            final_mod: linear(cp, "final_layer.adaLN_modulation.1", dev)?,
            final_linear: linear(cp, "final_layer.linear", dev)?,
        })
    }

    /// One Fibo2 velocity prediction. `encoder_features` contains the primary
    /// Qwen bundle followed by the five injection bundles in model-card order.
    pub fn forward(
        &self,
        noisy_latents: &Tensor,
        context_latents: &[Tensor],
        timestep: &Tensor,
        encoder_features: &Tensor,
    ) -> Result<Tensor> {
        let (batch, channels, height, width) = noisy_latents.dims4()?;
        if batch != 1 || channels != CHANNELS {
            bail!("Fibo2 requires one 32-channel latent tensor per call");
        }
        let (enc_batch, bundles, text_len, cap_dim) = encoder_features.dims4()?;
        if enc_batch != 1 || bundles < 6 || cap_dim != CAP_DIM {
            bail!(
                "Qwen feature tensor must be [1, >=6, text_len, 7680], got {:?}",
                encoder_features.dims()
            );
        }
        let h_tokens = height / PATCH;
        let w_tokens = width / PATCH;
        let image_count = h_tokens * w_tokens;
        let mut token_pieces = vec![patchify(noisy_latents)?];
        let plane = 513;
        let mut ids = grid_ids(plane, h_tokens, w_tokens);
        for (i, context) in context_latents.iter().enumerate() {
            token_pieces.push(patchify(context)?);
            let (_, _, ch, cw) = context.dims4()?;
            ids.extend(grid_ids(plane + i + 1, ch / PATCH, cw / PATCH));
        }
        let patch_tokens = Tensor::cat(&token_pieces.iter().collect::<Vec<_>>(), 1)?;
        let patch_tokens = self.x_embedder.forward(&patch_tokens)?;
        let patch_freqs = Rope::new(&ids, noisy_latents.device())?;

        let t = timestep.affine(1000.0, 0.0)?;
        let t_embed =
            timestep_embedding(&t, noisy_latents.device())?.to_dtype(noisy_latents.dtype())?;
        let noisy_embed = silu(&dense_activation(&self.t0, &t_embed)?)?;
        let noisy_adaln = dense_activation(&self.t2, &noisy_embed)?;
        let zero_t = Tensor::zeros_like(&t)?;
        let clean_embed =
            timestep_embedding(&zero_t, noisy_latents.device())?.to_dtype(noisy_latents.dtype())?;
        let clean_embed = silu(&dense_activation(&self.t0, &clean_embed)?)?;
        let clean_adaln = dense_activation(&self.t2, &clean_embed)?;

        let context_count = patch_tokens.dim(1)? - image_count;
        let ref_mask = noise_mask(noisy_latents.device(), image_count, context_count, 0)?;
        let mut refined = patch_tokens;
        for block in &self.noise_refiner {
            refined = block.forward(
                &refined,
                &noisy_adaln,
                &clean_adaln,
                &ref_mask,
                Some(&patch_freqs),
                None,
            )?;
        }

        let gist_count = gist_count(text_len);
        let text = encoder_features.narrow(1, 0, 1)?.squeeze(1)?;
        let queries = self
            .gist_queries
            .narrow(0, 0, gist_count)?
            .unsqueeze(0)?
            .broadcast_as((1, gist_count, DIM))?
            .to_dtype(text.dtype())?;
        let mut gist = queries;
        for layer in &self.perceiver {
            gist = layer.forward(&gist, &text)?;
        }
        gist = self.perceiver_norm.forward(&gist)?;
        let mut all_ids = (0..gist_count).map(|i| [i + 1, 0, 0]).collect::<Vec<_>>();
        all_ids.extend(ids);
        let full_rope = Rope::new(&all_ids, noisy_latents.device())?;
        let hidden = Tensor::cat(&[&gist, &refined], 1)?;
        let full_mask = noise_mask(
            noisy_latents.device(),
            image_count,
            context_count,
            gist_count,
        )?;
        let injection_bundle = [6usize, 14, 22, 30, 38];
        let mut hidden = hidden;
        for (index, block) in self.layers.iter().enumerate() {
            let text =
                if let Some(bundle) = injection_bundle.iter().position(|&layer| layer == index) {
                    Some(encoder_features.narrow(1, bundle + 1, 1)?.squeeze(1)?)
                } else {
                    None
                };
            hidden = block.forward(
                &hidden,
                &noisy_adaln,
                &clean_adaln,
                &full_mask,
                Some(&full_rope),
                text.as_ref(),
            )?;
        }

        let image_tokens = hidden.narrow(1, gist_count, image_count)?;
        let modulation =
            dense_activation(&self.final_mod, &silu(&noisy_adaln)?)?.affine(1.0, 1.0)?;
        let normalized =
            final_layer_norm(&image_tokens)?.broadcast_mul(&modulation.unsqueeze(1)?)?;
        let projected = self.final_linear.forward(&normalized)?;
        Ok(projected
            .reshape((batch, h_tokens, w_tokens, CHANNELS, PATCH, PATCH))?
            .permute((0, 3, 1, 4, 2, 5))?
            .reshape((batch, CHANNELS, height, width))?)
    }

    /// Flow-matching Euler loop. `latents` is the initial FP32 noise and
    /// `sigmas` must contain each of the four-or-more noise levels plus a final
    /// zero, as returned by Diffusers' shifted FlowMatch scheduler.
    pub fn denoise(
        &self,
        latents: &Tensor,
        context_latents: &[Tensor],
        encoder_features: &Tensor,
        sigmas: &[f32],
    ) -> Result<Tensor> {
        if sigmas.len() < 2 || sigmas.last().copied() != Some(0.0) {
            bail!("sigmas must contain at least one step and end with 0.0");
        }
        if sigmas.windows(2).any(|w| w[0] < w[1]) {
            bail!("sigmas must be in descending order");
        }
        let mut sample = latents.to_dtype(DType::F32)?;
        let contexts = context_latents
            .iter()
            .map(|t| Ok(t.to_dtype(DType::BF16)?))
            .collect::<Result<Vec<_>>>()?;
        let features = encoder_features.to_dtype(DType::BF16)?;
        for pair in sigmas.windows(2) {
            let timestep = Tensor::from_vec(vec![pair[0]], (1,), sample.device())?;
            let model_input = sample.to_dtype(DType::BF16)?;
            let velocity = self
                .forward(&model_input, &contexts, &timestep, &features)?
                .to_dtype(DType::F32)?;
            sample = sample.broadcast_add(&velocity.affine((pair[1] - pair[0]) as f64, 0.0)?)?;
        }
        Ok(sample)
    }
}

fn gist_count(text_len: usize) -> usize {
    let ratio = ((text_len as f32 - 256.0) / (2048.0 - 256.0)).clamp(0.0, 1.0);
    ((((256.0 + ratio * 256.0) / 64.0).ceil() as usize) * 64).clamp(256, 512)
}

fn noise_mask(dev: &Device, noisy: usize, clean: usize, gist: usize) -> Result<Tensor> {
    let mut mask = Vec::with_capacity(noisy + clean + gist);
    mask.extend(std::iter::repeat_n(0f32, gist));
    mask.extend(std::iter::repeat_n(1f32, noisy));
    mask.extend(std::iter::repeat_n(0f32, clean));
    Ok(Tensor::from_vec(mask, (1, noisy + clean + gist, 1), dev)?)
}
