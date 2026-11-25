//! Qwen3-VL feature extraction for Bria RMBG-3.0. The decoder layers and
//! vision blocks follow Candle's Apache-2.0 Qwen3-VL implementation, with
//! hidden-state capture added for Bria's six selected layer bundles.

mod config;
mod conv3d_temporal_2;
mod text;
mod vision;

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::{DynamicImage, imageops::FilterType};
use tokenizers::Tokenizer;

use self::{config::Config, text::Qwen3VLTextModel, vision::Qwen3VLVisionModel};
use super::quantized::Checkpoint;

const MAX_TOKENS: usize = 4096;
const MAX_IMAGE_PIXELS: usize = 3_500_000;
const MIN_IMAGE_PIXELS: usize = 65_536;
const HIDDEN_LAYER_BUNDLES: [[usize; 3]; 6] = [
    [9, 20, 31],
    [5, 16, 27],
    [7, 18, 29],
    [9, 20, 31],
    [11, 22, 33],
    [13, 24, 35],
];
const INSTRUCTION: &str = "Generate a detailed grayscale alpha matte. Map the opaque foreground to white and the background to black. Produce soft, anti-aliased grayscale gradients at the edges of the subject to represent fine details and transparency.";

struct Qwen3VLModel {
    text: Qwen3VLTextModel,
    vision: Qwen3VLVisionModel,
    config: Config,
    mrope_section: [usize; 3],
}

impl Qwen3VLModel {
    fn new(config: Config, mrope_section: [usize; 3], vb: VarBuilder) -> Result<Self> {
        let vision = Qwen3VLVisionModel::new(&config.vision_config, vb.pp("model").pp("visual"))?;
        let text = Qwen3VLTextModel::new(&config.text_config, vb.clone())?;
        Ok(Self {
            text,
            vision,
            config,
            mrope_section,
        })
    }

    fn forward(
        &self,
        input_ids: &[u32],
        pixels: &Tensor,
        grid: &Tensor,
        image_token_span: std::ops::Range<usize>,
        causal_mask: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let device = pixels.device();
        let ids = Tensor::from_vec(input_ids.to_vec(), (1, input_ids.len()), device)?;
        let mut embeddings = self.text.embed_tokens(&ids)?;
        let (image_embeds, deepstack) = self.vision.forward(pixels, grid)?;
        let image_embeds = image_embeds
            .to_device(device)?
            .to_dtype(embeddings.dtype())?;
        let expected = image_token_span.end - image_token_span.start;
        if image_embeds.dim(0)? != expected {
            bail!(
                "Qwen image processor produced {} visual tokens for {} image placeholders",
                image_embeds.dim(0)?,
                expected
            );
        }
        embeddings = embeddings.slice_assign(
            &[
                0..1,
                image_token_span.start..image_token_span.end,
                0..self.config.text_config.hidden_size,
            ],
            &image_embeds.unsqueeze(0)?,
        )?;
        let mut visual_mask = vec![0u8; input_ids.len()];
        visual_mask[image_token_span.clone()].fill(1);
        let visual_mask = Tensor::from_vec(visual_mask, (1, input_ids.len()), device)?;
        let grid = grid.to_vec2::<u32>()?;
        let merge = self.config.vision_config.spatial_merge_size;
        let position_ids = multimodal_position_ids(
            input_ids.len(),
            image_token_span.clone(),
            grid[0][1] as usize / merge,
            grid[0][2] as usize / merge,
            device,
        )?;
        Ok(self.text.forward_embeds_hidden_states(
            embeddings,
            Some(causal_mask),
            &position_ids,
            self.mrope_section,
            Some(&visual_mask),
            Some(&deepstack),
        )?)
    }
}

/// Native Qwen3-VL encoder that creates the six `[1, 6, tokens, 7680]`
/// feature bundles expected by Bria's Fibo2 transformer.
pub struct Qwen3Encoder {
    model: Qwen3VLModel,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

impl Qwen3Encoder {
    /// Loads Bria's quantized Qwen3-VL checkpoint and its public model metadata.
    pub fn load<P: AsRef<Path>>(
        weights: P,
        config_path: P,
        tokenizer_path: P,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let config_bytes =
            std::fs::read(config_path.as_ref()).context("reading Qwen3-VL config")?;
        let raw_config: serde_json::Value = serde_json::from_slice(&config_bytes)?;
        let config: Config = serde_json::from_value(raw_config.clone())?;
        let section = raw_config
            .pointer("/text_config/rope_scaling/mrope_section")
            .or_else(|| raw_config.pointer("/text_config/rope_parameters/mrope_section"))
            .and_then(serde_json::Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.as_u64().unwrap_or_default() as usize)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![24, 20, 20]);
        let mrope_section: [usize; 3] = section
            .try_into()
            .map_err(|_| anyhow::anyhow!("Qwen mrope_section must contain three values"))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .context("opening Qwen3-VL tokenizer")?;
        if config.text_config.num_hidden_layers <= 35 {
            bail!("Qwen3-VL config has too few text layers for Bria's feature selection");
        }
        // Candle uses a 14x14 default for older Qwen vision checkpoints; load
        // the patch size and merge factor from Qwen3-VL's own config instead.
        let checkpoint = Checkpoint::open(&[weights])?;
        let vb = checkpoint.into_var_builder(dtype, device);
        let model = Qwen3VLModel::new(config, mrope_section, vb)?;
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            dtype,
        })
    }

    /// Produces Bria's feature tensor for the fixed RMBG matte instruction.
    pub fn encode(&self, image: &DynamicImage) -> Result<Tensor> {
        let patch = self.model.config.vision_config.patch_size;
        let merge = self.model.config.vision_config.spatial_merge_size;
        let factor = patch * merge;
        let rgb = image.to_rgb8();
        let (width, height) = smart_resize(rgb.width() as usize, rgb.height() as usize, factor)?;
        let resized =
            image::imageops::resize(&rgb, width as u32, height as u32, FilterType::CatmullRom);
        let (pixel_values, grid, merged_tokens) =
            qwen_image_patches(&resized, patch, merge, &self.device)?;

        let marker = format!(
            "<|vision_start|>{}<|vision_end|>",
            "<|image_pad|>".repeat(merged_tokens)
        );
        // Keep the key order used by Diffusers' `_get_edit_prompt`: image
        // first, then edit_instruction. `serde_json::Map` sorts keys, which
        // changes the Qwen conditioning if used here.
        let prompt_json = format!(
            "{{\"image\":{},\"edit_instruction\":{}}}",
            serde_json::to_string(&marker)?,
            serde_json::to_string(INSTRUCTION)?
        );
        let prompt = format!("<|im_start|>user\n{prompt_json}<|im_end|>\n<|im_start|>assistant\n");
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let input_ids = encoding.get_ids().to_vec();
        if input_ids.len() > MAX_TOKENS {
            bail!(
                "Qwen input is {} tokens; maximum is {MAX_TOKENS}",
                input_ids.len()
            );
        }
        let image_token = self.model.config.image_token_id;
        let positions = input_ids
            .iter()
            .enumerate()
            .filter_map(|(index, token)| (*token == image_token).then_some(index))
            .collect::<Vec<_>>();
        if positions.len() != merged_tokens {
            bail!(
                "tokenizer emitted {} image markers for {merged_tokens} Qwen visual tokens",
                positions.len()
            );
        }
        let start = *positions
            .first()
            .context("Qwen prompt omitted image tokens")?;
        let end = positions.last().copied().unwrap() + 1;
        if end - start != positions.len() {
            bail!("Qwen image marker tokens are not contiguous in the prepared prompt");
        }
        // Keep this explicit so the same sequence is used by the image and text
        // paths even if a future tokenizer adds special tokens automatically.
        let causal_mask = causal_mask(input_ids.len(), self.dtype, &self.device)?;
        let states =
            self.model
                .forward(&input_ids, &pixel_values, &grid, start..end, &causal_mask)?;
        feature_bundles(&states, self.dtype)
    }
}

fn causal_mask(seq: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(seq * seq);
    for row in 0..seq {
        for col in 0..seq {
            values.push(if col > row { f32::NEG_INFINITY } else { 0.0 });
        }
    }
    Ok(Tensor::from_vec(values, (1, 1, seq, seq), device)?.to_dtype(dtype)?)
}

fn multimodal_position_ids(
    seq_len: usize,
    image_span: std::ops::Range<usize>,
    grid_h: usize,
    grid_w: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut axes: [Vec<u32>; 3] = std::array::from_fn(|_| Vec::with_capacity(seq_len));
    for position in 0..image_span.start {
        for axis in &mut axes {
            axis.push(position as u32);
        }
    }
    let offset = image_span.start as u32;
    for row in 0..grid_h {
        for column in 0..grid_w {
            axes[0].push(offset);
            axes[1].push(offset + row as u32);
            axes[2].push(offset + column as u32);
        }
    }
    let mut next = offset + grid_h.max(grid_w) as u32;
    for _ in image_span.end..seq_len {
        for axis in &mut axes {
            axis.push(next);
        }
        next += 1;
    }
    if axes.iter().any(|axis| axis.len() != seq_len) {
        bail!("Qwen multimodal position sequence does not match tokenizer output");
    }
    Ok(Tensor::from_vec(
        axes.into_iter().flatten().collect::<Vec<_>>(),
        (3, seq_len),
        device,
    )?)
}

fn feature_bundles(states: &[Tensor], dtype: DType) -> Result<Tensor> {
    let mut bundles = Vec::with_capacity(HIDDEN_LAYER_BUNDLES.len());
    for layers in HIDDEN_LAYER_BUNDLES {
        let normalized = layers
            .iter()
            .map(|&index| {
                let state = states
                    .get(index)
                    .with_context(|| format!("Qwen hidden state {index} is missing"))?
                    .to_dtype(DType::F32)?;
                let rms = state
                    .sqr()?
                    .mean_keepdim(D::Minus1)?
                    .affine(1.0, 1e-6)?
                    .sqrt()?;
                Ok(state.broadcast_div(&rms)?.to_dtype(dtype)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let parts = normalized.iter().collect::<Vec<_>>();
        bundles.push(Tensor::cat(&parts, D::Minus1)?);
    }
    let bundles = bundles.iter().collect::<Vec<_>>();
    Ok(Tensor::stack(&bundles, 1)?)
}

fn smart_resize(width: usize, height: usize, factor: usize) -> Result<(usize, usize)> {
    if width == 0 || height == 0 || width.max(height) > 200 * width.min(height) {
        bail!("Qwen image aspect ratio must be at most 200:1");
    }
    let round = |value: usize| ((value as f64 / factor as f64).round() as usize).max(1) * factor;
    let mut resized_w = round(width);
    let mut resized_h = round(height);
    let area = resized_w * resized_h;
    if area > MAX_IMAGE_PIXELS {
        let scale = (area as f64 / MAX_IMAGE_PIXELS as f64).sqrt();
        resized_w = (((resized_w as f64 / scale) as usize) / factor).max(1) * factor;
        resized_h = (((resized_h as f64 / scale) as usize) / factor).max(1) * factor;
    } else if area < MIN_IMAGE_PIXELS {
        let scale = (MIN_IMAGE_PIXELS as f64 / area as f64).sqrt();
        resized_w = ((resized_w as f64 * scale / factor as f64).ceil() as usize).max(1) * factor;
        resized_h = ((resized_h as f64 * scale / factor as f64).ceil() as usize).max(1) * factor;
    }
    Ok((resized_w, resized_h))
}

fn qwen_image_patches(
    image: &image::RgbImage,
    patch: usize,
    merge: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, usize)> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let grid_w = width / patch;
    let grid_h = height / patch;
    let merged_tokens = (grid_w / merge) * (grid_h / merge);
    let patch_values = 3 * 2 * patch * patch;
    let mut values = Vec::with_capacity(grid_w * grid_h * patch_values);
    let mean = [0.5f32; 3];
    let std = [0.5f32; 3];
    for merged_y in 0..grid_h / merge {
        for merged_x in 0..grid_w / merge {
            for inner_y in 0..merge {
                for inner_x in 0..merge {
                    let block_y = merged_y * merge + inner_y;
                    let block_x = merged_x * merge + inner_x;
                    for channel in 0..3 {
                        for _frame in 0..2 {
                            for y in 0..patch {
                                for x in 0..patch {
                                    let pixel = image.get_pixel(
                                        (block_x * patch + x) as u32,
                                        (block_y * patch + y) as u32,
                                    );
                                    values.push(
                                        (pixel[channel] as f32 / 255.0 - mean[channel])
                                            / std[channel],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let grid = Tensor::from_vec(vec![1u32, grid_h as u32, grid_w as u32], (1, 3), device)?;
    Ok((
        Tensor::from_vec(values, (grid_h * grid_w, patch_values), device)?,
        grid,
        merged_tokens,
    ))
}
