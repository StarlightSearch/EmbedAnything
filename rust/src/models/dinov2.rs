//! Implementation of the DINOv2 models from Meta Research.
//!
//! This module implements the DINOv2 vision transformer model from Meta AI Research.
//! DINOv2 is a self-supervised learning model that can learn visual features
//! without using any labeled data. See: ["DINOv2: Learning Robust Visual Features without Supervision"](https://github.com/facebookresearch/dinov2)
//!
//! ## Running an example with color map and CUDA
//!
//! ```bash
//! cargo run \
//!   --features cuda,depth_anything_v2 \
//!   --package candle-examples \
//!   --example depth_anything_v2 \
//!   -- --color-map \
//!   --image candle-examples/examples/yolo-v8/assets/bike.jpg
//! ```
//!
//! ## Running as an ImageNet classifier
//!
//! The model returns the probability for the image to belong to each of the 1000 ImageNet categories.
//!
//! <div align=center>
//!   <img src="https://github.com/huggingface/candle/raw/main/candle-examples/examples/yolo-v8/assets/bike.jpg" alt="" width=640>
//! </div>
//!
//! ```bash
//! cargo run \
//!   --example dinov2 \
//!   --release \
//!   -- --image candle-examples/examples/yolo-v8/assets/bike.jpg
//!
//! > mountain bike, all-terrain bike, off-roader: 43.67%
//! > bicycle-built-for-two, tandem bicycle, tandem: 33.20%
//! > crash helmet            : 13.23%
//! > unicycle, monocycle     : 2.44%
//! > maillot                 : 2.42%
//! ```
//!

use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)
    }
}

#[derive(Debug)]
pub struct Dinov2Embeddings {
    cls_token: Tensor,
    position_embeddings: Tensor,
    patch_embed: PatchEmbed,
    patch_size: usize,
}

impl Dinov2Embeddings {
    fn new(vb: VarBuilder, embed_dim: usize, img_size: usize, patch_size: usize) -> Result<Self> {
        let patch_embed = PatchEmbed::new(
            vb.pp("patch_embeddings"),
            img_size,
            patch_size,
            3,
            embed_dim,
        )?;
        let cls_token = vb.get((1, 1, embed_dim), "cls_token")?;
        let position_embeddings = vb.get(
            (1, patch_embed.num_patches + 1, embed_dim),
            "position_embeddings",
        )?;
        Ok(Self {
            cls_token,
            position_embeddings,
            patch_embed,
            patch_size,
        })
    }

    fn interpolate_pos_encoding(&self, xs: &Tensor, w: usize, h: usize) -> Result<Tensor> {
        let npatch = xs.dim(1)? - 1;
        let n = self.position_embeddings.dim(1)? - 1;
        let sqrt_n = (n as f64).sqrt();
        if npatch == n && w == h {
            return Ok(xs.clone());
        }
        let class_pos_embed = self.position_embeddings.i((.., ..1))?;
        let patch_pos_embed = self.position_embeddings.i((.., 1..))?;
        let dim = xs.dim(D::Minus1)?;
        let (w0, h0) = (
            (w / self.patch_size) as f64 + 0.1,
            (h / self.patch_size) as f64 + 0.1,
        );
        let patch_pos_embed = patch_pos_embed
            .reshape((1, sqrt_n as usize, sqrt_n as usize, dim))?
            .transpose(2, 3)?
            .transpose(1, 2)?;
        // This uses bicubic interpolation in the original implementation.
        let patch_pos_embed = patch_pos_embed.upsample_nearest2d(h0 as usize, w0 as usize)?;
        let el_count = patch_pos_embed.shape().elem_count();
        let patch_pos_embed =
            patch_pos_embed
                .transpose(1, 2)?
                .transpose(2, 3)?
                .reshape((1, el_count / dim, dim))?;
        Tensor::cat(&[&class_pos_embed, &patch_pos_embed], 1)
    }
}

impl Module for Dinov2Embeddings {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _nc, w, h) = xs.dims4()?;
        let xs = self.patch_embed.forward(xs)?;

        // Repeat (tile) cls_token along the batch dimension to match xs batch size before concat:
        let bsz = xs.dim(0)?;
        let cls_token = self.cls_token.repeat((bsz, 1, 1))?;
        let xs = Tensor::cat(&[&cls_token, &xs], 1)?;
        Tensor::broadcast_add(&xs, &self.interpolate_pos_encoding(&xs, w, h)?)
    }
}

#[derive(Debug)]
struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl SelfAttention {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize, qkv_bias: bool) -> Result<Self> {
        let query = linear(vb.pp("query"), dim, dim, qkv_bias)?;
        let key = linear(vb.pp("key"), dim, dim, qkv_bias)?;
        let value = linear(vb.pp("value"), dim, dim, qkv_bias)?;
        let scale = 1. / ((dim / num_heads) as f64).sqrt();
        let head_dim = dim / num_heads;
        Ok(Self {
            query,
            key,
            value,
            num_heads,
            scale,
            head_dim,
        })
    }
}

impl Module for SelfAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, q_len, _) = xs.dims3()?;
        let query_states = xs.apply(&self.query)?;
        let key_states = xs.apply(&self.key)?;
        let value_states = xs.apply(&self.value)?;

        let shape = (batch_size, q_len, self.num_heads, self.head_dim);
        let query_states = query_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let key_states = key_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let value_states = value_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let attn_weights = (query_states.matmul(&key_states.t()?)? * self.scale)?;

        // The original implementation upcasts to f32 but candle_nn::ops::softmax should handle this properly.
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_outputs = attn_weights
            .matmul(&value_states)?
            .transpose(1, 2)?
            .reshape((batch_size, q_len, ()))?;
        Ok(attn_outputs)
    }
}

#[derive(Debug)]
struct SelfOutput {
    dense: Linear,
}

impl SelfOutput {
    fn new(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        let dense = linear(vb.pp("dense"), hidden_size, hidden_size, true)?;
        Ok(Self { dense })
    }
}

impl Module for SelfOutput {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.dense.forward(xs)?;
        Ok(xs)
    }
}

#[derive(Debug)]
struct Attention {
    self_attention: SelfAttention,
    self_output: SelfOutput,
}

impl Attention {
    fn new(vb: VarBuilder, hidden_size: usize, num_heads: usize) -> Result<Self> {
        let self_attention = SelfAttention::new(vb.pp("attention"), hidden_size, num_heads, true)?;
        let self_output = SelfOutput::new(vb.pp("output"), hidden_size)?;
        Ok(Self {
            self_attention,
            self_output,
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.self_attention.forward(xs)?;
        let xs = self.self_output.forward(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug)]
struct LayerScale {
    gamma: Tensor,
}

impl LayerScale {
    fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let gamma = vb.get(dim, "lambda1")?;
        Ok(Self { gamma })
    }
}

impl Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.gamma)
    }
}

#[derive(Debug)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(vb: VarBuilder, in_features: usize, hidden_features: usize, bias: bool) -> Result<Self> {
        let out_features = in_features;
        let fc1 = linear(vb.pp("fc1"), in_features, hidden_features, bias)?;
        let fc2 = linear(vb.pp("fc2"), hidden_features, out_features, bias)?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?.gelu()?;
        let xs = self.fc2.forward(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    ls1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

impl Block {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let norm1 = layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let attn = Attention::new(vb.pp("attention"), dim, num_heads)?;
        let ls1 = LayerScale::new(vb.pp("layer_scale1"), dim)?;
        let norm2 = layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        let mlp = Mlp::new(vb.pp("mlp"), dim, dim * 4, true)?;
        let ls2 = LayerScale::new(vb.pp("layer_scale2"), dim)?;
        Ok(Self {
            norm1,
            attn,
            ls1,
            norm2,
            mlp,
            ls2,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self
            .ls1
            .forward(&self.attn.forward(&self.norm1.forward(xs)?)?)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .ls2
            .forward(&self.mlp.forward(&self.norm2.forward(&xs)?)?)?;
        xs + residual
    }
}

#[derive(Debug)]
struct PatchEmbed {
    proj: candle_nn::Conv2d,
    patch_size: (usize, usize),
    num_patches: usize,
}

impl PatchEmbed {
    fn new(
        vb: VarBuilder,
        img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
    ) -> Result<Self> {
        let config = candle_nn::Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(in_chans, embed_dim, patch_size, config, vb.pp("projection"))?;
        let num_patches = (img_size / patch_size) * (img_size / patch_size);
        Ok(Self {
            proj,
            patch_size: (patch_size, patch_size),
            num_patches,
        })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = xs.dims4()?;
        let (patch_h, patch_w) = self.patch_size;
        if (h % patch_h) != 0 {
            candle_core::bail!("image height {h} is not a multiple of patch height {patch_h}")
        }
        if (w % patch_w) != 0 {
            candle_core::bail!("image width {w} is not a multiple of patch width {patch_w}")
        }
        let xs = self.proj.forward(xs)?;
        let (b, c, h, w) = xs.dims4()?;
        // flatten embeddings.
        xs.reshape((b, c, h * w))?.transpose(1, 2)
    }
}

#[derive(Debug)]
pub struct DinoVisionTransformer {
    blocks: Vec<Block>,
    norm: LayerNorm,
    embeddings: Dinov2Embeddings,
}

impl DinoVisionTransformer {
    pub fn new(
        vb: VarBuilder,
        depth: usize,
        embed_dim: usize,
        num_heads: usize,
        img_size: usize,
        patch_size: usize,
    ) -> Result<Self> {
        let embeddings =
            Dinov2Embeddings::new(vb.pp("embeddings"), embed_dim, img_size, patch_size)?;

        let norm = layer_norm(embed_dim, 1e-5, vb.pp("layernorm"))?;
        let vb_b = vb.pp("encoder.layer");
        let blocks = (0..depth)
            .map(|i| Block::new(vb_b.pp(i.to_string()), embed_dim, num_heads))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            blocks,
            norm,
            embeddings,
        })
    }
}

impl Module for DinoVisionTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.embeddings.forward(xs)?;
        for blk in self.blocks.iter() {
            xs = blk.forward(&xs)?
        }
        let xs = self.norm.forward(&xs)?;
        let xs_norm_clstoken = xs.i((.., 0, ..))?;
        // let xs_norm_patchtokens = xs.i((.., 1.., ..))?.mean(1)?;

        // let xs = Tensor::cat(&[xs_norm_clstoken, xs_norm_patchtokens], D::Minus2)?;
        Ok(xs_norm_clstoken)
    }
}

pub fn vit_small(vb: VarBuilder) -> Result<DinoVisionTransformer> {
    DinoVisionTransformer::new(vb, 12, 384, 6, 518, 14)
}
