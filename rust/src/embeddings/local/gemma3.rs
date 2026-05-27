use crate::{
    embeddings::{embed::EmbeddingResult, normalize_l2, select_device, utils::tokenize_batch},
    models::gemma3::{Config, Model},
};
use anyhow::Error;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use hf_hub::{api::sync::ApiBuilder, Repo};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use super::{
    colpali::hub_load_safetensors,
    pooling::{ModelOutput, PooledOutputType, Pooling},
};

pub trait Gemma3Embed {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
        late_chunking: Option<bool>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error>;
}

pub struct Gemma3Embedder {
    pub model: std::sync::RwLock<Model>,
    pub tokenizer: Tokenizer,
    pub device: Device,
    dense1: Linear,
    dense2: Linear,
}

/// Loads a sentence-transformers `Dense` module (a single no-bias `nn.Linear`
/// stored under the key `linear.weight`) from a safetensors file in the repo.
fn load_dense(
    repo: &hf_hub::api::sync::ApiRepo,
    path: &str,
    device: &Device,
) -> Result<Linear, anyhow::Error> {
    let weights_path = repo.get(path)?;
    let mut tensors = candle_core::safetensors::load(weights_path, device)?;
    let weight = tensors
        .remove("linear.weight")
        .ok_or_else(|| anyhow::anyhow!("missing 'linear.weight' in {path}"))?
        .to_dtype(DType::F32)?;
    Ok(Linear::new(weight, None))
}

impl Gemma3Embedder {
    pub fn new(
        model_id: &str,
        revision: Option<String>,
        token: Option<&str>,
        dtype: Option<crate::Dtype>,
    ) -> Result<Self, anyhow::Error> {
        let mut api_builder = ApiBuilder::from_env();
        if let Some(token) = token {
            api_builder = api_builder.with_token(Some(token.to_string()));
        }
        let api = api_builder.build()?;

        let repo = match revision {
            Some(rev) => api.repo(Repo::with_revision(
                model_id.to_string(),
                hf_hub::RepoType::Model,
                rev,
            )),
            None => api.repo(hf_hub::Repo::new(
                model_id.to_string(),
                hf_hub::RepoType::Model,
            )),
        };
        let (config_filename, tokenizer_filename, weights_filename) = {
            let config = repo.get("config.json")?;
            let tokenizer = repo.get("tokenizer.json")?;
            let weights = repo.get("model.safetensors");

            (config, tokenizer, weights)
        };

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Left,
            ..Default::default()
        };
        let trunc = TruncationParams {
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            max_length: 1024,
            ..Default::default()
        };

        tokenizer
            .with_padding(Some(pp))
            .with_truncation(Some(trunc))
            .map_err(Error::msg)?;

        let device = select_device();

        let dtype = match dtype {
            Some(crate::Dtype::F16) => DType::F16,
            Some(crate::Dtype::F32) => DType::F32,
            Some(crate::Dtype::BF16) => DType::BF16,
            _ => DType::F32,
        };

        let vb = match weights_filename {
            Ok(weights) => unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)?
            },
            Err(_) => {
                let weights = hub_load_safetensors(&repo, "model.safetensors.index.json")?;
                unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device)? }
            }
        };

        let model = Model::new(false, &config, vb)?; // use_flash_attn = false by default

        // EmbeddingGemma applies a 768 -> 3072 -> 768 dense projection head
        // (sentence-transformers modules 2_Dense / 3_Dense, both no-bias with
        // Identity activation) between mean pooling and L2 normalization. These
        // weights live outside model.safetensors and must be applied separately.
        let dense1 = load_dense(&repo, "2_Dense/model.safetensors", &device)?;
        let dense2 = load_dense(&repo, "3_Dense/model.safetensors", &device)?;

        Ok(Self {
            model: std::sync::RwLock::new(model),
            tokenizer,
            device,
            dense1,
            dense2,
        })
    }
}

impl Gemma3Embed for Gemma3Embedder {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
        _late_chunking: Option<bool>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let batch_size = batch_size.unwrap_or(32);
        let mut encodings: Vec<EmbeddingResult> = Vec::new();

        for mini_text_batch in text_batch.chunks(batch_size) {
            let (token_ids, attention_mask) =
                tokenize_batch(&self.tokenizer, mini_text_batch, &self.device)?;

            // Forward pass through the model. EmbeddingGemma uses bidirectional
            // attention, so the padding mask must be passed in (not just used for pooling).
            let embeddings: Tensor = self
                .model
                .write()
                .unwrap()
                .forward(&token_ids, Some(&attention_mask), 0)
                .unwrap()
                .to_dtype(DType::F32)?;

            self.model.write().unwrap().clear_kv_cache();

            // Convert attention_mask to the expected format for pooling
            let attention_mask = PooledOutputType::from(attention_mask);
            let attention_mask = Some(&attention_mask);
            let model_output = ModelOutput::Tensor(embeddings.clone());
            let pooled_output = Pooling::Mean.pool(&model_output, attention_mask).unwrap();
            let pooled_output = pooled_output.to_tensor()?;
            let projected = self.dense2.forward(&self.dense1.forward(pooled_output)?)?;
            let embeddings = normalize_l2(&projected)?;
            let batch_encodings = embeddings.to_vec2::<f32>()?;

            encodings.extend(
                batch_encodings
                    .iter()
                    .map(|x| EmbeddingResult::DenseVector(x.to_vec())),
            );
        }

        Ok(encodings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma3_embed() {
        let embedder = Gemma3Embedder::new(
            "google/embeddinggemma-300m",
            None,
            None,
            Some(crate::Dtype::F32),
        );
        let Ok(embedder) = embedder else {
            return;
        };
        let embeddings = embedder
            .embed(
                &["Hello, world!", "I am a rust programmer now"],
                Some(2),
                None,
            )
            .unwrap();

        // Exercise full pipeline and check first 6 dims.
        // bidirectional transformer -> mean pool
        // -> dense(768->3072) -> dense(3072->768) -> L2 norm
        let test_embeddings: Vec<f32> = vec![
            -0.17819002,
            0.02147517,
            0.06739803,
            -0.03160102,
            0.02198322,
            -0.00981485,
        ];
        let first_embeddings = embeddings[0].to_dense().unwrap()[0..6].to_vec();
        println!("{:?}", first_embeddings);
        assert!(first_embeddings
		.iter()
		.zip(test_embeddings.iter())
		.all(|(a, b)| (a.abs() - b.abs()).abs() < 1e-6));
        let test_embeddings: Vec<f32> = vec![
            -0.19414055,
            -0.01050718,
            0.02919163,
            0.0027125,
            0.037645,
            0.04710986,
        ];
        let second_embeddings = embeddings[1].to_dense().unwrap()[0..6].to_vec();
        println!("{:?}", second_embeddings);
        assert!(second_embeddings
		.iter()
		.zip(test_embeddings.iter())
		.all(|(a, b)| (a.abs() - b.abs()).abs() < 1e-6));
    }
}
