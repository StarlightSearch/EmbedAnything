use crate::{
    embeddings::{embed::EmbeddingResult, normalize_l2, select_device, utils::tokenize_batch},
    models::qwen3::{Config, Model},
};
use anyhow::Error;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::ApiBuilder, Repo};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use super::{
    colpali::hub_load_safetensors,
    pooling::{ModelOutput, PooledOutputType, Pooling},
};

pub trait Qwen3Embed {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error>;
}

pub struct Qwen3Embedder {
    pub model: std::sync::RwLock<Model>,
    pub tokenizer: Tokenizer,
    pub device: Device,
}

impl Qwen3Embedder {
    pub fn new(
        model_id: &str,
        revision: Option<String>,
        token: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        let api = ApiBuilder::new()
            .with_token(token.map(|s| s.to_string()))
            .build()
            .unwrap();

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
            ..Default::default()
        };
        let trunc = TruncationParams {
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            max_length: config.max_position_embeddings as usize,
            ..Default::default()
        };

        tokenizer
            .with_padding(Some(pp))
            .with_truncation(Some(trunc))
            .map_err(Error::msg)?;

        let device = select_device();

        let vb = match weights_filename {
            Ok(weights) => unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights], DType::BF16, &device)?
            },
            Err(_) => {
                let weights = hub_load_safetensors(&repo, "model.safetensors.index.json")?;
                unsafe { VarBuilder::from_mmaped_safetensors(&weights, DType::BF16, &device)? }
            }
        };

        let model = Model::new(&config, vb)?;

        Ok(Self {
            model: std::sync::RwLock::new(model),
            tokenizer,
            device,
        })
    }
}

impl Qwen3Embed for Qwen3Embedder {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let batch_size = batch_size.unwrap_or(32);
        let mut encodings: Vec<EmbeddingResult> = Vec::new();

        for mini_text_batch in text_batch.chunks(batch_size) {
            let (token_ids, attention_mask) =
                tokenize_batch(&self.tokenizer, mini_text_batch, &self.device)?;
            let embeddings: Tensor = self.model.write().unwrap()
                .forward(&token_ids, 0)
                .unwrap()
                .to_dtype(DType::F32)?;

            self.model.write().unwrap().clear_kv_cache();
            let attention_mask = PooledOutputType::from(attention_mask);
            let attention_mask = Some(&attention_mask);
            let model_output = ModelOutput::Tensor(embeddings.clone());
            let pooled_output = Pooling::LastToken
                .pool(&model_output, attention_mask)
                .unwrap();
            let pooled_output = pooled_output.to_tensor()?;
            let embeddings = normalize_l2(pooled_output)?;
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
    fn test_qwen3_embed() {
        let mut embedder = Qwen3Embedder::new("Qwen/Qwen3-Embedding-0.6B", None, None).unwrap();
        let embeddings = embedder
            .embed(&["Hello, world!", "I am a rust programmer now"], Some(1))
            .unwrap();
        // let test_embeddings: Vec<f32> = vec![
        //     -3.81771438e-02,
        //     3.29110473e-02,
        //     -5.45941433e-03,
        //     1.43699292e-02,
        //     -4.02910188e-02,
        //     -1.16532497e-01,
        // ];
        // let embeddings = embeddings[0].to_dense().unwrap()[0..6].to_vec();
        // println!("{:?}", embeddings);
        // assert!(
        //     (embeddings
        //         .iter()
        //         .zip(test_embeddings.iter())
        //         .all(|(a, b)| (a.abs() - b.abs()).abs() < 1e-6))
        // );
    }
}
