use crate::{
    embeddings::{embed::EmbeddingResult, normalize_l2, select_device, utils::tokenize_batch},
    models::gemma3::{Config, Model},
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
}

impl Gemma3Embedder {
    pub fn new(
        model_id: &str,
        revision: Option<String>,
        token: Option<&str>,
        dtype: Option<crate::Dtype>,
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

        Ok(Self {
            model: std::sync::RwLock::new(model),
            tokenizer,
            device,
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
            
            // Forward pass through the model - Gemma3 forward takes input_ids and seqlen_offset
            let embeddings: Tensor = self
                .model
                .write()
                .unwrap()
                .forward(&token_ids, 0)
                .unwrap()
                .to_dtype(DType::F32)?;

            self.model.write().unwrap().clear_kv_cache();
            
            // Convert attention_mask to the expected format for pooling
            let attention_mask = PooledOutputType::from(attention_mask);
            let attention_mask = Some(&attention_mask);
            let model_output = ModelOutput::Tensor(embeddings.clone());
            let pooled_output = Pooling::Mean
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
    fn test_gemma3_embed() {
        // Test with a small Gemma3 model if available
        // Note: You may need to adjust the model_id based on available models
        let embedder = Gemma3Embedder::new(
            "google/embeddinggemma-300m", // Adjust model ID as needed
            None,
            None,
            Some(crate::Dtype::F32),
        );
        
        // Only run test if model is available
        if let Ok(embedder) = embedder {
            let embeddings = embedder
                .embed(
                    &["Hello, world!", "I am a rust programmer now"],
                    Some(2),
                    None,
                )
                .unwrap();
            
            // Basic assertions - embeddings should not be empty
            assert!(!embeddings.is_empty());
            assert_eq!(embeddings.len(), 2);
            
            // Check that embeddings have reasonable dimensions
            for embedding in &embeddings {
                let dense = embedding.to_dense().unwrap();
                assert!(!dense.is_empty());
                assert!(dense.len() > 100); // Gemma3 should have reasonable embedding dimensions
            }
        }
    }
}