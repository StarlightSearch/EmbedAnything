use crate::{
    embeddings::{embed::EmbeddingResult, normalize_l2, select_device, utils::tokenize_batch},
    models::qwen3::{Config, Model},
};
use anyhow::Error;
use anyhow::Error as E;
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
        late_chunking: Option<bool>,
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

        let model = Model::new(&config, vb)?;

        Ok(Self {
            model: std::sync::RwLock::new(model),
            tokenizer,
            device,
        })
    }

    pub fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let batch_size = batch_size.unwrap_or(32);
        let mut encodings: Vec<EmbeddingResult> = Vec::new();

        for mini_text_batch in text_batch.chunks(batch_size) {
            let (token_ids, attention_mask) =
                tokenize_batch(&self.tokenizer, mini_text_batch, &self.device)?;
            let embeddings: Tensor = self
                .model
                .write()
                .unwrap()
                .forward(&token_ids, &attention_mask, 0)
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
    pub fn embed_late_chunking(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error>{
        let batch_size = batch_size.unwrap_or(32);
        let mut results = Vec::new();
        for mini_text_batch in text_batch.chunks(batch_size) {
            let tokens = self
                .tokenizer
                .encode_batch(mini_text_batch.to_vec(), true)
                .map_err(E::msg)?;

            let token_ids = tokens
                .iter()
                .map(|tokens| {
                    let tokens = tokens.get_ids().to_vec();
                    tokens
                })
                .collect::<Vec<_>>();

            let attention_mask = tokens
                .iter()
                .map(|tokens| {
                    let tokens = tokens.get_attention_mask().to_vec();
                    tokens
                })
                .collect::<Vec<_>>();

            // Keep track of original sequence lengths for later splitting
            let sequence_lengths: Vec<usize> = token_ids.iter().map(|seq| seq.len()).collect();
            let cumulative_seq_lengths: Vec<usize> = sequence_lengths
                .iter()
                .scan(0, |acc, &x| {
                    *acc += x;
                    Some(*acc)
                })
                .collect();

            // merge the token ids and attention mask into a single sequence
            let token_ids_merged = token_ids.concat();
            let attention_mask_merged = attention_mask.concat();

            // Convert to tensors
            let device = &self.device;
            let token_ids_tensor =
                Tensor::new(token_ids_merged.as_slice(), device)?.unsqueeze(0)?;
            let attention_mask_tensor =
                Tensor::new(attention_mask_merged.as_slice(), device)?.unsqueeze(0)?;

            // Run the model
            let embeddings: Tensor = self
                .model
                .write()
                .unwrap()
                .forward(&token_ids_tensor, &attention_mask_tensor, 0)
                .unwrap()
                .to_dtype(DType::F32)?;

            self.model.write().unwrap().clear_kv_cache();
            // Apply attention mask for pooling
            let attention_mask_tensor = PooledOutputType::from(attention_mask_tensor);

            for (i, &end_idx) in cumulative_seq_lengths.iter().enumerate() {
                let start_idx = if i == 0 {
                    0
                } else {
                    cumulative_seq_lengths[i - 1]
                };

                // Extract embeddings for this sequence
                let seq_embeddings = embeddings.narrow(1, start_idx, end_idx - start_idx)?;

                // Create attention mask for this sequence
                let seq_attention_mask =
                    attention_mask_tensor
                        .to_tensor()?
                        .narrow(1, start_idx, end_idx - start_idx)?;

                // Pool and normalize the embeddings for this sequence
                let model_output = ModelOutput::Tensor(seq_embeddings);
                let pooled_output = Pooling::Mean.pool(
                    &model_output,
                    Some(&PooledOutputType::from(seq_attention_mask)),
                )?;
                let pooled_tensor = pooled_output.to_tensor()?;
                let normalized = normalize_l2(pooled_tensor)?.squeeze(0)?;

                // Convert to vector
                let embedding_vec = normalized.to_vec1::<f32>().unwrap();
                results.push(EmbeddingResult::DenseVector(embedding_vec));
            }
        }

        Ok(results)
    }
}

impl Qwen3Embed for Qwen3Embedder {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
        late_chunking: Option<bool>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        if late_chunking.unwrap_or(false) {
            self.embed_late_chunking(text_batch, batch_size)
        } else {
            self.embed(text_batch, batch_size)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_embed() {
        let embedder = Qwen3Embedder::new(
            "Qwen/Qwen3-Embedding-0.6B",
            None,
            None,
            Some(crate::Dtype::F32),
        )
        .unwrap();
        let embeddings = embedder
            .embed(
                &["Hello, world!", "I am a rust programmer now"],
                Some(2),
                None,
            )
            .unwrap();
        let test_embeddings: Vec<f32> = vec![
            0.00555867,
            0.00928946,
            -0.00985782,
            -0.06393453,
            0.00829317,
            0.00708855,
        ];
        let first_embeddings = embeddings[0].to_dense().unwrap()[0..6].to_vec();
        println!("{:?}", first_embeddings);
        assert!(
            (first_embeddings
                .iter()
                .zip(test_embeddings.iter())
                .all(|(a, b)| (a.abs() - b.abs()).abs() < 1e-6))
        );
        let test_embeddings: Vec<f32> = vec![
            0.03579775,
            -0.04019123,
            -0.01412615,
            -0.05743032,
            0.04517555,
            -0.0193235,
        ];

        let second_embeddings = embeddings[1].to_dense().unwrap()[0..6].to_vec();
        println!("{:?}", second_embeddings);
        assert!(
            (second_embeddings
                .iter()
                .zip(test_embeddings.iter())
                .all(|(a, b)| (a.abs() - b.abs()).abs() < 1e-6))
        );
    }
}
