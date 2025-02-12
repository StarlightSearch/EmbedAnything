#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::collections::HashMap;

use crate::embeddings::embed::EmbeddingResult;
use crate::embeddings::local::text_embedding::get_model_info_by_hf_id;
use crate::embeddings::utils::tokenize_batch;
use crate::embeddings::{normalize_l2, select_device};
use anyhow::Error as E;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertForMaskedLM, BertModel, Config, DTYPE};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Repo;

use serde::Deserialize;
use tokenizers::{AddedToken, PaddingParams, Tokenizer, TruncationParams};

use super::pooling::{ModelOutput, PooledOutputType, Pooling};

pub trait BertEmbed {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error>;
}
#[derive(Debug, Deserialize, Clone)]
pub struct TokenizerConfig {
    pub max_length: Option<usize>,
    pub model_max_length: Option<usize>,
    pub mask_token: Option<String>,
    pub added_tokens_decoder: Option<HashMap<String, AddedToken>>,
}

impl TokenizerConfig {
    pub fn get_token_id_from_token(&self, token_string: &str) -> Option<i64> {
        // First check if added_tokens_decoder exists
        self.added_tokens_decoder
            .as_ref()?
            .iter()
            .find(|(_, value)| value.content == token_string)
            .and_then(|(key, _)| key.parse::<i64>().ok())
    }
}

pub struct BertEmbedder {
    pub model: BertModel,
    pub pooling: Pooling,
    pub tokenizer: Tokenizer,
}

impl Default for BertEmbedder {
    fn default() -> Self {
        Self::new(
            "sentence-transformers/all-MiniLM-L12-v2".to_string(),
            None,
            None,
        )
        .unwrap()
    }
}
impl BertEmbedder {
    pub fn new(model_id: String, revision: Option<String>, token: Option<&str>) -> Result<Self, E> {
        let model_info = get_model_info_by_hf_id(&model_id);
        let pooling = match model_info {
            Some(info) => info
                .model
                .get_default_pooling_method()
                .unwrap_or(Pooling::Mean),
            None => Pooling::Mean,
        };

        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = ApiBuilder::new()
                .with_token(token.map(|s| s.to_string()))
                .build()
                .unwrap();
            let api = match revision {
                Some(rev) => api.repo(Repo::with_revision(model_id, hf_hub::RepoType::Model, rev)),
                None => api.repo(hf_hub::Repo::new(
                    model_id.to_string(),
                    hf_hub::RepoType::Model,
                )),
            };
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = match api.get("model.safetensors") {
                Ok(safetensors) => safetensors,
                Err(_) => match api.get("pytorch_model.bin") {
                    Ok(pytorch_model) => pytorch_model,
                    Err(e) => {
                        return Err(anyhow::Error::msg(format!(
                            "Model weights not found. The weights should either be a `model.safetensors` or `pytorch_model.bin` file.  Error: {}",
                            e
                        )));
                    }
                },
            };

            (config, tokenizer, weights)
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

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
            .map_err(E::msg)?;

        let device = select_device();
        let vb = if weights_filename.ends_with("model.safetensors") {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        } else {
            println!("Can't find model.safetensors, loading from pytorch_model.bin");
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        };

        let model = BertModel::load(vb, &config)?;
        let tokenizer = tokenizer;

        Ok(BertEmbedder {
            model,
            tokenizer,
            pooling,
        })
    }
}

impl BertEmbed for BertEmbedder {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let batch_size = batch_size.unwrap_or(32);
        let mut encodings: Vec<EmbeddingResult> = Vec::new();

        for mini_text_batch in text_batch.chunks(batch_size) {
            let (token_ids, attention_mask) =
                tokenize_batch(&self.tokenizer, mini_text_batch, &self.model.device)?;

            let token_type_ids = token_ids.zeros_like()?;
            let embeddings: Tensor =
                self.model
                    .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

            let attention_mask = PooledOutputType::from(attention_mask);
            let attention_mask = Some(&attention_mask);
            let model_output = ModelOutput::Tensor(embeddings.clone());
            let pooled_output = match self.pooling {
                Pooling::Cls => self.pooling.pool(&model_output, None)?,
                Pooling::Mean => self.pooling.pool(&model_output, attention_mask)?,
            };
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

pub struct SparseBertEmbedder {
    pub tokenizer: Tokenizer,
    pub model: BertForMaskedLM,
    pub device: Device,
    pub dtype: DType,
}

impl SparseBertEmbedder {
    pub fn new(model_id: String, revision: Option<String>, token: Option<&str>) -> Result<Self, E> {
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = ApiBuilder::new()
                .with_token(token.map(|s| s.to_string()))
                .build()
                .unwrap();
            let api = match revision {
                Some(rev) => api.repo(Repo::with_revision(model_id, hf_hub::RepoType::Model, rev)),
                None => api.repo(hf_hub::Repo::new(
                    model_id.to_string(),
                    hf_hub::RepoType::Model,
                )),
            };
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = match api.get("model.safetensors") {
                Ok(safetensors) => safetensors,
                Err(_) => match api.get("pytorch_model.bin") {
                    Ok(pytorch_model) => pytorch_model,
                    Err(e) => {
                        return Err(anyhow::Error::msg(format!(
                            "Model weights not found. The weights should either be a `model.safetensors` or `pytorch_model.bin` file.  Error: {}",
                            e
                        )));
                    }
                },
            };

            (config, tokenizer, weights)
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

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
            .map_err(E::msg)?;

        let device = select_device();
        let vb = if weights_filename.ends_with("model.safetensors") {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        } else {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        };
        let model = BertForMaskedLM::load(vb, &config)?;
        let tokenizer = tokenizer;

        Ok(SparseBertEmbedder {
            model,
            tokenizer,
            device,
            dtype: DTYPE,
        })
    }
}

impl BertEmbed for SparseBertEmbedder {
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
            let token_type_ids = token_ids.zeros_like()?;
            let embeddings: Tensor =
                self.model
                    .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

            let batch_encodings = Tensor::log(
                &Tensor::try_from(1.0)?
                    .to_dtype(self.dtype)?
                    .to_device(&self.device)?
                    .broadcast_add(&embeddings.relu()?)?,
            )?;

            let batch_encodings = batch_encodings
                .broadcast_mul(&attention_mask.unsqueeze(2)?.to_dtype(self.dtype)?)?
                .max(1)?;
            let batch_encodings = normalize_l2(&batch_encodings)?;

            encodings.extend(
                batch_encodings
                    .to_vec2::<f32>()?
                    .into_iter()
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
    fn test_bert_embed() {
        let embedder = BertEmbedder::new(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            None,
            None,
        )
        .unwrap();
        let embeddings = embedder
            .embed(&["Hello, world!", "I am a rust programmer"], Some(32))
            .unwrap();
        let test_embeddings: Vec<f32> = vec![
            -3.81771736e-02,
            3.29111032e-02,
            -5.45938499e-03,
            1.43699143e-02,
        ];
        let embeddings = embeddings[0].to_dense().unwrap()[0..4].to_vec();
        assert!(
            (embeddings
                .iter()
                .zip(test_embeddings.iter())
                .all(|(a, b)| a.abs() - b.abs() < 1e-5))
        );
    }
}
