#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::{ops::Mul, sync::RwLock};

use anyhow::{Error as E, Result};
use hf_hub::{api::sync::Api, Repo};
use ndarray::{Array2, Axis};
use ort::{
    execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::embeddings::{embed::EmbeddingResult, utils::tokenize_batch_ndarray};

use super::bert::{BertEmbed, TokenizerConfig};

pub trait ColbertEmbed {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
        is_doc: bool,
    ) -> Result<Vec<EmbeddingResult>, E>;
}

#[derive(Debug)]
pub struct OrtColbertEmbedder {
    pub tokenizer: Tokenizer,
    pub model: RwLock<Session>,
    pub document_marker_token_id: Option<i64>,
    pub query_marker_token_id: Option<i64>,
    pub pad_id: Option<i64>,
    pub mask_token: Option<String>,
}

impl OrtColbertEmbedder {
    pub fn new(
        model_id: Option<&str>,
        revision: Option<&str>,
        path_in_repo: Option<&str>,
    ) -> Result<Self, E> {
        let path_in_repo = path_in_repo.unwrap_or("model.onnx");
        let hf_model_id = match model_id {
            Some(id) => id,
            None => return Err(anyhow::anyhow!("Please provide hf model id")),
        };

        let (_, tokenizer_filename, weights_filename, tokenizer_config_filename, data_filename) = {
            let api = Api::new().unwrap();
            let api = match revision {
                Some(rev) => api.repo(Repo::with_revision(
                    hf_model_id.to_string(),
                    hf_hub::RepoType::Model,
                    rev.to_string(),
                )),
                None => api.repo(hf_hub::Repo::new(
                    hf_model_id.to_string(),
                    hf_hub::RepoType::Model,
                )),
            };
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let tokenizer_config = api.get("tokenizer_config.json")?;

            let weights = api.get(path_in_repo);
            let data = api.get(format!("{path_in_repo}_data").as_str());

            (config, tokenizer, weights, tokenizer_config, data)
        };

        let weights_filename = match weights_filename {
            Ok(weights) => weights,
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Specified ONNX weights not found for the model. {}",
                    e
                ));
            }
        };

        let _ = data_filename.ok();

        let tokenizer_config = std::fs::read_to_string(tokenizer_config_filename)?;
        let tokenizer_config: TokenizerConfig = serde_json::from_str(&tokenizer_config)?;
        // Set max_length to the minimum of max_length and model_max_length if both are present
        let max_length = match (
            tokenizer_config.max_length,
            tokenizer_config.model_max_length,
        ) {
            (Some(max_len), Some(model_max_len)) => std::cmp::min(max_len, model_max_len),
            (Some(max_len), None) => max_len,
            (None, Some(model_max_len)) => model_max_len,
            (None, None) => 128,
        };

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let mask_token = tokenizer_config.clone().mask_token;
        let pad_id = match mask_token.clone() {
            Some(mask_token) => tokenizer_config.get_token_id_from_token(&mask_token),
            None => None,
        };
        let document_marker_token_id = tokenizer_config.get_token_id_from_token("[DocumentMarker]");
        let query_marker_token_id = tokenizer_config.get_token_id_from_token("[QueryMarker]");

        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        let trunc = TruncationParams {
            max_length,
            ..Default::default()
        };
        tokenizer
            .with_padding(Some(pp))
            .with_truncation(Some(trunc))
            .unwrap();

        let cuda = CUDAExecutionProvider::default();

        if !cuda.is_available()? {
            eprintln!("CUDAExecutionProvider is not available");
        } else {
            println!("Session is using CUDAExecutionProvider");
        }

        let threads = std::thread::available_parallelism().unwrap().get();
        let model = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
                CoreMLExecutionProvider::default().build(),
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(weights_filename)?;

        Ok(OrtColbertEmbedder {
            tokenizer,
            model: RwLock::new(model),
            document_marker_token_id,
            query_marker_token_id,
            pad_id,
            mask_token,
        })
    }
}

impl ColbertEmbed for OrtColbertEmbedder {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
        is_doc: bool,
    ) -> Result<Vec<EmbeddingResult>, E> {
        let mut tokenizer = self.tokenizer.clone();

        if !is_doc && self.mask_token.is_some() && self.pad_id.is_some() {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::Fixed(32),
                pad_token: self.mask_token.clone().unwrap(),
                pad_id: self.pad_id.unwrap() as u32,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        let batch_size = batch_size.unwrap_or(32);

        let model_guard = self.model.read().unwrap();
        let (input_names, output_name) = {
            let names = model_guard
                .inputs
                .iter()
                .map(|input| input.name.to_string())
                .collect::<Vec<_>>();
            let out_name = model_guard.outputs.first().unwrap().name.to_string();
            (names, out_name)
        };

        drop(model_guard);

        let mut model_guard = self.model.write().unwrap();

        let encodings = text_batch
            .chunks(batch_size)
            .flat_map(|mini_text_batch| -> Result<Vec<EmbeddingResult>, E> {
                let (mut input_ids, mut attention_mask): (Array2<i64>, Array2<i64>) =
                    tokenize_batch_ndarray(&tokenizer, mini_text_batch)?;
                let token_type_ids: Array2<i64> = Array2::zeros(input_ids.raw_dim());

                // Insert marker token after the first token if available
                if let Some(marker_id) = if is_doc {
                    self.document_marker_token_id
                } else {
                    self.query_marker_token_id
                } {
                    for (mut row, mut mask_row) in input_ids.rows_mut().into_iter().zip(attention_mask.rows_mut().into_iter()) {
                        // Shift all tokens after position 0 one position to the right
                        for i in (2..row.len()).rev() {
                            row[i] = row[i - 1];
                            mask_row[i] = mask_row[i - 1];
                        }
                        // Insert marker token at position 1 (after first token)
                        row[1] = marker_id;
                        mask_row[1] = 1;
                    }
                }
                let input_ids_tensor = ort::value::TensorRef::from_array_view(&input_ids)?;
                let attention_mask_tensor = ort::value::TensorRef::from_array_view(&attention_mask)?;
                let mut inputs =
                    ort::inputs!["input_ids" => input_ids_tensor, "attention_mask" => attention_mask_tensor.clone()];
                if input_names.iter().any(|x| x == "token_type_ids") {
                    inputs.push((
                        "token_type_ids".into(),
                        Value::from_array(token_type_ids.clone())?.into(),
                    ));
                }
                let outputs = model_guard.run(inputs)?;
                let embeddings = outputs[output_name.as_str()].try_extract_array::<f32>()?.to_owned().into_dimensionality::<ndarray::Ix3>()?;

                let attention_mask = attention_mask.mapv(|x| x as f32).insert_axis(Axis(2));
                let embeddings = embeddings.mul(attention_mask);
                let (batch_size, seq_len, embed_dim) = embeddings.dim();
                // Normalize each token's embedding vector
                let normalized_embeddings = embeddings.to_owned().to_shape((batch_size * seq_len, embed_dim))?
                    .outer_iter()
                    .map(|vector| {
                        let norm = (vector.dot(&vector)).sqrt();
                        vector.map(|&x| x / (norm + 1e-10)).to_vec()
                    })
                    .collect::<Vec<_>>();

                // Reshape back to [Batch, Seq, Embedding Dimension]
                let normalized_embeddings = normalized_embeddings
                    .chunks(seq_len)
                    .map(|batch| batch.to_vec())
                    .collect::<Vec<_>>();

                let e = normalized_embeddings
                    .into_iter()
                    .map(EmbeddingResult::MultiVector)
                    .collect::<Vec<_>>();

                Ok(e)
            })
            .flatten()
            .collect::<Vec<_>>();

        Ok(encodings)
    }
}

impl BertEmbed for OrtColbertEmbedder {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
        _late_chunking: Option<bool>,
    ) -> Result<Vec<EmbeddingResult>, E> {
        let batch_size = batch_size.unwrap_or(32);
        let mut model_guard = self.model.write().unwrap();

        let (input_names, output_name) = {
            let names = model_guard
                .inputs
                .iter()
                .map(|input| input.name.to_string())
                .collect::<Vec<_>>();
            let out_name = model_guard.outputs.first().unwrap().name.to_string();
            (names, out_name)
        };
        let encodings = text_batch
            .chunks(batch_size)
            .flat_map(|mini_text_batch| -> Result<Vec<EmbeddingResult>, E> {
                let (input_ids, attention_mask): (Array2<i64>, Array2<i64>) =
                        tokenize_batch_ndarray(&self.tokenizer, mini_text_batch)?;
                let token_type_ids: Array2<i64> = Array2::zeros(input_ids.raw_dim());

                let input_ids_tensor = ort::value::TensorRef::from_array_view(&input_ids)?;
                let attention_mask_tensor = ort::value::TensorRef::from_array_view(&attention_mask)?;
                let mut inputs =
                    ort::inputs!["input_ids" => input_ids_tensor, "attention_mask" => attention_mask_tensor.clone()];
                if input_names.iter().any(|x| x == "token_type_ids") {
                    inputs.push((
                        "token_type_ids".into(),
                        Value::from_array(token_type_ids.clone())?.into(),
                    ));
                }
                let outputs = model_guard.run(inputs)?;
                let embeddings = outputs[output_name.as_str()].try_extract_array::<f32>()?.to_owned().into_dimensionality::<ndarray::Ix3>()?;

                let attention_mask = attention_mask.mapv(|x| x as f32).insert_axis(Axis(2));
                let embeddings = embeddings.mul(attention_mask);
                let (batch_size, seq_len, embed_dim) = embeddings.dim();
                // Normalize each token's embedding vector
                let normalized_embeddings = embeddings.to_owned().to_shape((batch_size * seq_len, embed_dim))?
                    .outer_iter()
                    .map(|vector| {
                        let norm = (vector.dot(&vector)).sqrt();
                        vector.map(|&x| x / (norm + 1e-10)).to_vec()
                    })
                    .collect::<Vec<_>>();

                // Reshape back to [Batch, Seq, Embedding Dimension]
                let normalized_embeddings = normalized_embeddings
                    .chunks(seq_len)
                    .map(|batch| batch.to_vec())
                    .collect::<Vec<_>>();

                let e = normalized_embeddings
                    .into_iter()
                    .map(EmbeddingResult::MultiVector)
                    .collect::<Vec<_>>();

                Ok(e)
            })
            .flatten()
            .collect::<Vec<_>>();

        Ok(encodings)
    }
}
