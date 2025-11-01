use std::sync::RwLock;

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use hf_hub::{api::sync::Api, Repo};
use ndarray::Array2;
use ort::{
    execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::Dtype;
use crate::{embeddings::local::bert::TokenizerConfig, reranker::qwen3};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct RerankerResult {
    pub query: String,
    pub documents: Vec<DocumentRank>,
}

#[derive(Debug, Serialize, Clone)]
pub struct DocumentRank {
    pub document: String,
    pub relevance_score: f32,
    pub rank: usize,
}

pub struct Reranker {
    model: RwLock<Session>,
    model_type: Option<String>,
    tokenizer: Tokenizer,
}

impl Reranker {
    pub fn new(
        model_id: &str,
        revision: Option<&str>,
        dtype: Dtype,
        path_in_repo: Option<&str>,
    ) -> Result<Self, E> {
        let (config_filename, tokenizer_filename, weights_filename, tokenizer_config_filename) = {
            let api = Api::new().unwrap();
            let api = match revision {
                Some(rev) => api.repo(Repo::with_revision(
                    model_id.to_string(),
                    hf_hub::RepoType::Model,
                    rev.to_string(),
                )),
                None => api.repo(hf_hub::Repo::new(
                    model_id.to_string(),
                    hf_hub::RepoType::Model,
                )),
            };
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let tokenizer_config = api.get("tokenizer_config.json")?;

            let mut path_in_repo = path_in_repo.unwrap_or_default().to_string();
            if !path_in_repo.is_empty() {
                path_in_repo.push('/');
            }
            let weights = match dtype {
                Dtype::Q4F16 => api.get(format!("{}model_q4f16.onnx", path_in_repo).as_str())?,
                Dtype::F16 => api.get(format!("{}model_fp16.onnx", path_in_repo).as_str())?,
                Dtype::INT8 => api.get(format!("{}model_int8.onnx", path_in_repo).as_str())?,
                Dtype::Q4 => api.get(format!("{}model_q4.onnx", path_in_repo).as_str())?,
                Dtype::UINT8 => api.get(format!("{}model_uint8.onnx", path_in_repo).as_str())?,
                Dtype::BNB4 => api.get(format!("{}model_bnb4.onnx", path_in_repo).as_str())?,
                Dtype::F32 => api.get(format!("{}model.onnx", path_in_repo).as_str())?,
                Dtype::BF16 => api.get(format!("{}model_bf16.onnx", path_in_repo).as_str())?,
                Dtype::QUANTIZED => {
                    api.get(format!("{}model_quantized.onnx", path_in_repo).as_str())?
                }
            };
            (config, tokenizer, weights, tokenizer_config)
        };
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

        let config_file = std::fs::read_to_string(config_filename)?;
        let config: serde_json::Value = serde_json::from_str(&config_file)?;
        let model_type = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

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

        // Get physical core count (excluding hyperthreading)
        let threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        // For CPU-bound workloads like ONNX inference, it's often better to use
        // physical cores rather than logical cores to avoid context switching overhead
        let optimal_threads = std::cmp::max(1, threads / 2);

        let model = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
                CoreMLExecutionProvider::default().build(),
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(optimal_threads)? // Use optimal thread count
            .with_inter_threads(1)? // Set inter-op parallelism to 1 when using GPU
            .commit_from_file(weights_filename)?;

        Ok(Reranker {
            model: RwLock::new(model),
            model_type,
            tokenizer,
        })
    }

    pub fn compute_scores(
        &self,
        queries: Vec<&str>,
        documents: Vec<&str>,
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, E> {
        // Check model type once at the beginning
        let is_qwen3 = self.model_type.as_ref().is_some_and(|t| t == "qwen3");

        let prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
        let suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

        let pairs = queries
            .iter()
            .flat_map(|query| documents.iter().map(move |doc| (*query, *doc)))
            .collect::<Vec<_>>();

        let pairs = if is_qwen3 {
            pairs
                .iter()
                .map(|(query, doc)| (format!("{}{}", prefix, query), format!("{}{}", doc, suffix)))
                .collect::<Vec<_>>()
        } else {
            pairs
                .iter()
                .map(|(query, doc)| (query.to_string(), doc.to_string()))
                .collect::<Vec<_>>()
        };

        let mut scores = Vec::with_capacity(pairs.len());
        let mut model_guard = self.model.write().unwrap();

        let false_token_id = self
            .tokenizer
            .token_to_id("no")
            .ok_or(E::msg("no token found"))?;
        let true_token_id = self
            .tokenizer
            .token_to_id("yes")
            .ok_or(E::msg("yes token found"))?;

        for pair in pairs.chunks(batch_size) {
            let input_ids = self.tokenize_batch_ndarray(pair)?;
            let attention_mask = self.get_attention_mask_ndarray(pair)?;
            let input_ids_tensor = ort::value::TensorRef::from_array_view(&input_ids)?;
            let attention_mask_tensor = ort::value::TensorRef::from_array_view(&attention_mask)?;
            let model_inputs = model_guard
                .inputs
                .iter()
                .map(|i| i.name.clone())
                .collect::<Vec<_>>();

            let seq_len = input_ids.shape()[1];
            let position_ids_vec: Vec<i64> = (0..seq_len as i64)
                .collect::<Vec<i64>>()
                .repeat(input_ids.shape()[0]);
            let position_ids =
                Array2::<i64>::from_shape_vec((input_ids.shape()[0], seq_len), position_ids_vec)?;
            let position_ids_tensor = ort::value::TensorRef::from_array_view(&position_ids)?;

            let inputs = if model_inputs.contains(&"position_ids".to_string()) {
                ort::inputs!["input_ids" => input_ids_tensor, "attention_mask" => attention_mask_tensor, "position_ids" => position_ids_tensor]
            } else {
                ort::inputs!["input_ids" => input_ids_tensor, "attention_mask" => attention_mask_tensor]
            };

            if is_qwen3 {
                let logits = qwen3::compute_scores(
                    &mut model_guard,
                    input_ids,
                    attention_mask,
                    position_ids,
                    false_token_id,
                    true_token_id,
                )?;
                scores.extend(logits);
            } else {
                let outputs = model_guard.run(inputs)?;
                let logits = outputs["logits".to_string()]
                    .try_extract_array::<f32>()?
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix2>()?;
                scores.extend(
                    logits
                        .outer_iter()
                        .flat_map(|row| row.to_vec())
                        .collect::<Vec<_>>(),
                );
            }
        }

        let scores_tensor = Tensor::from_vec(
            scores.clone(),
            (queries.len(), documents.len()),
            &Device::Cpu,
        )?;

        if is_qwen3 {
            Ok(scores_tensor.to_vec2::<f32>()?)
        } else {
            let sigmoid_scores = candle_nn::ops::sigmoid(&scores_tensor)?;
            Ok(sigmoid_scores.to_vec2::<f32>()?)
        }
    }

    pub fn rerank(
        &self,
        queries: Vec<&str>,
        documents: Vec<&str>,
        batch_size: usize,
    ) -> Result<Vec<RerankerResult>, E> {
        let scores = self.compute_scores(queries.clone(), documents.clone(), batch_size)?;
        let mut reranker_results = Vec::new();
        for (i, query) in queries.iter().enumerate() {
            let scores = scores[i].clone();
            let mut indices: Vec<usize> = (0..scores.len()).collect();
            indices.sort_by(|&j, &k| {
                scores[k]
                    .partial_cmp(&scores[j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let document_ranks = scores
                .iter()
                .enumerate()
                .map(|(p, score)| DocumentRank {
                    document: documents[p].to_string(),
                    relevance_score: *score,
                    rank: indices.iter().position(|&i| i == p).unwrap() + 1,
                })
                .collect::<Vec<_>>();

            reranker_results.push(RerankerResult {
                query: query.to_string(),
                documents: document_ranks,
            });
        }
        Ok(reranker_results)
    }

    pub fn tokenize_batch_ndarray(
        &self,
        pairs: &[(String, String)],
    ) -> anyhow::Result<Array2<i64>> {
        let token_ids = self
            .tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(E::msg)?
            .iter()
            .map(|tokens| {
                tokens
                    .get_ids()
                    .iter()
                    .map(|&id| id as i64)
                    .collect::<Vec<i64>>()
            })
            .collect::<Vec<Vec<i64>>>();

        let token_ids_array = Array2::from_shape_vec(
            (token_ids.len(), token_ids[0].len()),
            token_ids.into_iter().flatten().collect::<Vec<i64>>(),
        )
        .unwrap();
        Ok(token_ids_array)
    }

    pub fn get_attention_mask_ndarray(
        &self,
        pairs: &[(String, String)],
    ) -> anyhow::Result<Array2<i64>> {
        let attention_mask = self
            .tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(E::msg)?
            .iter()
            .map(|tokens| {
                tokens
                    .get_attention_mask()
                    .iter()
                    .map(|&id| id as i64)
                    .collect::<Vec<i64>>()
            })
            .collect::<Vec<Vec<i64>>>();

        let attention_mask_array = Array2::from_shape_vec(
            (attention_mask.len(), attention_mask[0].len()),
            attention_mask.into_iter().flatten().collect::<Vec<i64>>(),
        )
        .unwrap();
        Ok(attention_mask_array)
    }
}
