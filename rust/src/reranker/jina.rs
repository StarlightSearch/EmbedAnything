use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use hf_hub::{api::sync::Api, Repo};
use ndarray::Array2;
use ort::{
    execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::embeddings::local::bert::TokenizerConfig;
use serde::Serialize;
pub enum Dtype {
    FP16,
    INT8,
    Q4,
    UINT8,
    BNB4,
}

#[derive(Debug, Serialize)]
pub struct RerankerResult {
    pub query: String,
    pub documents: Vec<DocumentRank>,
}

#[derive(Debug, Serialize, Clone)]
pub struct DocumentRank{
    pub document: String,
    pub relevance_score: f32,
    pub rank: usize,
}

pub struct JinaReranker {
    model: Session,
    tokenizer: Tokenizer,
}

impl JinaReranker {
    pub fn new(model_id: &str, revision: Option<&str>, dtype: Dtype) -> Result<Self, E> {
        let (_, tokenizer_filename, weights_filename, tokenizer_config_filename) = {
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
            let weights = match dtype {
                Dtype::FP16 => api.get("onnx/model_fp16.onnx")?,
                Dtype::INT8 => api.get("onnx/model_int8.onnx")?,
                Dtype::Q4 => api.get("onnx/model_q4.onnx")?,
                Dtype::UINT8 => api.get("onnx/model_uint8.onnx")?,
                Dtype::BNB4 => api.get("onnx/model_bnb4.onnx")?,
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

        let threads = std::thread::available_parallelism().unwrap().get();
        let model = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
                CoreMLExecutionProvider::default().build(),
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(weights_filename)?;

        Ok(JinaReranker {
            model,
            tokenizer,
        })
    }

    pub fn compute_scores(
        &self,
        queries: Vec<&str>,
        documents: Vec<&str>,
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, E> {
        let pairs = queries.iter()
            .flat_map(|query| documents.iter().map(move |doc| (*query, *doc)))
            .collect::<Vec<_>>();
        let mut scores= Vec::with_capacity(pairs.len());
        for pair in pairs.chunks(batch_size) {
            let input_ids = self.tokenize_batch_ndarray(pair)?;
            let attention_mask = self.get_attention_mask_ndarray(pair)?;
            let outputs = self
                .model
                .run(ort::inputs!["input_ids" => input_ids, "attention_mask" => attention_mask]?)?;
            let logits = outputs["logits"]
                .try_extract_tensor::<f32>()?
                .to_owned()
                .into_dimensionality::<ndarray::Ix2>()?;
            scores.extend(logits.outer_iter().map(|row| row.to_vec()).flatten().collect::<Vec<_>>());
        }
        let scores_tensor = Tensor::from_vec(scores.clone(), (queries.len(), documents.len()), &Device::Cpu)?;
        let sigmoid_scores = candle_nn::ops::sigmoid(&scores_tensor).unwrap();
        Ok(sigmoid_scores.to_vec2::<f32>()?)
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
            indices.sort_by(|&j, &k| scores[k].partial_cmp(&scores[j]).unwrap_or(std::cmp::Ordering::Equal));
            let document_ranks = scores.iter().enumerate().map(|(p, score)| DocumentRank {
                document: documents[p].to_string(),
                relevance_score: score.clone(),
                rank: indices.iter().position(|&i| i == p).unwrap()+1,
            }).collect::<Vec<_>>();
            
            reranker_results.push(RerankerResult {
                query: query.to_string(),
                documents: document_ranks,
            });
        }
        Ok(reranker_results)
    }

    pub fn tokenize_batch_ndarray(
        &self,
        pairs: &[(&str, &str)],
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
        pairs: &[(&str, &str)],
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
