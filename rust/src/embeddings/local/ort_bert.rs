use super::bert::{BertEmbed, TokenizerConfig};
use super::pooling::{ModelOutput, Pooling};
use super::text_embedding::ONNXModel;
use crate::embeddings::embed::EmbeddingResult;
use crate::embeddings::utils::{
    get_attention_mask_ndarray, get_type_ids_ndarray, tokenize_batch_ndarray,
};
use crate::embeddings::local::text_embedding::models_map;

use crate::Dtype;
use hf_hub::api::sync::Api;
use hf_hub::Repo;
use ndarray::prelude::*;
use ort::execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use rayon::prelude::*;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};
use anyhow::Error as E;

#[derive(Debug)]
pub struct OrtBertEmbedder {
    pub tokenizer: Tokenizer,
    pub model: Session,
    pub pooling: Pooling,
}

impl OrtBertEmbedder {
    pub fn new(
        model_name: Option<ONNXModel>,
        model_id: Option<&str>,
        revision: Option<&str>,
        dtype: Option<Dtype>,
        path_in_repo: Option<&str>,
    ) -> Result<Self, E> {
        let hf_model_id = match model_id {
            Some(id) => id,
            None => match model_name {
                Some(name) => models_map().get(&name).unwrap().model_code.as_str(),
                None => {
                    return Err(anyhow::anyhow!(
                        "Please provide either model_name or model_id"
                    ))
                }
            },
        };

        let pooling = match model_name {
            Some(name) => models_map()
                .get(&name)
                .unwrap()
                .model
                .get_default_pooling_method()
                .unwrap_or(Pooling::Mean),
            None => Pooling::Mean,
        };
        let path = match path_in_repo {
            Some(path) => path,
            None => match model_name {
                Some(name) => models_map().get(&name).unwrap().model_file.as_str(),
                None => "model.onnx",
            },
        };

        let (_, tokenizer_filename, weights_filename, tokenizer_config_filename) = {
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
            let base_path = path.rsplit_once('/').map(|(p, _)| p).unwrap_or("");
            let model_path = match dtype {
                Some(Dtype::Q4F16) => format!("{base_path}/model_q4f16.onnx"),
                Some(Dtype::F16) => format!("{base_path}/model_fp16.onnx"),
                Some(Dtype::INT8) => format!("{base_path}/model_int8.onnx"),
                Some(Dtype::Q4) => format!("{base_path}/model_q4.onnx"),
                Some(Dtype::UINT8) => format!("{base_path}/model_uint8.onnx"),
                Some(Dtype::BNB4) => format!("{base_path}/model_bnb4.onnx"),
                Some(Dtype::F32) => format!("{base_path}/model.onnx"),
                Some(Dtype::QUANTIZED) => format!("{base_path}/model_quantized.onnx"),
                None => path.to_string(),
            };
            let weights = api.get(model_path.as_str());
            (config, tokenizer, weights, tokenizer_config)
        };

        let weights_filename = match weights_filename {
            Ok(weights) => weights,
            Err(e) => {
                return Err(anyhow::anyhow!("ONNX weights not found for the model. Please check if the weights for the specified dtype exists. {}", e));
            }
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

        Ok(OrtBertEmbedder {
            tokenizer,
            model,
            pooling,
        })
    }
}

impl BertEmbed for OrtBertEmbedder {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, E> {
        let batch_size = batch_size.unwrap_or(32);
        let encodings = text_batch
            .par_chunks(batch_size)
            .flat_map(|mini_text_batch| -> Result<Vec<Vec<f32>>, E> {
                let input_ids: Array2<i64> =
                    tokenize_batch_ndarray(&self.tokenizer, mini_text_batch)?;
                let token_type_ids: Array2<i64> = Array2::zeros(input_ids.raw_dim());
                let attention_mask: Array2<i64> = Array2::ones(input_ids.raw_dim());

                let input_names = self
                    .model
                    .inputs
                    .iter()
                    .map(|input| input.name.as_str())
                    .collect::<Vec<_>>();

                let mut inputs =
                    ort::inputs!["input_ids" => input_ids, "attention_mask" => attention_mask]?;
                if input_names.iter().any(|&x| x == "token_type_ids") {
                    inputs.push((
                        "token_type_ids".into(),
                        Value::from_array(token_type_ids.clone())?.into(),
                    ));
                }
                let outputs = self.model.run(inputs)?;
                let embeddings: Array3<f32> = outputs
                    [self.model.outputs.first().unwrap().name.as_str()]
                .try_extract_tensor::<f32>()?
                .to_owned()
                .into_dimensionality::<ndarray::Ix3>()?;
                let (_, _, _) = embeddings.dim();
                let embeddings = self
                    .pooling
                    .pool(&ModelOutput::Array(embeddings))?
                    .to_array()?;
                let norms = embeddings.mapv(|x| x * x).sum_axis(Axis(1)).mapv(f32::sqrt);
                let embeddings = &embeddings / &norms.insert_axis(Axis(1));

                Ok(embeddings.outer_iter().map(|row| row.to_vec()).collect())
            })
            .flatten()
            .collect::<Vec<_>>();

        Ok(encodings
            .iter()
            .map(|x| EmbeddingResult::DenseVector(x.to_vec()))
            .collect())
    }
}

pub struct OrtSparseBertEmbedder {
    pub tokenizer: Tokenizer,
    pub model: Session,
}

impl OrtSparseBertEmbedder {
    pub fn new(
        model_name: Option<ONNXModel>,
        model_id: Option<&str>,
        revision: Option<&str>,
        path_in_repo: Option<&str>,
    ) -> Result<Self, E> {
        let hf_model_id = match model_id {
            Some(id) => id,
            None => match model_name {
                Some(name) => models_map().get(&name).unwrap().model_code.as_str(),
                None => {
                    return Err(anyhow::anyhow!(
                        "Please provide either model_name or model_id"
                    ))
                }
            },
        };

        let path = match path_in_repo {
            Some(path) => path,
            None => match model_name {
                Some(name) => models_map().get(&name).unwrap().model_file.as_str(),
                None => "model.onnx",
            },
        };

        let (_, tokenizer_filename, weights_filename, tokenizer_config_filename) = {
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
            let weights = api.get(path)?;
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

        Ok(OrtSparseBertEmbedder { tokenizer, model })
    }
}

impl BertEmbed for OrtSparseBertEmbedder {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let batch_size = batch_size.unwrap_or(32);
        let encodings = text_batch.par_chunks(batch_size).flat_map(|mini_text_batch| -> Result<Vec<Vec<f32>>, E> {
            let token_ids: Array2<i64> = tokenize_batch_ndarray(&self.tokenizer, mini_text_batch)?;
            let token_type_ids: Array2<i64> = get_type_ids_ndarray(&self.tokenizer, mini_text_batch)?;
            let attention_mask = get_attention_mask_ndarray(&self.tokenizer, mini_text_batch)?;
            let outputs = self.model.run(ort::inputs!["input_ids" => token_ids, "input_mask" => attention_mask.clone(), "segment_ids" => token_type_ids]?).unwrap();
            let embeddings: Array3<f32> = outputs["output"]
                .try_extract_tensor::<f32>()?
                .to_owned()
                .into_dimensionality::<ndarray::Ix3>()?;
            let relu_log: ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 3]>> = embeddings.mapv(|x| (1.0 + x.max(0.0)).ln());
            let weighted_log = relu_log * attention_mask.clone().mapv(|x| x as f32).insert_axis(Axis(2));
            let scores = weighted_log.fold_axis(Axis(1), f32::NEG_INFINITY, |r, &v| r.max(v));
            let norms = scores.mapv(|x| x * x).sum_axis(Axis(1)).mapv(f32::sqrt);
            let embeddings = &scores / &norms.insert_axis(Axis(1));
            Ok(embeddings.outer_iter().map(|row| row.to_vec()).collect())
        }).flatten().collect::<Vec<_>>();

        Ok(encodings
            .iter()
            .map(|x| EmbeddingResult::DenseVector(x.to_vec()))
            .collect())
    }
}
