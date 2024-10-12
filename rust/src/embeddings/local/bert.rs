#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use crate::embeddings::embed::EmbeddingResult;
use crate::embeddings::local::text_embedding::{get_model_info_by_hf_id, models_map};
use crate::embeddings::normalize_l2;
use crate::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use anyhow::Error as E;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo};
use ndarray::prelude::*;
use ort::{CUDAExecutionProvider, ExecutionProvider, GraphOptimizationLevel, Session};
use rayon::prelude::*;
use serde::Deserialize;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use super::pooling::{ModelOutput, Pooling};
use super::text_embedding::ONNXModel;

pub trait BertEmbed {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error>;
}
#[derive(Debug, Deserialize)]
pub struct TokenizerConfig {
    pub max_length: Option<usize>,
    pub model_max_length: Option<usize>,
}

#[derive(Debug)]
pub struct OrtBertEmbedder {
    pub tokenizer: Tokenizer,
    pub model: Session,
    pub pooling: Pooling,
}

impl OrtBertEmbedder {
    pub fn new(model: ONNXModel, revision: Option<String>) -> Result<Self, E> {
        let model_info = models_map().get(&model).ok_or_else(|| {
            E::msg("ONNX model does not exist for the specified model")
        })?;
        let pooling = model_info
            .model
            .get_default_pooling_method()
            .unwrap_or(Pooling::Mean);

        let (_, tokenizer_filename, weights_filename, tokenizer_config_filename) = {
            let api = Api::new().unwrap();
            let api = match revision {
                Some(rev) => api.repo(Repo::with_revision(
                    model_info.model_code.to_string(),
                    hf_hub::RepoType::Model,
                    rev,
                )),
                None => api.repo(hf_hub::Repo::new(
                    model_info.model_code.to_string(),
                    hf_hub::RepoType::Model,
                )),
            };
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let tokenizer_config = api.get("tokenizer_config.json")?;
            let weights = api.get(model_info.model_file.as_str())?;
            (config, tokenizer, weights, tokenizer_config)
        };
        let tokenizer_config = std::fs::read_to_string(tokenizer_config_filename)?;
        let tokenizer_config: TokenizerConfig = serde_json::from_str(&tokenizer_config)?;

        // Set max_length to the minimum of max_length and model_max_length if both are present
        let max_length = match (tokenizer_config.max_length, tokenizer_config.model_max_length) {
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
            .with_execution_providers([CUDAExecutionProvider::default()
                .build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(weights_filename)?;

        Ok(OrtBertEmbedder {
            tokenizer,
            model,
            pooling,
        })
    }

    fn tokenize_batch(&self, text_batch: &[String]) -> Result<Array2<i64>, E> {
        let token_ids = self
            .tokenizer
            .encode_batch(text_batch.to_vec(), true)
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

    // fn reshape_into_2d_vector(&self, raw_data: Vec<f32>, dim: Option<usize>) -> Vec<Vec<f32>> {
    //     let dim = dim.expect("Dimension must be provided");
    //     raw_data.chunks(dim).map(|chunk| chunk.to_vec()).collect()
    // }
}

pub struct BertEmbedder {
    pub model: BertModel,
    pub pooling: Pooling,
    pub tokenizer: Tokenizer,
}

impl Default for BertEmbedder {
    fn default() -> Self {
        Self::new("sentence-transformers/all-MiniLM-L12-v2".to_string(), None).unwrap()
    }
}
impl BertEmbedder {
    pub fn new(model_id: String, revision: Option<String>) -> Result<Self, E> {
        let model_info = get_model_info_by_hf_id(&model_id);
        let pooling = match model_info {
            Some(info) => info.model.get_default_pooling_method().unwrap_or(Pooling::Mean),
            None => Pooling::Mean,
        };

        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new().unwrap();
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
        let mut config: Config = serde_json::from_str(&config)?;
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
            .unwrap();

        println!("Loading weights from {:?}", weights_filename);

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let vb = if weights_filename.ends_with("model.safetensors") {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)?
            }
        } else {
            println!("Loading weights from pytorch_model.bin");
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        };

        config.hidden_act = HiddenAct::GeluApproximate;

        let model = BertModel::load(vb, &config)?;
        let tokenizer = tokenizer;

        Ok(BertEmbedder {
            model,
            tokenizer,
            pooling,
        })
    }
    fn tokenize_batch(&self, text_batch: &[String], device: &Device) -> anyhow::Result<Tensor> {
        let tokens = self
            .tokenizer
            .encode_batch(text_batch.to_vec(), true)
            .map_err(E::msg)?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Tensor::new(tokens.as_slice(), device)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        Ok(Tensor::stack(&token_ids, 0)?)
    }
}

impl BertEmbed for OrtBertEmbedder {
    fn embed(&self, text_batch: &[String], batch_size: Option<usize>) -> Result<Vec<EmbeddingResult>, E> {
        let batch_size = batch_size.unwrap_or(32);
        let encodings = text_batch
            .par_chunks(batch_size)
            .flat_map(|mini_text_batch| -> Result<Vec<Vec<f32>>, E> {
                let token_ids: Array2<i64> = self.tokenize_batch(mini_text_batch)?;
                let token_type_ids: Array2<i64> = Array2::zeros(token_ids.raw_dim());
                let attention_mask: Array2<i64> = Array2::ones(token_ids.raw_dim());
                let outputs =
                    self.model
                        .run(ort::inputs![token_ids, token_type_ids, attention_mask]?)?;
                let embeddings: Array3<f32> = outputs["last_hidden_state"]
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

                Ok(embeddings
                    .outer_iter()
                    .map(|row| row.to_vec())
                    .collect())
            })
            .flatten()
            .collect::<Vec<_>>();

        Ok(encodings
            .iter()
            .map(|x| EmbeddingResult::Dense(x.to_vec()))
            .collect())
    }
}

impl BertEmbed for BertEmbedder {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let batch_size = batch_size.unwrap_or(32);
        let mut encodings: Vec<EmbeddingResult> = Vec::new();

        for mini_text_batch in text_batch.chunks(batch_size) {
            let token_ids = self
                .tokenize_batch(mini_text_batch, &self.model.device)
                .unwrap();
            let token_type_ids = token_ids.zeros_like().unwrap();
            let embeddings: Tensor = self
                .model
                .forward(&token_ids, &token_type_ids, None)
                .unwrap();
            let pooled_output = self
                .pooling
                .pool(&ModelOutput::Tensor(embeddings.clone()))?
                .to_tensor()?;

            let embeddings = normalize_l2(&pooled_output).unwrap();
            let batch_encodings = embeddings.to_vec2::<f32>().unwrap();

            encodings.extend(batch_encodings.iter().map(|x| EmbeddingResult::Dense(x.to_vec())));
        }
        Ok(encodings)
    }
}

