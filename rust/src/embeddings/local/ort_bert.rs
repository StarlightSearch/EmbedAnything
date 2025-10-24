use std::sync::RwLock;

use super::bert::{BertEmbed, TokenizerConfig};
use super::pooling::{ModelOutput, PooledOutputType, Pooling};
use super::text_embedding::ONNXModel;
use crate::embeddings::embed::EmbeddingResult;
use crate::embeddings::local::text_embedding::models_map;
use crate::embeddings::utils::{get_type_ids_ndarray, tokenize_batch_ndarray};

use crate::Dtype;
use anyhow::Error as E;
use hf_hub::api::sync::Api;
use hf_hub::Repo;
use ndarray::prelude::*;
use ort::execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

#[derive(Debug)]
pub struct OrtBertEmbedder {
    pub tokenizer: Tokenizer,
    pub model: RwLock<Session>,
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
            let mut base_path = path
                .rsplit_once('/')
                .map(|(p, _)| p.to_string())
                .unwrap_or_default();
            if !base_path.is_empty() {
                base_path.push('/');
            }
            let model_path = match dtype {
                Some(Dtype::Q4F16) => format!("{base_path}model_q4f16.onnx"),
                Some(Dtype::F16) => format!("{base_path}model_fp16.onnx"),
                Some(Dtype::INT8) => format!("{base_path}model_int8.onnx"),
                Some(Dtype::Q4) => format!("{base_path}model_q4.onnx"),
                Some(Dtype::UINT8) => format!("{base_path}model_uint8.onnx"),
                Some(Dtype::BNB4) => format!("{base_path}model_bnb4.onnx"),
                Some(Dtype::F32) => format!("{base_path}model.onnx"),
                Some(Dtype::QUANTIZED) => format!("{base_path}model_quantized.onnx"),
                Some(Dtype::BF16) => format!("{base_path}model_bf16.onnx"),
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

        Ok(OrtBertEmbedder {
            tokenizer,
            model: RwLock::new(model),
            pooling,
        })
    }

    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, E> {
        let batch_size = batch_size.unwrap_or(32);

        // Pre-compute input names once
        let mut model_guard = self.model.write().unwrap();
        let input_names: Vec<_> = model_guard
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect();
        let output_name = model_guard.outputs.first().unwrap().name.to_string();
        let needs_token_type = input_names.contains(&"token_type_ids");

        // Run model and extract embeddings
        let encodings = text_batch
            .chunks(batch_size)
            .flat_map(|mini_text_batch| -> Result<Vec<Vec<f32>>, E> {
                // Tokenize and prepare inputs
                let (input_ids, attention_mask) =
                    tokenize_batch_ndarray(&self.tokenizer, mini_text_batch)?;

                // Build inputs more efficiently
                let input_ids_tensor = ort::value::TensorRef::from_array_view(&input_ids)?;
                let attention_mask_tensor =
                    ort::value::TensorRef::from_array_view(&attention_mask)?;
                let token_type_ids = Array2::<i64>::zeros(input_ids.raw_dim());

                let token_type_id_tensor = ort::value::TensorRef::from_array_view(&token_type_ids)?;
                let inputs = if needs_token_type {
                    ort::inputs![
                        "input_ids" => input_ids_tensor,
                        "attention_mask" => attention_mask_tensor,
                        "token_type_ids" => token_type_id_tensor
                    ]
                } else {
                    ort::inputs![
                        "input_ids" => input_ids_tensor,
                        "attention_mask" => attention_mask_tensor
                    ]
                };

                let embeddings = model_guard.run(inputs)?;
                let embeddings: ndarray::ArrayViewD<f32> =
                    embeddings[output_name.as_str()].try_extract_array()?;

                // Prepare attention mask for pooling
                let attention_mask = if matches!(self.pooling, Pooling::Mean) {
                    Some(PooledOutputType::from(attention_mask.mapv(|x| x as f32)))
                } else {
                    None
                };

                // Pool and normalize embeddings
                let embeddings = embeddings.into_dimensionality::<Ix3>()?.to_owned();
                let model_output = ModelOutput::Array(embeddings);
                let pooled = self.pooling.pool(&model_output, attention_mask.as_ref())?;
                let embeddings = pooled.to_array()?;

                // Normalize in one step
                let norms = embeddings.mapv(|x| x * x).sum_axis(Axis(1)).mapv(f32::sqrt);
                let normalized = embeddings / &norms.insert_axis(Axis(1));

                Ok(normalized.outer_iter().map(|row| row.to_vec()).collect())
            })
            .flatten()
            .collect::<Vec<_>>();

        Ok(encodings
            .into_iter() // Use into_iter since we don't need the original vector
            .map(EmbeddingResult::DenseVector)
            .collect())
    }

    pub fn embed_late_chunking(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, E> {
        let batch_size = batch_size.unwrap_or(32);
        let mut results = Vec::new();

        // Pre-compute input names once
        let mut model_guard = self.model.write().unwrap();
        let input_names: Vec<_> = model_guard
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect();
        let needs_token_type = input_names.contains(&"token_type_ids");
        let output_name = model_guard.outputs.first().unwrap().name.to_string();

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
            let token_ids_merged = vec![token_ids.concat()];
            let attention_mask_merged = vec![attention_mask.concat()];

            // Convert to ndarray
            let token_ids_ndarray = Array2::from_shape_vec(
                (token_ids_merged.len(), token_ids_merged[0].len()),
                token_ids_merged
                    .into_iter()
                    .flatten()
                    .map(|x| x as i64)
                    .collect::<Vec<i64>>(),
            )?;

            let attention_mask_ndarray = Array2::from_shape_vec(
                (attention_mask_merged.len(), attention_mask_merged[0].len()),
                attention_mask_merged
                    .into_iter()
                    .flatten()
                    .map(|x| x as i64)
                    .collect::<Vec<i64>>(),
            )?;

            let token_type_ids: Array2<i64> = Array2::zeros(token_ids_ndarray.raw_dim());
            let input_ids_tensor = ort::value::TensorRef::from_array_view(&token_ids_ndarray)?;
            let attention_mask_tensor =
                ort::value::TensorRef::from_array_view(&attention_mask_ndarray)?;
            let inputs = if needs_token_type {
                let token_type_tensor = ort::value::TensorRef::from_array_view(&token_type_ids)?;
                ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                    "token_type_ids" => token_type_tensor
                ]
            } else {
                ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor
                ]
            };
            let embeddings = model_guard.run(inputs)?;
            let embeddings: ndarray::ArrayViewD<f32> =
                embeddings[output_name.as_str()].try_extract_array()?;

            let attention_mask = attention_mask_ndarray.mapv(|x| x as f32);

            for (i, &end_idx) in cumulative_seq_lengths.iter().enumerate() {
                let start_idx = if i == 0 {
                    0
                } else {
                    cumulative_seq_lengths[i - 1]
                };

                let embedding_slice = embeddings.slice(s![.., start_idx..end_idx, ..]);
                let attention_mask_slice = attention_mask.slice(s![.., start_idx..end_idx]);
                let model_output = ModelOutput::Array(embedding_slice.to_owned());
                let attention_mask = PooledOutputType::from(attention_mask_slice.to_owned());
                let attention_mask = Some(&attention_mask);

                let pooled_output = match self.pooling {
                    Pooling::Cls => self.pooling.pool(&model_output, None)?,
                    Pooling::Mean => self.pooling.pool(&model_output, attention_mask)?,
                    Pooling::LastToken => self.pooling.pool(&model_output, attention_mask)?,
                };
                let embedding = pooled_output.to_array()?;
                let norms = embedding.mapv(|x| x * x).sum_axis(Axis(1)).mapv(f32::sqrt);
                let embedding = embedding / &norms.insert_axis(Axis(1));
                results.push(EmbeddingResult::DenseVector(embedding.row(0).to_vec()));
            }
        }
        Ok(results)
    }
}

impl BertEmbed for OrtBertEmbedder {
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
pub struct OrtSparseBertEmbedder {
    pub tokenizer: Tokenizer,
    pub model: RwLock<Session>,
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
            (None, None) => 256,
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

        Ok(OrtSparseBertEmbedder {
            tokenizer,
            model: RwLock::new(model),
        })
    }
}

impl BertEmbed for OrtSparseBertEmbedder {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
        _late_chunking: Option<bool>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let batch_size = batch_size.unwrap_or(32);
        let mut model_guard = self.model.write().unwrap();

        let encodings = text_batch
            .chunks(batch_size)
            .flat_map(|mini_text_batch| -> Result<Vec<Vec<f32>>, E> {
                let (token_ids, attention_mask): (Array2<i64>, Array2<i64>) =
                    tokenize_batch_ndarray(&self.tokenizer, mini_text_batch)?;
                let token_type_ids: Array2<i64> =
                    get_type_ids_ndarray(&self.tokenizer, mini_text_batch)?;
                let token_ids_tensor = ort::value::TensorRef::from_array_view(&token_ids)?;
                let attention_mask_tensor =
                    ort::value::TensorRef::from_array_view(&attention_mask)?;
                let token_type_tensor = ort::value::TensorRef::from_array_view(&token_type_ids)?;
                let outputs = model_guard.run(ort::inputs![
                    "input_ids" => token_ids_tensor,
                    "input_mask" => attention_mask_tensor,
                    "segment_ids" => token_type_tensor
                ])?;
                let (shape, data) = outputs["output"].try_extract_tensor::<f32>()?;
                let embeddings = Array3::from_shape_vec(
                    (shape[0] as usize, shape[1] as usize, shape[2] as usize),
                    data.to_vec(),
                )?;
                let relu_log: ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 3]>> =
                    embeddings.mapv(|x| (1.0 + x.max(0.0)).ln());
                let weighted_log = relu_log
                    * attention_mask
                        .clone()
                        .mapv(|x| x as f32)
                        .insert_axis(Axis(2));
                let scores = weighted_log.fold_axis(Axis(1), f32::NEG_INFINITY, |r, &v| r.max(v));
                let norms = scores.mapv(|x| x * x).sum_axis(Axis(1)).mapv(f32::sqrt);
                let embeddings = &scores / &norms.insert_axis(Axis(1));
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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_ort_bert_embed() {
        let embedder = OrtBertEmbedder::new(
            None,
            Some("sentence-transformers/all-MiniLM-L6-v2"),
            None,
            None,
            Some("onnx/model.onnx"),
        )
        .unwrap();
        let embeddings = embedder
            .embed(&["Hello, world!", "I am a rust programmer"], Some(32))
            .unwrap();
        println!("embeddings: {:?}", embeddings);

        let test_embeddings: Vec<f32> = vec![
            -3.817_717_4e-2,
            3.291_110_3e-2,
            -5.459_385e-3,
            1.436_991_4e-2,
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
