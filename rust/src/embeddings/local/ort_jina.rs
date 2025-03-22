use super::bert::TokenizerConfig;
use super::jina::JinaEmbed;
use super::pooling::{ModelOutput, PooledOutputType, Pooling};
use super::text_embedding::{models_map, ONNXModel};
use crate::embeddings::embed::EmbeddingResult;
use crate::embeddings::utils::tokenize_batch_ndarray;
use crate::Dtype;
use anyhow::Error as E;
use hf_hub::api::sync::Api;
use hf_hub::Repo;
use ndarray::prelude::*;
use rayon::prelude::*;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use {
    ort::execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider},
    ort::session::builder::GraphOptimizationLevel,
    ort::session::Session,
};
#[derive(Debug)]
pub struct OrtJinaEmbedder {
    pub session: Session,
    pub version: String,
    pub tokenizer: Tokenizer,
    pub pooling: Pooling,
}

impl OrtJinaEmbedder {
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
            let _ = api.get(format!("{path}_data").as_str());

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

        let version = match (model_name, model_id) {
            (Some(ONNXModel::JINAV3), _) => "v3",
            (_, Some(id)) if id.contains("jina-embeddings-v3") => "v3",
            _ => "v2",
        };

        Ok(OrtJinaEmbedder {
            session: model,
            version: version.to_string(),
            tokenizer,
            pooling,
        })
    }

    pub fn embed_late_chunking(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, E> {
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
            )
            .unwrap();
            let attention_mask_ndarray = Array2::from_shape_vec(
                (attention_mask_merged.len(), attention_mask_merged[0].len()),
                attention_mask_merged
                    .into_iter()
                    .flatten()
                    .map(|x| x as i64)
                    .collect::<Vec<i64>>(),
            )
            .unwrap();

            let token_type_ids: Array2<i64> = Array2::zeros(token_ids_ndarray.raw_dim());

            let embeddings = if self.version == "v3" {
                let outputs = self.session.run(ort::inputs! {
                    "input_ids" => token_ids_ndarray,
                    "attention_mask" => attention_mask_ndarray.clone(),
                    "task_id" => Array1::<i64>::from_vec(vec![4])
                }?)?;
                outputs["text_embeds"]
                    .try_extract_tensor::<f32>()?
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix3>()?
            } else {
                let outputs = self.session.run(ort::inputs! {
                    "input_ids" => token_ids_ndarray,
                    "token_type_ids" => token_type_ids,
                    "attention_mask" => attention_mask_ndarray.clone()
                }?)?;
                outputs["last_hidden_state"]
                    .try_extract_tensor::<f32>()?
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix3>()?
            };

            let (_, _, _) = embeddings.dim();
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

impl JinaEmbed for OrtJinaEmbedder {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
        late_chunking: Option<bool>,
    ) -> Result<Vec<EmbeddingResult>, E> {
        if late_chunking.unwrap_or(false) {
            self.embed_late_chunking(text_batch, batch_size)
        } else {
            let batch_size: usize = batch_size.unwrap_or(32);
            let output_name = self.session.outputs.first().unwrap().name.as_str();

            let encodings = text_batch
                .par_chunks(batch_size)
                .flat_map(|mini_text_batch| -> Result<Vec<Vec<f32>>, E> {
                    let (token_ids, attention_mask): (Array2<i64>, Array2<i64>) =
                        tokenize_batch_ndarray(&self.tokenizer, mini_text_batch)?;
                    let token_type_ids: Array2<i64> = Array2::zeros(token_ids.raw_dim());

                    let embeddings = if self.version == "v3" {
                        let outputs = self.session.run(ort::inputs! {
                            "input_ids" => token_ids,
                            "attention_mask" => attention_mask.clone(),
                            "task_id" => Array1::<i64>::from_vec(vec![4])
                        }?)?;
                        outputs[output_name]
                            .try_extract_tensor::<f32>()?
                            .to_owned()
                            .into_dimensionality::<ndarray::Ix3>()?
                    } else {
                        let outputs = self.session.run(ort::inputs! {
                            "input_ids" => token_ids,
                            "token_type_ids" => token_type_ids,
                            "attention_mask" => attention_mask.clone()
                        }?)?;
                        outputs[output_name]
                            .try_extract_tensor::<f32>()?
                            .to_owned()
                            .into_dimensionality::<ndarray::Ix3>()?
                    };

                    let (_, _, _) = embeddings.dim();
                    let attention_mask = attention_mask.mapv(|x| x as f32);
                    let attention_mask = PooledOutputType::from(attention_mask);
                    let attention_mask = Some(&attention_mask);
                    let model_output = ModelOutput::Array(embeddings.clone());
                    let pooled_output = match self.pooling {
                        Pooling::Cls => self.pooling.pool(&model_output, None)?,
                        Pooling::Mean => self.pooling.pool(&model_output, attention_mask)?,
                    };
                    let embeddings = pooled_output.to_array()?;

                    let norms = embeddings.mapv(|x| x * x).sum_axis(Axis(1)).mapv(f32::sqrt);
                    let embeddings = embeddings / &norms.insert_axis(Axis(1));

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
}
