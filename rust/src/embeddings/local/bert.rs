#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::path::PathBuf;

use crate::embeddings::normalize_l2;
use crate::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use anyhow::Error as E;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo};
use ndarray::prelude::*;
use ort::{CUDAExecutionProvider, GraphOptimizationLevel, Session};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

pub trait BertEmbed {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, anyhow::Error>;
}

#[derive(Debug)]
pub struct OrtBertEmbedder {
    pub tokenizer: Tokenizer,
    pub model: Session,
    // pub providers: Arc<Vec<ExecutionProviderDispatch>>,
}

impl OrtBertEmbedder {
    pub fn new(model_id: String, revision: Option<String>) -> Result<Self, E> {
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
            // let weights = api.get("onnx/model.onnx")?;
            let weights = PathBuf::from("model_optimized.onnx");

            (config, tokenizer, weights)
        };

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;

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
        let model = Session::builder()?
            .with_execution_providers([CUDAExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file(weights_filename)?;

        Ok(OrtBertEmbedder {
            tokenizer,
            model,
            // providers,
        })
    }

    fn tokenize_batch(&self, text_batch: &[String]) -> Result<Array2<i64>, E> {
        let tokens = self
            .tokenizer
            .encode_batch(text_batch.to_vec(), true)
            .map_err(E::msg)?;
        let token_ids = tokens
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

    fn reshape_into_2d_vector(&self, raw_data: Vec<f32>, dim: Option<usize>) -> Vec<Vec<f32>> {
        let dim = dim.expect("Dimension must be provided");
        let mut reshaped: Vec<Vec<f32>> = Vec::new();

        for chunk in raw_data.chunks(dim) {
            reshaped.push(chunk.to_vec());
        }

        reshaped
    }
}

pub struct BertEmbedder {
    pub model: BertModel,
    pub tokenizer: Tokenizer,
}

impl Default for BertEmbedder {
    fn default() -> Self {
        Self::new("sentence-transformers/all-MiniLM-L12-v2".to_string(), None).unwrap()
    }
}
impl BertEmbedder {
    pub fn new(model_id: String, revision: Option<String>) -> Result<Self, E> {
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
                VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device).unwrap()
            }
        } else {
            println!("Loading weights from pytorch_model.bin");
            VarBuilder::from_pth(&weights_filename, DTYPE, &device).unwrap()
        };

        config.hidden_act = HiddenAct::GeluApproximate;

        let model = BertModel::load(vb, &config).unwrap();
        let tokenizer = tokenizer;

        Ok(BertEmbedder { model, tokenizer })
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
    fn embed(&self, text_batch: &[String], batch_size: Option<usize>) -> Result<Vec<Vec<f32>>, E> {
        let batch_size = batch_size.unwrap_or(32);
        let mut encodings: Vec<Vec<f32>> = Vec::new();

        for mini_text_batch in text_batch.chunks(batch_size) {
            let token_ids: Array2<i64> = self.tokenize_batch(mini_text_batch).unwrap();
            let token_type_ids: Array2<i64> = Array2::zeros(token_ids.raw_dim());
            let attention_mask: Array2<i64> = Array2::ones(token_ids.raw_dim());
            let outputs = self
                .model
                .run(ort::inputs![token_ids, token_type_ids, attention_mask]?)
                .unwrap();
            let embeddings = outputs["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .unwrap();
            let shape = embeddings.shape();
            let (_, n_tokens, hidden_size) = (shape[0], shape[1], shape[2]);
            let embeddings = embeddings.sum_axis(Axis(1)) / n_tokens as f32;
            let denominator = embeddings
                .mapv(|x| x * x)
                .sum_axis(Axis(1))
                .sqrt()
                .insert_axis(Axis(1));
            let embeddings = embeddings.clone() / denominator;

            let norm = embeddings.mapv(|x| x * x).sum_axis(Axis(1)).sqrt();
            let batch_encodings: (Vec<f32>, Option<usize>) = embeddings.into_raw_vec_and_offset();

            let batch_encodings =
                self.reshape_into_2d_vector(batch_encodings.0, Some(hidden_size as usize));
            encodings.extend(batch_encodings);
        }
        Ok(encodings)
    }
}

impl BertEmbed for BertEmbedder {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        let batch_size = batch_size.unwrap_or(32);
        let mut encodings: Vec<Vec<f32>> = Vec::new();

        for mini_text_batch in text_batch.chunks(batch_size) {
            let token_ids = self
                .tokenize_batch(mini_text_batch, &self.model.device)
                .unwrap();
            let token_type_ids = token_ids.zeros_like().unwrap();
            let embeddings: Tensor = self
                .model
                .forward(&token_ids, &token_type_ids, None)
                .unwrap();
            let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3().unwrap();

            let embeddings = (embeddings.sum(1).unwrap() / (n_tokens as f64)).unwrap();
            let embeddings = normalize_l2(&embeddings).unwrap();
            let batch_encodings = embeddings.to_vec2::<f32>().unwrap();

            encodings.extend(batch_encodings);
        }

        Ok(encodings)
    }
}

#[cfg(test)]

mod tests {
    use crate::embeddings::embed::Embeder;
    use crate::embeddings::embed_audio;
    use crate::file_processor::audio::audio_processor::Segment;

    use super::*;
    use std::path::PathBuf;
    use std::time::Instant;

    #[test]
    fn test_bert_embeder() {
        let embeder = BertEmbedder::default();
        let text = vec![
            "Rust is a systems programming language".to_string(),
            "It is blazingly fast".to_string(),
        ];
        let start = Instant::now();
        let embeddings = embeder.embed(&text, None).unwrap();
        println!("Time taken: {:?}", start.elapsed());
        assert_eq!(embeddings.len(), 2);
    }

    #[tokio::test]
    async fn test_bert_embeder_audio() {
        let bert_model =
            Embeder::from_pretrained_hf("bert", "sentence-transformers/all-MiniLM-L6-v2", None)
                .unwrap();
        let segments = vec![
            Segment {
                start: 0.0,
                duration: 1.0,
                dr: Default::default(),
            },
            Segment {
                start: 1.0,
                duration: 1.0,
                dr: Default::default(),
            },
        ];
        let audio_file = PathBuf::from("tests/data/sample.wav");
        let start = Instant::now();
        let embeddings = embed_audio(&bert_model, segments, audio_file, None)
            .await
            .unwrap();
        println!("Time taken: {:?}", start.elapsed());
        assert_eq!(embeddings.len(), 2);
    }
}
