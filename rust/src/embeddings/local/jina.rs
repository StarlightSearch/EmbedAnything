#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use crate::embeddings::select_device;
use crate::embeddings::{embed::EmbeddingResult, normalize_l2};
use crate::models::jina_bert::{BertModel, Config};
use anyhow::Error as E;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use hf_hub::api::sync::Api;
use hf_hub::Repo;
use ndarray::prelude::*;
use ort::execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use super::bert::TokenizerConfig;
use super::pooling::{ModelOutput, Pooling};
use super::text_embedding::{models_map, ONNXModel};
use rayon::prelude::*;

pub trait JinaEmbed {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error>;
}

#[derive(Debug)]
pub struct OrtJinaEmbedder {
    pub session: Session,
    pub model: ONNXModel,
    pub tokenizer: Tokenizer,
    pub pooling: Pooling,
}

impl OrtJinaEmbedder {
    pub fn new(model: ONNXModel, revision: Option<&str>) -> Result<Self, E> {
        let model_info = models_map()
            .get(&model)
            .ok_or_else(|| E::msg("ONNX model does not exist for the specified model"))?;
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
                    rev.to_string(),
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

            if model == ONNXModel::JINAV3 {
                let weights = api.get(&("onnx/model_fp16.onnx"))?;
                (config, tokenizer, weights, tokenizer_config)
            } else {
                (config, tokenizer, weights, tokenizer_config)
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
        let session = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
                CoreMLExecutionProvider::default().build(),
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(weights_filename)?;

        Ok(OrtJinaEmbedder {
            session,
            model,
            tokenizer,
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
}

impl JinaEmbed for OrtJinaEmbedder {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, E> {
        let batch_size = batch_size.unwrap_or(32);
        let encodings = text_batch
            .par_chunks(batch_size)
            .flat_map(|mini_text_batch| -> Result<Vec<Vec<f32>>, E> {
                let token_ids: Array2<i64> = self.tokenize_batch(mini_text_batch)?;
                let token_type_ids: Array2<i64> = Array2::zeros(token_ids.raw_dim());
                let attention_mask: Array2<i64> = Array2::ones(token_ids.raw_dim());

                let embeddings = if self.model == ONNXModel::JINAV3 {
                    let outputs = self.session.run(ort::inputs! {
                        "input_ids" => token_ids,
                        "attention_mask" => attention_mask,
                        "task_id" => Array1::<i64>::from_vec(vec![4])
                    }?)?;
                    outputs["text_embeds"]
                        .try_extract_tensor::<f32>()?
                        .to_owned()
                        .into_dimensionality::<ndarray::Ix3>()?
                } else {
                    let outputs = self.session.run(ort::inputs! {
                        "input_ids" => token_ids,
                        "token_type_ids" => token_type_ids,
                        "attention_mask" => attention_mask
                    }?)?;
                    outputs["last_hidden_state"]
                        .try_extract_tensor::<f32>()?
                        .to_owned()
                        .into_dimensionality::<ndarray::Ix3>()?
                };

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

///jina-embeddings-v2-base-en is an English, monolingual embedding model supporting 8192 sequence length. It is based on a BERT architecture (JinaBERT) that supports the symmetric bidirectional variant of ALiBi to allow longer sequence length. The backbone jina-bert-v2-base-en is pretrained on the C4 dataset. The model is further trained on Jina AI's collection of more than 400 millions of sentence pairs and hard negatives. These pairs were obtained from various domains and were carefully selected through a thorough cleaning process.
///
///The embedding model was trained using 512 sequence length, but extrapolates to 8k sequence length (or even longer) thanks to ALiBi. This makes our model useful for a range of use cases, especially when processing long documents is needed, including long document retrieval, semantic textual similarity, text reranking, recommendation, RAG and LLM-based generative search, etc.
///
///With a standard size of 137 million parameters, the model enables fast inference while delivering better performance than our small model. It is recommended to use a single GPU for inference. Additionally, we provide the following embedding models:
///
///- jina-embeddings-v2-small-en: 33 million parameters.
///- jina-embeddings-v2-base-en: 137 million parameters .
///- jina-embeddings-v2-base-zh: Chinese-English Bilingual embeddings.
///- jina-embeddings-v2-base-de: German-English Bilingual embeddings.
///- jina-embeddings-v2-base-es: Spanish-English Bilingual embedding
pub struct JinaEmbedder {
    pub model: BertModel,
    pub tokenizer: Tokenizer,
}

impl Default for JinaEmbedder {
    fn default() -> Self {
        Self::new("jinaai/jina-embeddings-v2-small-en", None).unwrap()
    }
}

impl JinaEmbedder {
    pub fn new(model_id: &str, revision: Option<&str>) -> Result<Self, E> {
        let api = hf_hub::api::sync::Api::new()?;
        let api = match revision {
            Some(rev) => api.repo(Repo::with_revision(
                model_id.to_string(),
                hf_hub::RepoType::Model,
                rev.to_string(),
            )),
            None => api.repo(Repo::new(model_id.to_string(), hf_hub::RepoType::Model)),
        };

        let config_filename = api.get("config.json")?;
        let tokenizer_filename = api.get("tokenizer.json")?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let device = select_device();
        let vb = match api.get("model.safetensors") {
            Ok(safetensors) => unsafe {
                VarBuilder::from_mmaped_safetensors(&[safetensors], DType::F32, &device)?
            },
            Err(_) => match api.get("pytorch_model.bin") {
                Ok(pytorch_model) => VarBuilder::from_pth(pytorch_model, DType::F32, &device)?,
                Err(e) => {
                    return Err(anyhow::Error::msg(format!(
                        "Model weights not found. The weights should either be a `model.safetensors` or `pytorch_model.bin` file.  Error: {}",
                        e
                    )));
                }
            },
        };
        let model = BertModel::new(vb, &config)?;
        // let mut tokenizer = Self::get_tokenizer(None)?;
        let pp = tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
        Ok(Self { model, tokenizer })
    }

    pub fn tokenize_batch(&self, text_batch: &[String], device: &Device) -> anyhow::Result<Tensor> {
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

    pub fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let mut encodings: Vec<EmbeddingResult> = Vec::new();
        let batch_size = batch_size.unwrap_or(32);
        for mini_text_batch in text_batch.chunks(batch_size) {
            let token_ids = self
                .tokenize_batch(mini_text_batch, &self.model.device)
                .unwrap();
            let embeddings = self.model.forward(&token_ids).unwrap();
            let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3().unwrap();

            let embeddings = (embeddings.sum(1).unwrap() / (n_tokens as f64)).unwrap();
            let embeddings = normalize_l2(&embeddings).unwrap();

            // Avoid using to_vec2() and instead work with the Tensor directly
            encodings.extend((0..embeddings.dim(0)?).map(|i| {
                EmbeddingResult::DenseVector(embeddings.get(i).unwrap().to_vec1().unwrap())
            }));
        }

        Ok(encodings)
    }
}

impl JinaEmbed for JinaEmbedder {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        self.embed(text_batch, batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed() {
        let embedder = JinaEmbedder::new("jinaai/jina-embeddings-v2-small-en", None).unwrap();
        let text_batch = vec!["Hello, world!".to_string()];

        let encodings = embedder.embed(&text_batch, None).unwrap();
        println!("{:?}", encodings);
    }
}
