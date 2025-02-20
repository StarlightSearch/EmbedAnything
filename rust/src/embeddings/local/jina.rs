#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use super::bert::TokenizerConfig;
use super::pooling::{ModelOutput, PooledOutputType, Pooling};
use crate::embeddings::select_device;
use crate::embeddings::utils::tokenize_batch;
use crate::embeddings::{embed::EmbeddingResult, normalize_l2};
use crate::models::jina_bert::{BertModel, Config};
use anyhow::Error as E;
use candle_core::DType;
use candle_nn::{Module, VarBuilder};
use hf_hub::Repo;

use tokenizers::Tokenizer;

pub trait JinaEmbed {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error>;
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
        Self::new("jinaai/jina-embeddings-v2-small-en", None, None).unwrap()
    }
}

impl JinaEmbedder {
    pub fn new(model_id: &str, revision: Option<&str>, token: Option<&str>) -> Result<Self, E> {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_token(token.map(|s| s.to_string()))
            .build()?;
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
        let tokenizer_config_filename = api.get("tokenizer_config.json")?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

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
        let trunc = tokenizers::TruncationParams {
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            max_length,
            ..Default::default()
        };
        tokenizer
            .with_padding(Some(pp))
            .with_truncation(Some(trunc))
            .unwrap();
        Ok(Self { model, tokenizer })
    }

    pub fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let mut encodings: Vec<EmbeddingResult> = Vec::new();
        let batch_size = batch_size.unwrap_or(32);
        for mini_text_batch in text_batch.chunks(batch_size) {
            let (token_ids, attention_mask) =
                tokenize_batch(&self.tokenizer, mini_text_batch, &self.model.device)?;

            let embeddings = self.model.forward(&token_ids)?;
            let attention_mask = PooledOutputType::from(attention_mask);
            let attention_mask = Some(&attention_mask);
            let model_output = ModelOutput::Tensor(embeddings.clone());
            let pooled_output = Pooling::Mean.pool(&model_output, attention_mask)?;

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

impl JinaEmbed for JinaEmbedder {
    fn embed(
        &self,
        text_batch: &[&str],
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
        let embedder = JinaEmbedder::new("jinaai/jina-embeddings-v2-small-en", None, None).unwrap();
        let text_batch = vec!["Hello, world!"];

        let encodings = embedder.embed(&text_batch, None).unwrap();
        println!("{:?}", encodings);
    }
}
