#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use super::embed::TextEmbed;
use crate::embedding_model::normalize_l2;
use anyhow::Error as E;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::jina_bert::{BertModel, Config};
use hf_hub::Repo;
use tokenizers::Tokenizer;


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
pub struct JinaEmbeder {
    pub model: BertModel,
    pub tokenizer: Tokenizer,
}

impl Default for JinaEmbeder {
    fn default() -> Self {
        Self::new("jinaai/jina-embeddings-v2-base-en".to_string(), None).unwrap()
    }
}

impl JinaEmbeder {
    pub fn new(model_id: String, revision: Option<String>) -> Result<Self, E> {
        let api = hf_hub::api::sync::Api::new()?;
        let api = match revision {
            Some(rev) => api.repo(Repo::with_revision(model_id, hf_hub::RepoType::Model, rev)),
            None => api.repo(Repo::new(model_id.to_string(), hf_hub::RepoType::Model)),
        };
        let model_file = api.get("model.safetensors")?;
        let config_filename = api.get("config.json")?;
        let tokenizer_filename = api.get("tokenizer.json")?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_file.clone()], DType::F32, &device)?
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

    pub fn embed(&self, text_batch: &[String]) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        let token_ids = self.tokenize_batch(text_batch, &self.model.device).unwrap();
        let embeddings = self.model.forward(&token_ids).unwrap();

        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3().unwrap();

        let embeddings = (embeddings.sum(1).unwrap() / (n_tokens as f64)).unwrap();
        let embeddings = normalize_l2(&embeddings).unwrap();

        let encodings = embeddings.to_vec2::<f32>().unwrap();
        Ok(encodings)
    }

   
}

impl TextEmbed for JinaEmbeder {
    fn embed(&self, text_batch: &[String]) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        self.embed(text_batch)
    }
}

