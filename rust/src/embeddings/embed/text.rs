use crate::embeddings::cloud::cohere::CohereEmbedder;
use crate::embeddings::cloud::gemini::GeminiEmbedder;
use crate::embeddings::cloud::openai::OpenAIEmbedder;
use crate::embeddings::local::bert::{BertEmbed, BertEmbedder, SparseBertEmbedder};
use crate::embeddings::local::jina::{JinaEmbed, JinaEmbedder};
use crate::embeddings::local::model2vec::Model2VecEmbedder;
use crate::embeddings::local::modernbert::ModernBertEmbedder;
use crate::embeddings::local::qwen3::{Qwen3Embed, Qwen3Embedder};
use crate::embeddings::local::text_embedding::ONNXModel;
use crate::Dtype;
use anyhow::{anyhow, Result};
use std::future::Future;

use super::types::EmbeddingResult;

#[cfg(feature = "ort")]
use crate::embeddings::local::colbert::OrtColbertEmbedder;
#[cfg(feature = "ort")]
use crate::embeddings::local::ort_bert::{OrtBertEmbedder, OrtSparseBertEmbedder};
#[cfg(feature = "ort")]
use crate::embeddings::local::ort_jina::OrtJinaEmbedder;

pub enum TextEmbedder {
    OpenAI(OpenAIEmbedder),
    Cohere(CohereEmbedder),
    Gemini(GeminiEmbedder),
    Jina(Box<dyn JinaEmbed + Send + Sync>),
    Model2Vec(Box<Model2VecEmbedder>),
    Bert(Box<dyn BertEmbed + Send + Sync>),
    Qwen3(Box<dyn Qwen3Embed + Send + Sync>),
    ColBert(Box<dyn BertEmbed + Send + Sync>),
    ModernBert(Box<dyn BertEmbed + Send + Sync>),
}

impl TextEmbedder {
    pub async fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
        late_chunking: Option<bool>,
    ) -> Result<Vec<EmbeddingResult>> {
        match self {
            TextEmbedder::OpenAI(embedder) => embedder.embed(text_batch).await,
            TextEmbedder::Cohere(embedder) => embedder.embed(text_batch).await,
            TextEmbedder::Gemini(embedder) => embedder.embed(text_batch).await,
            TextEmbedder::Model2Vec(embedder) => embedder.embed(text_batch, batch_size),
            TextEmbedder::Jina(embedder) => embedder.embed(text_batch, batch_size, late_chunking),
            TextEmbedder::Bert(embedder) => embedder.embed(text_batch, batch_size, late_chunking),
            TextEmbedder::Qwen3(embedder) => embedder.embed(text_batch, batch_size, late_chunking),
            TextEmbedder::ColBert(embedder) => {
                embedder.embed(text_batch, batch_size, late_chunking)
            }
            TextEmbedder::ModernBert(embedder) => {
                embedder.embed(text_batch, batch_size, late_chunking)
            }
        }
    }

    pub fn from_pretrained_hf(
        architecture: &str,
        model_id: &str,
        revision: Option<&str>,
        token: Option<&str>,
        dtype: Option<Dtype>,
    ) -> Result<Self> {
        match architecture {
            "JinaBertForMaskedLM" => Ok(Self::Jina(Box::new(JinaEmbedder::new(
                model_id, revision, token,
            )?))),

            "BertModel" => Ok(Self::Bert(Box::new(BertEmbedder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
                token,
            )?))),
            "BertForMaskedLM" => Ok(Self::Bert(Box::new(SparseBertEmbedder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
                token,
            )?))),
            "StaticModel" => Ok(Self::Model2Vec(Box::new(Model2VecEmbedder::new(
                model_id, token, None,
            )?))),

            "ModernBertForMaskedLM" => Ok(Self::ModernBert(Box::new(ModernBertEmbedder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
                token,
                dtype,
            )?))),
            "Qwen3ForCausalLM" => Ok(Self::Qwen3(Box::new(Qwen3Embedder::new(
                model_id,
                revision.map(|s| s.to_string()),
                token,
                dtype,
            )?))),
            _ => Err(anyhow!("Model not supported")),
        }
    }

    #[cfg(feature = "ort")]
    pub fn from_pretrained_ort(
        model_architecture: &str,
        model_name: Option<ONNXModel>,
        revision: Option<&str>,
        model_id: Option<&str>,
        dtype: Option<Dtype>,
        path_in_repo: Option<&str>,
    ) -> Result<Self> {
        if model_name.is_some() {
            match model_architecture {
                "Bert" | "bert" => Ok(Self::Bert(Box::new(OrtBertEmbedder::new(
                    model_name,
                    model_id,
                    revision,
                    dtype,
                    path_in_repo,
                )?))),
                "sparse-bert" | "SparseBert" | "SPARSE-BERT" => Ok(Self::Bert(Box::new(
                    OrtSparseBertEmbedder::new(model_name, model_id, revision, path_in_repo)?,
                ))),
                "jina" | "Jina" => Ok(Self::Jina(Box::new(OrtJinaEmbedder::new(
                    model_name,
                    model_id,
                    revision,
                    dtype,
                    path_in_repo,
                )?))),

                _ => Err(anyhow!("Model not supported")),
            }
        } else if model_id.is_some() {
            match model_architecture {
                "colbert" | "Colbert" | "COLBERT" => Ok(Self::ColBert(Box::new(
                    OrtColbertEmbedder::new(model_id, revision, path_in_repo)?,
                ))),
                "bert" | "Bert" => Ok(Self::Bert(Box::new(OrtBertEmbedder::new(
                    None,
                    model_id,
                    revision,
                    None,
                    path_in_repo,
                )?))),
                "jina" | "Jina" => Ok(Self::Jina(Box::new(OrtJinaEmbedder::new(
                    None,
                    model_id,
                    revision,
                    dtype,
                    path_in_repo,
                )?))),
                _ => Err(anyhow!("Model not supported")),
            }
        } else {
            Err(anyhow!("Please provide either model_name or model_id"))
        }
    }

    /// Creates a new instance of a cloud api based `Embedder` with the specified model and API key.
    pub fn from_pretrained_cloud(
        model: &str,
        model_id: &str,
        api_key: Option<String>,
    ) -> Result<Self> {
        match model {
            "openai" | "OpenAI" => Ok(Self::OpenAI(OpenAIEmbedder::new(
                model_id.to_string(),
                api_key,
            ))),
            "cohere" | "Cohere" => Ok(Self::Cohere(CohereEmbedder::new(
                model_id.to_string(),
                api_key,
            ))),
            "gemini" | "Gemini" => Ok(Self::Gemini(GeminiEmbedder::new(api_key))),
            _ => Err(anyhow!("Model not supported")),
        }
    }
}

pub trait TextEmbed {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> impl Future<Output = anyhow::Result<Vec<EmbeddingResult>>>;
}
