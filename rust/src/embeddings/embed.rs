use crate::file_processor::audio::audio_processor::Segment;

use super::cloud::cohere::CohereEmbedder;
use super::cloud::openai::OpenAIEmbedder;
use super::local::bert::{BertEmbed, BertEmbedder, OrtBertEmbedder, OrtSparseBertEmbedder, SparseBertEmbedder};
use super::local::clip::ClipEmbedder;
use super::local::colpali::{ColPaliEmbed, ColPaliEmbedder};
use super::local::jina::{JinaEmbed, JinaEmbedder, OrtJinaEmbedder};
use super::local::text_embedding::ONNXModel;
use anyhow::anyhow;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Deserialize, Debug, Clone)]
pub enum EmbeddingResult {
    DenseVector(Vec<f32>),
    MultiVector(Vec<Vec<f32>>),
}

impl From<Vec<f32>> for EmbeddingResult {
    fn from(value: Vec<f32>) -> Self {
        EmbeddingResult::DenseVector(value)
    }
}

impl From<Vec<Vec<f32>>> for EmbeddingResult {
    fn from(value: Vec<Vec<f32>>) -> Self {
        EmbeddingResult::MultiVector(value)
    }
}

impl EmbeddingResult {
    pub fn to_dense(&self) -> Result<Vec<f32>, anyhow::Error> {
        match self {
            EmbeddingResult::DenseVector(x) => Ok(x.to_vec()),
            EmbeddingResult::MultiVector(_) => Err(anyhow!(
                "Sparse Embedding are not supported for this operation"
            )),
        }
    }

    pub fn to_multi_vector(&self) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        match self {
            EmbeddingResult::MultiVector(x) => Ok(x.to_vec()),
            EmbeddingResult::DenseVector(_) => Err(anyhow!(
                "Dense Embedding are not supported for this operation"
            )),
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct EmbedData {
    pub embedding: EmbeddingResult,
    pub text: Option<String>,
    pub metadata: Option<HashMap<String, String>>,
}

impl EmbedData {
    pub fn new(
        embedding: EmbeddingResult,
        text: Option<String>,
        metadata: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            embedding,
            text,
            metadata,
        }
    }

    pub fn __str__(&self) -> String {
        format!(
            "EmbedData(embedding: {:?}, text: {:?}, metadata: {:?})",
            self.embedding,
            self.text,
            self.metadata.clone()
        )
    }
}

pub trait AudioDecoder {
    fn decode_audio(&mut self, audio_file: &std::path::Path)
        -> Result<Vec<Segment>, anyhow::Error>;
}

pub enum TextEmbedder {
    OpenAI(OpenAIEmbedder),
    Cohere(CohereEmbedder),
    Jina(Box<dyn JinaEmbed + Send + Sync>),
    Bert(Box<dyn BertEmbed + Send + Sync>),
}

impl TextEmbedder {
    pub async fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        match self {
            TextEmbedder::OpenAI(embedder) => embedder.embed(text_batch).await,
            TextEmbedder::Cohere(embedder) => embedder.embed(text_batch).await,
            TextEmbedder::Jina(embedder) => embedder.embed(text_batch, batch_size),
            TextEmbedder::Bert(embedder) => embedder.embed(text_batch, batch_size),
        }
    }

    pub fn from_pretrained_hf(
        model: &str,
        model_id: &str,
        revision: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        match model {
            "jina" | "Jina" => Ok(Self::Jina(Box::new(JinaEmbedder::new(model_id, revision)?))),

            "Bert" | "bert" => Ok(Self::Bert(Box::new(BertEmbedder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
            )?))),
            "sparse-bert" | "SparseBert" | "SPARSE-BERT" => Ok(Self::Bert(Box::new(
                SparseBertEmbedder::new(model_id.to_string(), revision.map(|s| s.to_string()))?,
            ))),
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }

    pub fn from_pretrained_ort(
        model_architecture: &str,
        model_name: ONNXModel,
        revision: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        match model_architecture {
            "Bert" | "bert" => Ok(Self::Bert(Box::new(OrtBertEmbedder::new(
                model_name,
                revision.map(|s| s.to_string()),
            )?))),
            "sparse-bert" | "SparseBert" | "SPARSE-BERT" => Ok(Self::Bert(Box::new(
                OrtSparseBertEmbedder::new(model_name, revision.map(|s| s.to_string()))?,
            ))),
            "jina" | "Jina" => Ok(Self::Jina(Box::new(OrtJinaEmbedder::new(
                model_name, revision,
            )?))),
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }

    /// Creates a new instance of a cloud api based `Embedder` with the specified model and API key.
    ///
    /// # Arguments
    ///
    /// * `model` - A string holds the model to be used for embedding. Choose from
    ///             - "openai"
    ///             - "cohere"
    ///
    /// * `model_id` - A string holds the model ID for the model to be used for embedding.
    ///     - For OpenAI, find available models at https://platform.openai.com/docs/guides/embeddings/embedding-models
    ///     - For Cohere, find available models at https://docs.cohere.com/docs/cohere-embed
    /// * `api_key` - An optional string holds the API key for authenticating requests to the Cohere API. If not provided, it is taken from the environment variable
    ///         - For OpenAI, create environment variable `OPENAI_API_KEY`
    ///         - For Cohere, create environment variable `CO_API_KEY`
    ///
    /// # Returns
    ///
    /// A new instance of `Embedder`.
    pub fn from_pretrained_cloud(
        model: &str,
        model_id: &str,
        api_key: Option<String>,
    ) -> Result<Self, anyhow::Error> {
        match model {
            "openai" | "OpenAI" => Ok(Self::OpenAI(OpenAIEmbedder::new(
                model_id.to_string(),
                api_key,
            ))),
            "cohere" | "Cohere" => Ok(Self::Cohere(CohereEmbedder::new(
                model_id.to_string(),
                api_key,
            ))),
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }
}

pub enum VisionEmbedder {
    Clip(ClipEmbedder),
    ColPali(Box<dyn ColPaliEmbed + Send + Sync>),
}

impl From<VisionEmbedder> for Embedder {
    fn from(value: VisionEmbedder) -> Self {
        Embedder::Vision(value)
    }
}

impl From<Embedder> for VisionEmbedder {
    fn from(value: Embedder) -> Self {
        match value {
            Embedder::Vision(value) => value,
            _ => panic!("Invalid embedder type"),
        }
    }
}

impl From<Embedder> for TextEmbedder {
    fn from(value: Embedder) -> Self {
        match value {
            Embedder::Text(value) => value,
            _ => panic!("Invalid embedder type"),
        }
    }
}

impl VisionEmbedder {
    pub fn from_pretrained_hf(
        model: &str,
        model_id: &str,
        revision: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        match model {
            "clip" | "Clip" | "CLIP" => Ok(Self::Clip(ClipEmbedder::new(
                model_id.to_string(),
                revision,
            )?)),
            "colpali" | "ColPali" | "COLPALI" => Ok(Self::ColPali(Box::new(ColPaliEmbedder::new(
                model_id, revision,
            )?))),
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }
}

pub enum Embedder {
    Text(TextEmbedder),
    Vision(VisionEmbedder),
}

impl Embedder {
    pub async fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        match self {
            Self::Text(embedder) => embedder.embed(text_batch, batch_size).await,
            Self::Vision(embedder) => embedder.embed(text_batch, batch_size),
        }
    }

    pub fn from_pretrained_hf(
        model: &str,
        model_id: &str,
        revision: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        match model {
            "clip" | "Clip" | "CLIP" => Ok(Self::Vision(VisionEmbedder::from_pretrained_hf(
                model, model_id, revision,
            )?)),
            "colpali" | "ColPali" | "COLPALI" => Ok(Self::Vision(
                VisionEmbedder::from_pretrained_hf(model, model_id, revision)?,
            )),
            "bert" | "Bert" => Ok(Self::Text(TextEmbedder::from_pretrained_hf(
                model, model_id, revision,
            )?)),
            "jina" | "Jina" => Ok(Self::Text(TextEmbedder::from_pretrained_hf(
                model, model_id, revision,
            )?)),
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }

    pub fn from_pretrained_cloud(
        model: &str,
        model_id: &str,
        api_key: Option<String>,
    ) -> Result<Self, anyhow::Error> {
        match model {
            "openai" | "OpenAI" => Ok(Self::Text(TextEmbedder::from_pretrained_cloud(
                model, model_id, api_key,
            )?)),
            "cohere" | "Cohere" => Ok(Self::Text(TextEmbedder::from_pretrained_cloud(
                model, model_id, api_key,
            )?)),
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }

    pub fn from_pretrained_onnx(
        model_architecture: &str,
        model_name: ONNXModel,
        revision: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        Ok(Self::Text(TextEmbedder::from_pretrained_ort(
            model_architecture,
            model_name,
            revision,
        )?))
    }
}

impl EmbedImage for Embedder {
    fn embed_image<T: AsRef<std::path::Path>>(
        &self,
        image_path: T,
        metadata: Option<HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData> {
        match self {
            Self::Vision(embedder) => embedder.embed_image(image_path, metadata),
            _ => Err(anyhow::anyhow!("Model not supported for vision embedding")),
        }
    }

    fn embed_image_batch<T: AsRef<std::path::Path>>(
        &self,
        image_paths: &[T],
    ) -> anyhow::Result<Vec<EmbedData>> {
        match self {
            Self::Vision(embedder) => embedder.embed_image_batch(image_paths),
            _ => Err(anyhow::anyhow!("Model not supported for vision embedding")),
        }
    }
}

pub trait TextEmbed {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error>;
}

impl TextEmbed for VisionEmbedder {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        match self {
            Self::Clip(embedder) => embedder.embed(text_batch, batch_size),
            Self::ColPali(embedder) => embedder.embed(text_batch, batch_size),
        }
    }
}

pub trait EmbedImage {
    fn embed_image<T: AsRef<std::path::Path>>(
        &self,
        image_path: T,
        metadata: Option<HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData>;
    fn embed_image_batch<T: AsRef<std::path::Path>>(
        &self,
        image_paths: &[T],
    ) -> anyhow::Result<Vec<EmbedData>>;
}

impl EmbedImage for VisionEmbedder {
    fn embed_image<T: AsRef<std::path::Path>>(
        &self,
        image_path: T,
        metadata: Option<HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData> {
        match self {
            Self::Clip(embedder) => embedder.embed_image(image_path, metadata),
            Self::ColPali(embedder) => {
                embedder.embed_image(PathBuf::from(image_path.as_ref()), metadata)
            }
        }
    }

    fn embed_image_batch<T: AsRef<std::path::Path>>(
        &self,
        image_paths: &[T],
    ) -> anyhow::Result<Vec<EmbedData>> {
        match self {
            Self::Clip(embedder) => embedder.embed_image_batch(image_paths),
            Self::ColPali(embedder) => embedder.embed_image_batch(
                &image_paths
                    .iter()
                    .map(|p| PathBuf::from(p.as_ref()))
                    .collect::<Vec<_>>(),
            ),
        }
    }
}
