use crate::file_processor::audio::audio_processor::Segment;

use super::cloud::cohere::CohereEmbedder;
use super::cloud::openai::OpenAIEmbedder;
use super::local::bert::{BertEmbed, BertEmbedder, OrtBertEmbedder, SparseBertEmbedder};
use super::local::clip::ClipEmbedder;
use super::local::colpali::{ColPaliEmbed, ColPaliEmbedder};
use super::local::jina::JinaEmbedder;
use super::local::text_embedding::ONNXModel;
use anyhow::anyhow;
use candle_core::WithDType;
use half::f16;
use num_traits::cast::FromPrimitive;
use num_traits::Float;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

pub trait NumericalType:
    std::fmt::Debug
    + Clone
    + WithDType
    + ort::PrimitiveTensorElementType
    + FromPrimitive
    + Float
    + Send
    + Sync
{
    fn from_f32(value: f32) -> Self;
    fn from_f16(value: f16) -> Self;
}

impl NumericalType for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }

    fn from_f16(value: f16) -> Self {
        value.to_f32()
    }
}
impl NumericalType for f16 {
    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }

    fn from_f16(value: f16) -> Self {
        value
    }
}

#[derive(Debug, Clone)]
pub enum EmbeddingResult<T: NumericalType> {
    DenseVector(Vec<T>),
    MultiVector(Vec<Vec<T>>),
}

impl<T: NumericalType> From<Vec<T>> for EmbeddingResult<T> {
    fn from(value: Vec<T>) -> Self {
        EmbeddingResult::DenseVector(value)
    }
}

impl<T: NumericalType> From<Vec<Vec<T>>> for EmbeddingResult<T> {
    fn from(value: Vec<Vec<T>>) -> Self {
        EmbeddingResult::MultiVector(value)
    }
}

impl<T: NumericalType> EmbeddingResult<T> {
    pub fn to_dense(&self) -> Result<Vec<T>, anyhow::Error> {
        match self {
            EmbeddingResult::DenseVector(x) => Ok(x.to_vec()),
            EmbeddingResult::MultiVector(_) => Err(anyhow!(
                "Sparse Embedding are not supported for this operation"
            )),
        }
    }

    pub fn to_multi_vector(&self) -> Result<Vec<Vec<T>>, anyhow::Error> {
        match self {
            EmbeddingResult::MultiVector(x) => Ok(x.to_vec()),
            EmbeddingResult::DenseVector(_) => Err(anyhow!(
                "Dense Embedding are not supported for this operation"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbedData<T: NumericalType> {
    pub embedding: EmbeddingResult<T>,
    pub text: Option<String>,
    pub metadata: Option<HashMap<String, String>>,
}

impl<T: NumericalType> EmbedData<T> {
    pub fn new(
        embedding: EmbeddingResult<T>,
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

pub enum TextEmbedder<F: NumericalType> {
    OpenAI(OpenAIEmbedder),
    Cohere(CohereEmbedder),
    Jina(JinaEmbedder),
    Bert(Box<dyn BertEmbed<F> + Send + Sync>),
}

impl<F: NumericalType + for<'de> Deserialize<'de>> TextEmbedder<F> {
    pub async fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult<F>>, anyhow::Error> {
        match self {
            TextEmbedder::OpenAI(embeder) => embeder.embed(text_batch).await,
            TextEmbedder::Cohere(embeder) => embeder.embed(text_batch).await,
            TextEmbedder::Jina(embeder) => embeder.embed(text_batch, batch_size),
            TextEmbedder::Bert(embeder) => embeder.embed(text_batch, batch_size),
        }
    }

    pub fn from_pretrained_hf(
        model: &str,
        model_id: &str,
        revision: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        match model {
            "jina" | "Jina" => Ok(Self::Jina(JinaEmbedder::new(model_id, revision)?)),

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
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }

    /// Creates a new instance of a cloud api based `Embeder` with the specified model and API key.
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
    /// A new instance of `Embeder`.
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

pub enum VisionEmbedder<F: NumericalType> {
    Clip(ClipEmbedder),
    ColPali(Box<dyn ColPaliEmbed<F> + Send + Sync>),
}

impl<F: NumericalType> From<VisionEmbedder<F>> for Embedder<F> {
    fn from(value: VisionEmbedder<F>) -> Self {
        Embedder::Vision(value)
    }
}

impl<F: NumericalType> From<Embedder<F>> for VisionEmbedder<F> {
    fn from(value: Embedder<F>) -> Self {
        match value {
            Embedder::Vision(value) => value,
            _ => panic!("Invalid embedder type"),
        }
    }
}

impl<F: NumericalType> From<Embedder<F>> for TextEmbedder<F> {
    fn from(value: Embedder<F>) -> Self {
        match value {
            Embedder::Text(value) => value,
            _ => panic!("Invalid embedder type"),
        }
    }
}

impl<F: NumericalType> VisionEmbedder<F> {
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

pub enum Embedder<F: NumericalType> {
    Text(TextEmbedder<F>),
    Vision(VisionEmbedder<F>),
}

impl<F: NumericalType + for<'de> Deserialize<'de>> Embedder<F> {
    pub async fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult<F>>, anyhow::Error> {
        match self {
            Self::Text(embeder) => embeder.embed(text_batch, batch_size).await,
            Self::Vision(embeder) => embeder.embed(text_batch, batch_size),
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

impl<F: NumericalType, T: AsRef<std::path::Path>> EmbedImage<F, T> for Embedder<F> {
    fn embed_image(
        &self,
        image_path: T,
        metadata: Option<HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData<F>> {
        match self {
            Self::Vision(embeder) => embeder.embed_image(image_path, metadata),
            _ => Err(anyhow::anyhow!("Model not supported for vision embedding")),
        }
    }

    fn embed_image_batch(&self, image_paths: &[T]) -> anyhow::Result<Vec<EmbedData<F>>> {
        match self {
            Self::Vision(embeder) => embeder.embed_image_batch(image_paths),
            _ => Err(anyhow::anyhow!("Model not supported for vision embedding")),
        }
    }
}

pub trait TextEmbed<F: NumericalType> {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult<F>>, anyhow::Error>;
}

impl<F: NumericalType> TextEmbed<F> for VisionEmbedder<F> {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult<F>>, anyhow::Error> {
        match self {
            Self::Clip(embeder) => embeder.embed(text_batch, batch_size),
            Self::ColPali(embeder) => embeder.embed(text_batch, batch_size),
        }
    }
}

pub trait EmbedImage<F: NumericalType, T: AsRef<std::path::Path>> {
    fn embed_image(
        &self,
        image_path: T,
        metadata: Option<HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData<F>>;
    fn embed_image_batch(&self, image_paths: &[T]) -> anyhow::Result<Vec<EmbedData<F>>>;
}

impl<F: NumericalType, T: AsRef<std::path::Path>> EmbedImage<F, T> for VisionEmbedder<F> {
    fn embed_image(
        &self,
        image_path: T,
        metadata: Option<HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData<F>> {
        match self {
            Self::Clip(embeder) => embeder.embed_image(image_path, metadata),
            Self::ColPali(embeder) => {
                embeder.embed_image(PathBuf::from(image_path.as_ref()), metadata)
            }
        }
    }

    fn embed_image_batch(&self, image_paths: &[T]) -> anyhow::Result<Vec<EmbedData<F>>> {
        match self {
            Self::Clip(embeder) => embeder.embed_image_batch(image_paths),
            Self::ColPali(embeder) => embeder.embed_image_batch(
                &image_paths
                    .iter()
                    .map(|p| PathBuf::from(p.as_ref()))
                    .collect::<Vec<_>>(),
            ),
        }
    }
}
