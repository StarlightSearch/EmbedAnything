use crate::file_processor::audio::audio_processor::Segment;

use super::cloud::cohere::CohereEmbeder;
use super::cloud::openai::OpenAIEmbeder;
use super::local::bert::BertEmbeder;
use super::local::clip::ClipEmbeder;
use super::local::jina::JinaEmbeder;
use serde::Deserialize;
use std::collections::HashMap;
use std::fmt::Debug;

#[derive(Deserialize, Debug, Clone, Default)]
pub struct EmbedData {
    pub embedding: Vec<f32>,
    pub text: Option<String>,
    pub metadata: Option<HashMap<String, String>>,
}

impl EmbedData {
    pub fn new(
        embedding: Vec<f32>,
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

pub enum Embeder {
    OpenAI(OpenAIEmbeder),
    Cohere(CohereEmbeder),
    Jina(JinaEmbeder),
    Clip(ClipEmbeder),
    Bert(BertEmbeder),
}

impl Embeder {
    pub async fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        match self {
            Embeder::OpenAI(embeder) => embeder.embed(text_batch).await,
            Embeder::Cohere(embeder) => embeder.embed(text_batch).await,
            Embeder::Jina(embeder) => embeder.embed(text_batch, batch_size),
            Embeder::Clip(embeder) => embeder.embed(text_batch, batch_size),
            Embeder::Bert(embeder) => embeder.embed(text_batch, batch_size),
        }
    }

    pub fn from_pretrained_hf(
        model: &str,
        model_id: &str,
        revision: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        match model {
            "jina" | "Jina" => Ok(Self::Jina(JinaEmbeder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
            )?)),
            "CLIP" | "Clip" | "clip" => Ok(Self::Clip(ClipEmbeder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
            )?)),
            "Bert" | "bert" => Ok(Self::Bert(BertEmbeder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
            )?)),
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
            "openai" | "OpenAI" => Ok(Self::OpenAI(OpenAIEmbeder::new(
                model_id.to_string(),
                api_key,
            ))),
            "cohere" | "Cohere" => Ok(Self::Cohere(CohereEmbeder::new(
                model_id.to_string(),
                api_key,
            ))),
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }
}

impl EmbedImage for Embeder {
    fn embed_image<T: AsRef<std::path::Path>>(
        &self,
        image_path: T,
        metadata: Option<HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData> {
        match self {
            Self::Clip(embeder) => embeder.embed_image(image_path, metadata),
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }

    fn embed_image_batch<T: AsRef<std::path::Path>>(
        &self,
        image_paths: &[T],
    ) -> anyhow::Result<Vec<EmbedData>> {
        match self {
            Self::Clip(embeder) => embeder.embed_image_batch(image_paths),
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }

    fn from_pretrained(model_id: &str, revision: Option<&str>) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        match model_id {
            "clip" | "Clip" | "CLIP" => Ok(Self::Clip(ClipEmbeder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
            )?)),
            _ => Err(anyhow::anyhow!("Model not supported")),
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

    fn from_pretrained(model_id: &str, revision: Option<&str>) -> Result<Self, anyhow::Error>
    where
        Self: Sized;
}
