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

pub trait TextEmbed {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, anyhow::Error>;

    fn from_pretrained(
        &self,
        model_id: &str,
        revision: Option<&str>,
    ) -> Result<Self, anyhow::Error>
    where
        Self: Sized;
}

pub enum Embeder {
    OpenAI(OpenAIEmbeder),
    Cohere(CohereEmbeder),
    Jina(JinaEmbeder),
    Clip(ClipEmbeder),
    Bert(BertEmbeder),
}

impl Embeder {
    pub fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        match self {
            Embeder::OpenAI(embeder) => embeder.embed(text_batch),
            Embeder::Cohere(embeder) => embeder.embed(text_batch),
            Embeder::Jina(embeder) => embeder.embed(text_batch, batch_size),
            Embeder::Clip(embeder) => embeder.embed(text_batch, batch_size),
            Embeder::Bert(embeder) => embeder.embed(text_batch, batch_size),
        }
    }

    pub fn from_pretrained(
        &self,
        model_id: &str,
        revision: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        match self {
            Embeder::OpenAI(_) => Ok(Self::OpenAI(OpenAIEmbeder::default())),
            Embeder::Cohere(_) => Ok(Self::Cohere(CohereEmbeder::default())),
            Embeder::Jina(_) => Ok(Self::Jina(JinaEmbeder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
            )?)),
            Embeder::Clip(_) => Ok(Self::Clip(ClipEmbeder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
            )?)),
            Embeder::Bert(_) => Ok(Self::Bert(BertEmbeder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
            )?)),
        }
    }
}

impl TextEmbed for Embeder {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        match self {
            Self::OpenAI(embeder) => embeder.embed(text_batch),
            Self::Cohere(embeder) => embeder.embed(text_batch),
            Self::Jina(embeder) => embeder.embed(text_batch, batch_size),
            Self::Clip(embeder) => embeder.embed(text_batch, batch_size),
            Self::Bert(embeder) => embeder.embed(text_batch, batch_size),
        }
    }

    fn from_pretrained(
        &self,
        model_id: &str,
        revision: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        match self {
            Self::OpenAI(_) => Ok(Self::OpenAI(OpenAIEmbeder::default())),
            Self::Cohere(_) => Ok(Self::Cohere(CohereEmbeder::default())),
            Self::Jina(_) => Ok(Self::Jina(JinaEmbeder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
            )?)),
            Self::Clip(_) => Ok(Self::Clip(ClipEmbeder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
            )?)),
            Self::Bert(_) => Ok(Self::Bert(BertEmbeder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
            )?)),
        }
    }
}

pub enum CloudEmbeder {
    OpenAI(OpenAIEmbeder),
    Cohere(CohereEmbeder),
}

impl CloudEmbeder {
    pub fn embed(&self, text_batch: &[String]) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        match self {
            Self::OpenAI(embeder) => embeder.embed(text_batch),
            Self::Cohere(embeder) => embeder.embed(text_batch),
        }
    }
}

impl TextEmbed for CloudEmbeder {
    fn embed(
        &self,
        text_batch: &[String],
        _batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        match self {
            Self::OpenAI(embeder) => embeder.embed(text_batch),
            Self::Cohere(embeder) => embeder.embed(text_batch),
        }
    }

    fn from_pretrained(
        &self,
        _model_id: &str,
        _revision: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        match self {
            Self::OpenAI(_) => Ok(Self::OpenAI(OpenAIEmbeder::default())),
            Self::Cohere(_) => Ok(Self::Cohere(CohereEmbeder::default())),
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

    fn from_pretrained(
        &self,
        model_id: &str,
        revision: Option<&str>,
    ) -> Result<Self, anyhow::Error>
    where
        Self: Sized;
}
