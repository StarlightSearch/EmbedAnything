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
}
pub enum Embeder {
    Cloud(CloudEmbeder),
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
            Embeder::Cloud(embeder) => embeder.embed(text_batch),
            Embeder::Jina(embeder) => embeder.embed(text_batch, batch_size),
            Embeder::Clip(embeder) => embeder.embed(text_batch, batch_size),
            Embeder::Bert(embeder) => embeder.embed(text_batch, batch_size),
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
