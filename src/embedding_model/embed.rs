use super::bert::BertEmbeder;
use super::clip::ClipEmbeder;
use super::cohere::CohereEmbeder;
use super::jina::JinaEmbeder;
use super::openai::OpenAIEmbeder;
use pyo3::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fmt::Debug;

#[derive(Deserialize, Debug, Default)]
pub struct OpenAIEmbedResponse {
    pub data: Vec<EmbedData>,
    pub usage: HashMap<String, usize>,
}

#[derive(Deserialize, Debug, Default)]
pub struct CohereEmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
}

#[pyclass]
#[derive(Deserialize, Debug, Clone, Default)]
pub struct EmbedData {
    #[pyo3(get, set)]
    pub embedding: Vec<f32>,
    #[pyo3(get, set)]
    pub text: Option<String>,
    #[pyo3(get, set)]
    pub metadata: Option<HashMap<String, String>>,
}

#[pymethods]
impl EmbedData {
    #[new]
    #[pyo3(signature = (embedding, text=None, metadata=None))]
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
    fn embed(&self, text_batch: &[String], batch_size: Option<usize>) -> Result<Vec<Vec<f32>>, anyhow::Error>;
}
pub enum Embeder {
    Cloud(CloudEmbeder),
    Jina(JinaEmbeder),
    Clip(ClipEmbeder),
    Bert(BertEmbeder),
}

impl Embeder {
    pub fn embed(&self, text_batch: &[String], batch_size:Option<usize>) -> Result<Vec<Vec<f32>>, anyhow::Error> {
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
    fn embed(&self, text_batch: &[String], _batch_size: Option<usize>) -> Result<Vec<Vec<f32>>, anyhow::Error> {
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
