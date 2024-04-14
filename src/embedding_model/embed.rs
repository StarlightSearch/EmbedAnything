use pyo3::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;


use super::jina::JinaEmbeder;
use super:: openai::OpenAIEmbeder;
use super :: clip::ClipEmbeder;
use super::bert::BertEmbeder;
#[derive(Deserialize, Debug)]
pub struct EmbedResponse {
    pub data: Vec<EmbedData>,
    pub usage: HashMap<String, usize>,
}
#[pyclass]
#[derive(Deserialize, Debug)]
pub struct EmbedData {
    #[pyo3(get, set)]
    pub embedding: Vec<f32>,
    #[pyo3(get, set)]
    pub text: Option<String>,
}

#[pymethods]
impl EmbedData {
    #[new]
    pub fn new(embedding: Vec<f32>, text: Option<String>) -> Self {
        Self { embedding, text }
    }

    pub fn __str__(&self) -> String {
        format!(
            "EmbedData(embedding: {:?}, text: {:?})",
            self.embedding, self.text
        )
    }
}

pub enum Embeder {
    OpenAI(OpenAIEmbeder),
    Jina(JinaEmbeder),
    Clip(ClipEmbeder),
    Bert(BertEmbeder),
}

impl Embeder {
    pub async fn embed(&self, text_batch: &[String]) -> Result<Vec<EmbedData>, reqwest::Error> {
        match self {
            Embeder::OpenAI(embeder) => embeder.embed(text_batch).await,
            Embeder::Jina(embeder) => embeder.embed(text_batch).await,
            Embeder::Clip(embeder) => embeder.embed(text_batch).await,
            Embeder::Bert(embeder) => embeder.embed(text_batch).await,
        }
    }
}


pub trait Embed {
    fn embed(
        &self,
        text_batch: &[String],
    ) -> impl std::future::Future<Output = Result<Vec<EmbedData>, reqwest::Error>>;
    
}

pub trait EmbedImage {
    fn embed_image_batch<T: AsRef<std::path::Path>>(&self, image_paths:&[T]) -> anyhow::Result<Vec<EmbedData>>;
}