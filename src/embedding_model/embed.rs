use pyo3::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::file_processor::audio::audio_processor::Segment;

use super::bert::BertEmbeder;
use super::clip::ClipEmbeder;
use super::jina::JinaEmbeder;
use super::openai::OpenAIEmbeder;

#[derive(Deserialize, Debug, Default)]
pub struct EmbedResponse {
    pub data: Vec<EmbedData>,
    pub usage: HashMap<String, usize>,
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

pub enum Embeder {
    OpenAI(OpenAIEmbeder),
    Jina(JinaEmbeder),
    Clip(ClipEmbeder),
    Bert(BertEmbeder),
}

impl Embeder {
    pub fn embed(
        &self,
        text_batch: &[String],
        metadata: Option<HashMap<String, String>>,
    ) -> Result<Vec<EmbedData>, anyhow::Error> {
        match self {
            Embeder::OpenAI(embeder) => TextEmbed::embed(embeder, text_batch, metadata),
            Embeder::Jina(embeder) => TextEmbed::embed(embeder, text_batch, metadata),
            Embeder::Clip(embeder) => Embed::embed(embeder, text_batch, metadata),
            Embeder::Bert(embeder) => TextEmbed::embed(embeder, text_batch, metadata),
        }
    }
}

pub trait Embed {
    fn embed(
        &self,
        text_batch: &[String],
        metadata: Option<HashMap<String, String>>,
    ) -> Result<Vec<EmbedData>, anyhow::Error>;
}

pub trait TextEmbed {
    fn embed(
        &self,
        text_batch: &[String],
        metadata: Option<HashMap<String, String>>,
    ) -> Result<Vec<EmbedData>, anyhow::Error>;
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

pub trait AudioEmbed {
    fn embed_audio<T: AsRef<std::path::Path>>(
        &self,
        segments: Vec<Segment>,
        audio_file: T,
    ) -> anyhow::Result<Vec<EmbedData>>;
}
