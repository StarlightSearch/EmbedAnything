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
        model: &str,
        model_id: &str,
        revision: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        match model {
            "openai"|"OpenAI" => Ok(Self::OpenAI(OpenAIEmbeder::default())),
            "cohere"|"Cohere" => Ok(Self::Cohere(CohereEmbeder::default())),
            "jina"|"Jina" => Ok(Self::Jina(JinaEmbeder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
            )?)),
            "CLIP"|"Clip"|"clip" => Ok(Self::Clip(ClipEmbeder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
            )?)),
            "Bert"|"bert"=> Ok(Self::Bert(BertEmbeder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
            )?)),
            _ => Err(anyhow::anyhow!("Model not supported")),
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

    fn from_pretrained(
        &self,
        model_id: &str,
        revision: Option<&str>,
    ) -> Result<Self, anyhow::Error>
    where
        Self: Sized;
}
