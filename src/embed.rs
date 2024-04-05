use futures::lock::Mutex;
use pyo3::prelude::*;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::{collections::HashMap, sync::Arc};

use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};

#[derive(Deserialize, Debug)]
pub struct EmbedResponse {
    pub data: Vec<EmbedData>,
    usage: HashMap<String, usize>,
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
    AllMiniLmL12V2(AllMiniLmL12V2Embeder),
}

impl Embeder {
    pub async fn embed(&self, text_batch: &[String]) -> Result<Vec<EmbedData>, reqwest::Error> {
        match self {
            Embeder::OpenAI(embeder) => embeder.embed(text_batch).await,
            Embeder::AllMiniLmL12V2(embeder) => embeder.embed(text_batch).await,
        }
    }
}

/// Represents an OpenAIEmbeder struct that contains the URL and API key for making requests to the OpenAI API.
#[derive(Deserialize, Debug)]
pub struct OpenAIEmbeder {
    url: String,
    api_key: String,
}

impl Default for OpenAIEmbeder {
    fn default() -> Self {
        Self::new(None)
    }
}

pub struct AllMiniLmL12V2Embeder {
    pub model: Arc<Mutex<SentenceEmbeddingsModel>>,
}
impl AllMiniLmL12V2Embeder {
    pub fn new() -> Self {
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
            .create_model()
            .unwrap();
        AllMiniLmL12V2Embeder {
            model: Arc::new(Mutex::new(model)),
        }
    }
}

pub trait Embed {
    fn embed(
        &self,
        text_batch: &[String],
    ) -> impl std::future::Future<Output = Result<Vec<EmbedData>, reqwest::Error>>;

}

impl Embed for AllMiniLmL12V2Embeder {
    async fn embed(&self, text_batch: &[String]) -> Result<Vec<EmbedData>, reqwest::Error> {
        let encodings = self.model.lock().await.encode(text_batch).unwrap();
        // let embeddings: Vec<EmbedData> = encodings
        //     .iter()
        //     .map(|i| EmbedData::new(Arc::new(i.to_vec())))
        //     .collect::<Vec<_>>();
        let embeddings = encodings
            .iter()
            .zip(text_batch)
            .map(|(data, text)| EmbedData::new(data.to_vec(), Some(text.clone())))
            .collect::<Vec<_>>();
        Ok(embeddings)
    }
}

impl Embed for OpenAIEmbeder {
    async fn embed(&self, text_batch: &[String]) -> Result<Vec<EmbedData>, reqwest::Error> {
        let client = Client::new();

        let response = client
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&json!({
                "input": text_batch,
                "model": "text-embedding-3-small",
            }))
            .send()
            .await?;

        let data = response.json::<EmbedResponse>().await?;
        println!("{:?}", data.usage);

        let emb_data = data
            .data
            .iter()
            .zip(text_batch)
            .map(move |(data, text)| EmbedData::new(data.embedding.clone(), Some(text.clone())))
            .collect::<Vec<_>>();

        Ok(emb_data)
    }
}

impl OpenAIEmbeder {
    pub fn new(api_key: Option<String>) -> Self {
        let api_key = api_key.unwrap_or_else(|| std::env::var("OPENAI_API_KEY").unwrap());

        Self {
            url: "https://api.openai.com/v1/embeddings".to_string(),
            api_key,
        }
    }
}
