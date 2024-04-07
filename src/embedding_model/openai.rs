use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use crate::embedding_model::embed::{EmbedData, EmbedResponse};

use super::embed::Embed;

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
