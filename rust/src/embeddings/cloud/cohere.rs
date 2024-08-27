use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use crate::embeddings::embed::TextEmbed;

#[derive(Deserialize, Debug, Default)]
pub struct CohereEmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
}

/// Represents an CohereEmebeder struct that contains the URL and API key for making requests to the OpenAI API.
#[derive(Debug)]
pub struct CohereEmbeder {
    url: String,
    model: String,
    api_key: String,
    runtime: tokio::runtime::Runtime,
    client: Client,
}

impl Default for CohereEmbeder {
    fn default() -> Self {
        Self::new("embed-english-v3.0".to_string(), None)
    }
}

impl TextEmbed for CohereEmbeder {
    fn embed(
        &self,
        text_batch: &[String],
        _batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        self.embed(text_batch)
    }

    fn from_pretrained(
        &self,
        model_id: &str,
        _revision: Option<&str>,
    ) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        Ok(Self::new(model_id.to_string(), None))
    }
}

impl CohereEmbeder {
    pub fn new(model: String, api_key: Option<String>) -> Self {
        let api_key =
            api_key.unwrap_or_else(|| std::env::var("CO_API_KEY").expect("API key not set"));

        Self {
            model,
            url: "https://api.cohere.com/v1/embed".to_string(),
            api_key,
            runtime: tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .build()
                .unwrap(),
            client: Client::new(),
        }
    }

    pub fn embed(&self, text_batch: &[String]) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        let data = self.runtime.block_on(async move {
            let response = self
                .client
                .post(&self.url)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(&json!({
                    "texts": text_batch,
                    "model": self.model,
                    "input_type": "search_document"
                }))
                .send()
                .await
                .unwrap();
            response.json::<CohereEmbedResponse>().await.unwrap()
        });

        let encodings = data.embeddings;

        Ok(encodings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cohere_embed() {
        let cohere = CohereEmbeder::default();
        let text_batch = vec![
            "Once upon a time".to_string(),
            "The quick brown fox jumps over the lazy dog".to_string(),
        ];

        let embeddings = cohere.embed(&text_batch).unwrap();
        assert_eq!(embeddings.len(), 2);
    }
}
