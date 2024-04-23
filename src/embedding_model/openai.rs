use std::collections::HashMap;

use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use crate::embedding_model::embed::{EmbedData, EmbedResponse};

use super::embed::{Embed, TextEmbed};

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
    fn embed(
        &self,
        text_batch: &[String],
        metadata: Option<HashMap<String, String>>,
    ) -> Result<Vec<EmbedData>, anyhow::Error> {
        self.embed(text_batch, metadata)
    }
}

impl TextEmbed for OpenAIEmbeder {
    fn embed(
        &self,
        text_batch: &[String],
        metadata: Option<HashMap<String, String>>,
    ) -> Result<Vec<EmbedData>, anyhow::Error> {
        self.embed(text_batch, metadata)
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

    fn embed(
        &self,
        text_batch: &[String],
        metadata: Option<HashMap<String, String>>,
    ) -> Result<Vec<EmbedData>, anyhow::Error> {
        let client = Client::new();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .build()
            .unwrap();

        let data = runtime.block_on(async move {
            let response = client
                .post(&self.url)
                .header("Content-Type", "application/json")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(&json!({
                    "input": text_batch,
                    "model": "text-embedding-3-small",
                }))
                .send()
                .await
                .unwrap();

            let data = response.json::<EmbedResponse>().await.unwrap();
            println!("{:?}", data.usage);
            data
        });

        let emb_data = data
            .data
            .iter()
            .zip(text_batch)
            .map(move |(data, text)| {
                EmbedData::new(data.embedding.clone(), Some(text.clone()), metadata.clone())
            })
            .collect::<Vec<_>>();

        Ok(emb_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_embed() {
        let openai = OpenAIEmbeder::default();
        let text_batch = vec![
            "Once upon a time".to_string(),
            "The quick brown fox jumps over the lazy dog".to_string(),
        ];

        let embeddings = openai.embed(&text_batch, None).unwrap();
        assert_eq!(embeddings.len(), 2);
    }
}
