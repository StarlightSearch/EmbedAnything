use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use crate::embeddings::embed::EmbeddingResult;

#[derive(Deserialize, Debug, Default)]
pub struct GeminiEmbedResponse {
    pub embeddings: Vec<GeminiEmbeddingData>,
}

#[derive(Deserialize, Debug, Default)]
pub struct GeminiEmbeddingData {
    pub embedding: Vec<f32>,
}

/// Represents a GeminiEmbedder struct that contains the URL and API key for making requests to the Gemini API.
#[derive(Debug)]
pub struct GeminiEmbedder {
    url: String,
    api_key: String,
    client: Client,
}

impl Default for GeminiEmbedder {
    fn default() -> Self {
        Self::new(None)
    }
}

impl GeminiEmbedder {
    pub fn new(api_key: Option<String>) -> Self {
        let api_key =
            api_key.unwrap_or_else(|| std::env::var("GEMINI_API_KEY").expect("API Key not set"));

        Self {
            url: "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent".to_string(),
            api_key,
            client: Client::new(),
        }
    }

    pub async fn embed(&self, text_batch: &[&str]) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        // Convert text_batch to the format expected by Gemini API
        let contents: Vec<serde_json::Value> = text_batch
            .iter()
            .map(|text| {
                json!({
                    "parts": [{"text": text}]
                })
            })
            .collect();

        let request_body = json!({
            "contents": contents,
            "embedding_config": {
                "task_type": "SEMANTIC_SIMILARITY"
            }
        });

        let response = self
            .client
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", &self.api_key)
            .json(&request_body)
            .send()
            .await?;

        let data = response.json::<GeminiEmbedResponse>().await?;
        let encodings = data
            .embeddings
            .iter()
            .map(|data| EmbeddingResult::DenseVector(data.embedding.clone()))
            .collect::<Vec<_>>();

        Ok(encodings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gemini_embed() {
        let gemini = GeminiEmbedder::default();
        let contents: Vec<serde_json::Value> = vec!["Hello world"]
            .iter()
            .map(|text| {
                json!({
                    "parts": [{"text": text}]
                })
            })
            .collect();

        let request_body = json!({
            "contents": contents,
            "embedding_config": {
                "task_type": "SEMANTIC_SIMILARITY"
            }
        });

        let response = gemini
            .client
            .post(&gemini.url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", &gemini.api_key)
            .json(&request_body)
            .send()
            .await
            .unwrap();

        let data = response.json::<GeminiEmbedResponse>().await.unwrap();
        println!("{:?}", data);
    }
}
