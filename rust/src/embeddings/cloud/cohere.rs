use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use crate::embeddings::embed::EmbeddingResult;

/// Represents the response from the Cohere embedding API.
#[derive(Deserialize, Debug, Default)]
pub struct CohereEmbedResponse {
    /// A vector of embeddings, where each embedding is a vector of 32-bit floating point numbers.
    pub embeddings: Vec<Vec<f32>>,
}

/// Represents a CohereEmbeder struct that contains the URL and API key for making requests to the Cohere API.
#[derive(Debug)]
pub struct CohereEmbedder {
    /// The URL of the Cohere API endpoint.
    url: String,
    /// The model to be used for embedding.
    model: String,
    /// The API key for authenticating requests to the Cohere API.
    api_key: String,
    /// The HTTP client for making requests.
    client: Client,
}

impl Default for CohereEmbedder {
    /// Creates a default instance of `CohereEmbeder` with the model set to "embed-english-v3.0" and no API key.
    fn default() -> Self {
        Self::new("embed-english-v3.0".to_string(), None)
    }
}

impl CohereEmbedder {
    /// Creates a new instance of `CohereEmbeder` with the specified model and API key.
    ///
    /// # Arguments
    ///
    /// * `model` - A string slice that holds the model to be used for embedding. Find available models at https://docs.cohere.com/docs/cohere-embed
    /// * `api_key` - An optional string slice that holds the API key for authenticating requests to the Cohere API.
    ///
    /// # Returns
    ///
    /// A new instance of `CohereEmbeder`.
    pub fn new(model: String, api_key: Option<String>) -> Self {
        let api_key =
            api_key.unwrap_or_else(|| std::env::var("CO_API_KEY").expect("API key not set"));

        Self {
            model,
            url: "https://api.cohere.com/v1/embed".to_string(),
            api_key,
            client: Client::new(),
        }
    }

    pub async fn embed(
        &self,
        text_batch: &[String],
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
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
            .await?;

        let data = response.json::<CohereEmbedResponse>().await?;
        let encodings = data.embeddings;

        let encodings = encodings
            .iter()
            .map(|embedding| EmbeddingResult::Dense(embedding.clone()))
            .collect::<Vec<_>>();

        Ok(encodings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cohere_embed() {
        let cohere = CohereEmbedder::default();
        let text_batch = vec![
            "Once upon a time".to_string(),
            "The quick brown fox jumps over the lazy dog".to_string(),
        ];

        let embeddings = cohere.embed(&text_batch).await.unwrap();
        assert_eq!(embeddings.len(), 2);
    }
}
