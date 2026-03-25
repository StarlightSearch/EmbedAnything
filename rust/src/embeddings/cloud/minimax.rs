use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use crate::embeddings::embed::EmbeddingResult;

/// Represents the base response status from the MiniMax API.
#[derive(Deserialize, Debug, Default)]
pub struct MiniMaxBaseResp {
    pub status_code: i32,
    pub status_msg: String,
}

/// Represents the response from the MiniMax embedding API.
#[derive(Deserialize, Debug, Default)]
pub struct MiniMaxEmbedResponse {
    pub vectors: Option<Vec<Vec<f32>>>,
    pub total_tokens: Option<usize>,
    pub base_resp: MiniMaxBaseResp,
}

/// Represents a MiniMaxEmbedder struct that contains the URL, model, and API key
/// for making requests to the MiniMax Embedding API.
///
/// MiniMax uses a native embedding API format (not OpenAI-compatible):
/// - Request: `{"model": "embo-01", "texts": [...], "type": "db"|"query"}`
/// - Response: `{"vectors": [[...]], "total_tokens": N, "base_resp": {...}}`
///
/// The `embo-01` model produces 1536-dimensional embeddings.
#[derive(Debug)]
pub struct MiniMaxEmbedder {
    url: String,
    model: String,
    api_key: String,
    client: Client,
}

impl Default for MiniMaxEmbedder {
    fn default() -> Self {
        Self::new("embo-01".to_string(), None)
    }
}

impl MiniMaxEmbedder {
    pub fn new(model: String, api_key: Option<String>) -> Self {
        let api_key = api_key
            .unwrap_or_else(|| std::env::var("MINIMAX_API_KEY").expect("MINIMAX_API_KEY not set"));

        Self {
            model,
            url: "https://api.minimax.io/v1/embeddings".to_string(),
            api_key,
            client: Client::new(),
        }
    }

    pub async fn embed(&self, text_batch: &[&str]) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let texts: Vec<&str> = text_batch.to_vec();

        let response = self
            .client
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&json!({
                "model": self.model,
                "texts": texts,
                "type": "db"
            }))
            .send()
            .await?;

        let data = response.json::<MiniMaxEmbedResponse>().await?;

        if data.base_resp.status_code != 0 {
            return Err(anyhow::anyhow!(
                "MiniMax API error: {}",
                data.base_resp.status_msg
            ));
        }

        let vectors = data.vectors.ok_or_else(|| {
            anyhow::anyhow!("MiniMax API returned no vectors")
        })?;

        let encodings = vectors
            .into_iter()
            .map(EmbeddingResult::DenseVector)
            .collect::<Vec<_>>();

        Ok(encodings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_minimax_embed() {
        let minimax = MiniMaxEmbedder::default();
        let result = minimax.embed(&["Hello world"]).await.unwrap();
        assert!(!result.is_empty());
        match &result[0] {
            EmbeddingResult::DenseVector(v) => assert_eq!(v.len(), 1536),
            _ => panic!("Expected DenseVector"),
        }
    }
}
