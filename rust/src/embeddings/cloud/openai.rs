use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use crate::embeddings::embed::{EmbeddingResult, NumericalType};

#[derive(Deserialize, Debug, Default)]
pub struct OpenAIEmbedResponse<F: NumericalType> {
    pub data: Vec<EmbeddingData<F>>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Deserialize, Debug, Default)]
pub struct EmbeddingData<F: NumericalType> {
    pub embedding: Vec<F>,
    pub index: usize,
}

#[derive(Deserialize, Debug, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

/// Represents an OpenAIEmbeder struct that contains the URL and API key for making requests to the OpenAI API.
#[derive(Debug)]
pub struct OpenAIEmbedder {
    url: String,
    model: String,
    api_key: String,
    client: Client,
}

impl Default for OpenAIEmbedder {
    fn default() -> Self {
        Self::new("text-embedding-3-small".to_string(), None)
    }
}

impl OpenAIEmbedder {
    pub fn new(model: String, api_key: Option<String>) -> Self {
        let api_key =
            api_key.unwrap_or_else(|| std::env::var("OPENAI_API_KEY").expect("API Key not set"));

        Self {
            model,
            url: "https://api.openai.com/v1/embeddings".to_string(),
            api_key,
            client: Client::new(),
        }
    }

    pub async fn embed<F: NumericalType + for<'de> Deserialize<'de>>(
        &self,
        text_batch: &[String],
    ) -> Result<Vec<EmbeddingResult<F>>, anyhow::Error> {
        let response = self
            .client
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&json!({
                "input": text_batch,
                "model": self.model,
                "encoding_format": "float"
            }))
            .send()
            .await?;
        let data = response.json::<OpenAIEmbedResponse<F>>().await?;

        println!("{:?}", data.usage);

        let encodings = data
            .data
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
    async fn test_openai_embed() {
        let openai = OpenAIEmbedder::default();
        let response = openai
            .client
            .post(&openai.url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", openai.api_key))
            .json(&json!({
                "input": vec!["Hello world"],
                "model": openai.model,
                "encoding_format": "float"
            }))
            .send()
            .await
            .unwrap();
        // println!("{}", response.text().await.unwrap());
        let data = response.json::<OpenAIEmbedResponse<f32>>().await.unwrap();
        println!("{:?}", data);
    }
}
