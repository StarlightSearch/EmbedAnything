use std::{collections::HashMap, fs};

use base64::Engine;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use crate::embeddings::embed::{EmbedData, EmbeddingResult};

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
    /// * `model` - A string slice that holds the model to be used for embedding. Find available models at <https://docs.cohere.com/docs/cohere-embed>
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

    fn load_image<T: AsRef<std::path::Path>>(&self, path: T) -> Result<String, anyhow::Error> {
        let img = image::ImageReader::open(path)?.decode()?;
        let img = img.to_rgb8();
        let mut buffer = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)?;
        let engine = base64::engine::general_purpose::STANDARD;
        let img = engine.encode(buffer);
        Ok(format!("data:image/png;base64,{}", img))
    }

    pub async fn embed(&self, text_batch: &[&str]) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
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
            .map(|embedding| EmbeddingResult::DenseVector(embedding.clone()))
            .collect::<Vec<_>>();

        Ok(encodings)
    }

    pub async fn embed_image(
        &self,
        image_path: impl AsRef<std::path::Path>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<EmbedData, anyhow::Error> {
        let img = self.load_image(image_path)?;

        let response = self
            .client
            .post(&self.url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&json!({
                "images": vec![img],
                "model": self.model,
                "input_type": "image"
            }))
            .send()
            .await?;

        let data = response.json::<CohereEmbedResponse>().await?;
        let encodings = data.embeddings;
        let embedding = encodings
            .iter()
            .map(|embedding| EmbeddingResult::DenseVector(embedding.clone()))
            .collect::<Vec<_>>();
        Ok(EmbedData::new(embedding[0].clone(), None, metadata))
    }

    pub async fn embed_image_batch(&self, image_paths: &[impl AsRef<std::path::Path>]) -> Result<Vec<EmbedData>, anyhow::Error> {
        let mut embeddings = Vec::new();
        for image_path in image_paths {
            let embedding = self.embed_image(image_path, None).await?;
            embeddings.push(embedding);
        }
        let embeddings = embeddings
        .iter()
        .zip(image_paths)
        .map(|(data, path)| {
            let mut metadata = HashMap::new();
            metadata.insert(
                "file_name".to_string(),
                fs::canonicalize(path)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string(),
            );

            let mut data = data.clone();
            data.text = Some(path.as_ref().to_str().unwrap().to_string());
            data.metadata = Some(metadata);
            data
        })
        .collect::<Vec<_>>();
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cohere_embed() {
        let cohere = CohereEmbedder::default();
        let text_batch = vec![
            "Once upon a time",
            "The quick brown fox jumps over the lazy dog",
        ];

        let embeddings = cohere.embed(&text_batch).await.unwrap();
        assert_eq!(embeddings.len(), 2);
    }
}
