use std::{collections::HashMap, fs};

use base64::Engine;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use crate::embeddings::embed::{EmbedData, EmbeddingResult};
use crate::embeddings::local::colpali::get_images_from_pdf;

/// Represents the response from the Cohere embedding API.
#[derive(Deserialize, Debug, Default)]
pub struct FloatResponse {
    /// A vector of embeddings, where each embedding is a vector of 32-bit floating point numbers.
    pub float: Vec<Vec<f32>>,
}

/// Represents the response from the Cohere embedding API.
#[derive(Deserialize, Debug, Default)]
pub struct CohereEmbedResponse {
    /// The ID of the request
    pub id: String,
    /// The embeddings data
    pub embeddings: FloatResponse,
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
            url: "https://api.cohere.com/v2/embed".to_string(),
            api_key,
            client: Client::new(),
        }
    }

    fn load_image<T: AsRef<std::path::Path>>(&self, path: T) -> Result<String, anyhow::Error> {
        let img = image::ImageReader::open(path)?.decode()?;
        let img = img.to_rgb8();
        let mut buffer = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut buffer),
            image::ImageFormat::Png,
        )?;
        let engine = base64::engine::general_purpose::STANDARD;
        let img = engine.encode(buffer);
        Ok(format!("data:image/png;base64,{}", img))
    }

    fn load_image_batch<T: AsRef<std::path::Path>>(
        &self,
        paths: &[T],
    ) -> Result<Vec<String>, anyhow::Error> {
        paths.iter().map(|path| self.load_image(path)).collect()
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
                "input_type": "search_document",
                "embedding_types": ["float"]
            }))
            .send()
            .await?;

        let data = match response.error_for_status() {
            Ok(resp) => resp.json::<CohereEmbedResponse>().await?,
            Err(e) => {
                println!("❌ API Error: {}", e);
                return Err(anyhow::anyhow!("API request failed: {}", e));
            }
        };
        let encodings = data.embeddings;

        let encodings = encodings
            .float
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

        let data = match response.error_for_status() {
            Ok(resp) => resp.json::<CohereEmbedResponse>().await?,
            Err(e) => {
                println!("❌ API Error: {}", e);
                return Err(anyhow::anyhow!("API request failed: {}", e));
            }
        };

        let encodings = data.embeddings;
        let embedding = encodings
            .float
            .iter()
            .map(|embedding| EmbeddingResult::DenseVector(embedding.clone()))
            .collect::<Vec<_>>();
        Ok(EmbedData::new(embedding[0].clone(), None, metadata))
    }

    pub async fn embed_image_batch(
        &self,
        image_paths: &[impl AsRef<std::path::Path>],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbedData>, anyhow::Error> {
        let mut embeddings = Vec::new();
        for image_path in image_paths.chunks(batch_size.unwrap_or(32)) {
            let imgs = self.load_image_batch(image_path)?;
            let response = self
                .client
                .post(&self.url)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(&json!({
                    "images": imgs,
                    "model": self.model,
                    "input_type": "image"
                }))
                .send()
                .await?;

            let data = match response.error_for_status() {
                Ok(resp) => resp.json::<CohereEmbedResponse>().await?,
                Err(e) => {
                    println!("❌ API Error: {}", e);
                    return Err(anyhow::anyhow!("API request failed: {}", e));
                }
            };
            let encodings = data.embeddings;
            let embedding = encodings.float.iter().cloned();
            embeddings.extend(embedding);
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

                EmbedData::new(
                    EmbeddingResult::DenseVector(data.clone()),
                    Some(path.as_ref().to_str().unwrap().to_string()),
                    Some(metadata),
                )
            })
            .collect::<Vec<_>>();
        Ok(embeddings)
    }

    pub async fn embed_pdf(
        &self,
        file_path: impl AsRef<std::path::Path>,
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbedData>, anyhow::Error> {
        let pages = get_images_from_pdf(&file_path)?;
        let mut embed_data = Vec::new();
        let batch_size = batch_size.unwrap_or(8);

        let pages_base64 = pages
            .iter()
            .map(|page| -> Result<String, anyhow::Error> {
                let img = page.to_rgb8();
                let mut buffer = Vec::new();
                img.write_to(
                    &mut std::io::Cursor::new(&mut buffer),
                    image::ImageFormat::Png,
                )?;
                let engine = base64::engine::general_purpose::STANDARD;
                let img = engine.encode(buffer);
                Ok(format!("data:image/png;base64,{}", img))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;

        for (index, batch) in pages_base64.chunks(batch_size).enumerate() {
            let start_page = index * batch_size + 1;
            let end_page = start_page + batch.len();
            let page_numbers = (start_page..=end_page).collect::<Vec<_>>();

            let response = self
                .client
                .post(&self.url)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(&json!({
                    "images": batch,
                    "model": self.model,
                    "input_type": "image",
                    "embedding_types": ["float"]
                }))
                .send()
                .await?;

            let data = match response.error_for_status() {
                Ok(resp) => resp.json::<CohereEmbedResponse>().await?,
                Err(e) => {
                    println!("❌ API Error: {}", e);
                    return Err(anyhow::anyhow!("API request failed: {}", e));
                }
            };
            let encodings = data.embeddings;
            let image_embeddings = encodings
                .float
                .iter()
                .map(|embedding| EmbeddingResult::DenseVector(embedding.clone()));

            // zip the embeddings with the page numbers
            let embed_data_batch = image_embeddings
                .zip(page_numbers.into_iter())
                .zip(batch.iter())
                .map(|((embedding, page_number), page_image)| {
                    let mut metadata = HashMap::new();

                    metadata.insert("page_number".to_string(), page_number.to_string());
                    metadata.insert(
                        "file_path".to_string(),
                        file_path.as_ref().to_str().unwrap_or("").to_string(),
                    );
                    metadata.insert("image".to_string(), page_image.clone());
                    EmbedData::new(embedding, None, Some(metadata))
                });
            embed_data.extend(embed_data_batch);
        }
        Ok(embed_data)
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

    #[tokio::test]
    async fn test_cohere_embed_pdf() {
        let cohere = CohereEmbedder::new("embed-v4.0".to_string(), None);
        let file_path = "../test_files/colpali.pdf";
        let embeddings = cohere.embed_pdf(file_path, None).await.unwrap();
        assert_eq!(embeddings.len(), 26);
    }
}
