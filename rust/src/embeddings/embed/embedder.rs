use std::path::PathBuf;
use std::sync::Arc;

use crate::config::{ImageEmbedConfig, TextEmbedConfig};
use crate::Dtype;
use crate::embeddings::local::pooling::Pooling;
use anyhow::{anyhow, Result};
use crate::hf_hub_utils::{build_client, download_file, model_repo};

use super::text::{TextEmbed, TextEmbedder};
use super::types::{EmbedData, EmbeddingResult};
use super::vision::{EmbedImage, VisionEmbedder};

pub enum Embedder {
    Text(TextEmbedder),
    Vision(Box<VisionEmbedder>),
}

impl Embedder {
    pub async fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
        late_chunking: Option<bool>,
    ) -> Result<Vec<EmbeddingResult>> {
        match self {
            Self::Text(embedder) => embedder.embed(text_batch, batch_size, late_chunking).await,
            Self::Vision(embedder) => embedder.embed(text_batch, batch_size).await,
        }
    }

    pub fn from_pretrained_hf(
        model_id: &str,
        revision: Option<&str>,
        token: Option<&str>,
        dtype: Option<Dtype>,
        pooling: Option<Pooling>,
    ) -> Result<Self> {
        let client = build_client(token)?;
        let repo = model_repo(&client, model_id);
        let config_filename = download_file(&repo, "config.json", revision)?;
        let config = std::fs::read_to_string(config_filename)?;
        let config: serde_json::Value = serde_json::from_str(&config)?;

        let architecture = config["architectures"]
            .as_array()
            .ok_or(anyhow!("Architecture not found"))?
            .first()
            .ok_or(anyhow!("Architecture not found"))?
            .as_str()
            .ok_or(anyhow!("Architecture not found"))?;
        match architecture {
            "CLIPModel" | "SiglipModel" => Ok(Self::Vision(Box::new(
                VisionEmbedder::from_pretrained_hf(architecture, model_id, revision, token)?,
            ))),
            "ColPali" => Ok(Self::Vision(Box::new(VisionEmbedder::from_pretrained_hf(
                architecture,
                model_id,
                revision,
                token,
            )?))),
            "Dinov2Model" => Ok(Self::Vision(Box::new(VisionEmbedder::from_pretrained_hf(
                architecture,
                model_id,
                revision,
                token,
            )?))),
            "BertModel" => Ok(Self::Text(TextEmbedder::from_pretrained_hf(
                architecture,
                model_id,
                revision,
                token,
                dtype,
                pooling,
            )?)),
            "JinaBertForMaskedLM" => Ok(Self::Text(TextEmbedder::from_pretrained_hf(
                architecture,
                model_id,
                revision,
                token,
                dtype,
                pooling,
            )?)),
            "StaticModel" => Ok(Self::Text(TextEmbedder::from_pretrained_hf(
                architecture,
                model_id,
                revision,
                token,
                dtype,
                pooling,
            )?)),
            "BertForMaskedLM" => Ok(Self::Text(TextEmbedder::from_pretrained_hf(
                architecture,
                model_id,
                revision,
                token,
                dtype,
                pooling,
            )?)),
            "ModernBertForMaskedLM" => Ok(Self::Text(TextEmbedder::from_pretrained_hf(
                architecture,
                model_id,
                revision,
                token,
                dtype,
                pooling,
            )?)),
            "Qwen3ForCausalLM" => Ok(Self::Text(TextEmbedder::from_pretrained_hf(
                architecture,
                model_id,
                revision,
                token,
                dtype,
                pooling,
            )?)),
            "Gemma3TextModel" => Ok(Self::Text(TextEmbedder::from_pretrained_hf(
                architecture,
                model_id,
                revision,
                token,
                dtype,
                pooling,
            )?)),
            _ => Err(anyhow!("Model not supported")),
        }
    }

    pub fn from_pretrained_cloud(
        model: &str,
        model_id: &str,
        api_key: Option<String>,
    ) -> Result<Self> {
        match model {
            "openai" | "OpenAI" => Ok(Self::Text(TextEmbedder::from_pretrained_cloud(
                model, model_id, api_key,
            )?)),
            "cohere" | "Cohere" => Ok(Self::Text(TextEmbedder::from_pretrained_cloud(
                model, model_id, api_key,
            )?)),
            "gemini" | "Gemini" => Ok(Self::Text(TextEmbedder::from_pretrained_cloud(
                model, model_id, api_key,
            )?)),
            "cohere-vision" | "CohereVision" => Ok(Self::Vision(Box::new(
                VisionEmbedder::from_pretrained_cloud(model, model_id, api_key)?,
            ))),
            _ => Err(anyhow!("Model not supported")),
        }
    }

    #[cfg(not(feature = "ort"))]
    pub fn from_pretrained_onnx(
        _model_architecture: &str,
        _model_name: Option<crate::embeddings::local::text_embedding::ONNXModel>,
        _revision: Option<&str>,
        _model_id: Option<&str>,
        _dtype: Option<Dtype>,
        _path_in_repo: Option<&str>,
    ) -> Result<Self> {
        Err(anyhow!(
            "The 'ort' feature must be enabled to use the 'from_pretrained_ort' function."
        ))
    }

    #[cfg(feature = "ort")]
    pub fn from_pretrained_onnx(
        model_architecture: &str,
        model_name: Option<crate::embeddings::local::text_embedding::ONNXModel>,
        revision: Option<&str>,
        model_id: Option<&str>,
        dtype: Option<Dtype>,
        path_in_repo: Option<&str>,
    ) -> Result<Self> {
        Ok(Self::Text(TextEmbedder::from_pretrained_ort(
            model_architecture,
            model_name,
            revision,
            model_id,
            dtype,
            path_in_repo,
        )?))
    }

    pub async fn embed_directory_stream(
        self: &Arc<Self>,
        directory: PathBuf,
        extensions: Option<Vec<String>>,
        config: Option<&TextEmbedConfig>,
        adapter: Option<Box<dyn FnMut(Vec<EmbedData>) + Send + Sync>>,
    ) -> Result<Option<Vec<EmbedData>>> {
        crate::embed_directory_stream(directory, self, extensions, config, adapter).await
    }

    pub async fn embed_image_directory(
        self: &Arc<Self>,
        directory: PathBuf,
        config: Option<&ImageEmbedConfig>,
        adapter: Option<Box<dyn FnMut(Vec<EmbedData>) + Send + Sync>>,
    ) -> Result<Option<Vec<EmbedData>>> {
        crate::embed_image_directory(directory, self, config, adapter).await
    }

    pub async fn embed_file<T: AsRef<std::path::Path>>(
        &self,
        file_path: T,
        config: Option<&TextEmbedConfig>,
        adapter: Option<Box<dyn FnOnce(Vec<EmbedData>) + Send + Sync>>,
    ) -> Result<Option<Vec<EmbedData>>> {
        crate::embed_file(file_path, self, config, adapter).await
    }

    pub async fn embed_webpage(
        &self,
        url: String,
        config: Option<&TextEmbedConfig>,
        adapter: Option<Box<dyn FnOnce(Vec<EmbedData>) + Send + Sync>>,
    ) -> Result<Option<Vec<EmbedData>>> {
        crate::embed_webpage(url, self, config, adapter).await
    }

    pub async fn embed_files_batch(
        self: &Arc<Self>,
        file_paths: impl IntoIterator<Item = impl AsRef<std::path::Path>>,
        config: Option<&TextEmbedConfig>,
        adapter: Option<Box<dyn FnMut(Vec<EmbedData>) + Send + Sync>>,
    ) -> Result<Option<Vec<EmbedData>>> {
        crate::embed_files_batch(file_paths, self, config, adapter).await
    }

    pub async fn embed_query(
        self: &Arc<Self>,
        query: &[&str],
        config: Option<&TextEmbedConfig>,
    ) -> Result<Vec<EmbedData>> {
        crate::embed_query(query, self, config).await
    }
}

impl From<VisionEmbedder> for Embedder {
    fn from(value: VisionEmbedder) -> Self {
        Embedder::Vision(Box::new(value))
    }
}

impl From<Embedder> for VisionEmbedder {
    fn from(value: Embedder) -> Self {
        match value {
            Embedder::Vision(value) => *value,
            _ => panic!("Invalid embedder type"),
        }
    }
}

impl From<Embedder> for TextEmbedder {
    fn from(value: Embedder) -> Self {
        match value {
            Embedder::Text(value) => value,
            _ => panic!("Invalid embedder type"),
        }
    }
}

impl EmbedImage for Embedder {
    async fn embed_image<T: AsRef<std::path::Path>>(
        &self,
        image_path: T,
        metadata: Option<std::collections::HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData> {
        match self {
            Self::Vision(embedder) => embedder.embed_image(image_path, metadata).await,
            _ => Err(anyhow!("Model not supported for vision embedding")),
        }
    }

    async fn embed_image_batch<T: AsRef<std::path::Path>>(
        &self,
        image_paths: &[T],
        batch_size: Option<usize>,
    ) -> anyhow::Result<Vec<EmbedData>> {
        match self {
            Self::Vision(embedder) => embedder.embed_image_batch(image_paths, batch_size).await,
            _ => Err(anyhow!("Model not supported for vision embedding")),
        }
    }

    async fn embed_pdf<T: AsRef<std::path::Path>>(
        &self,
        pdf_path: T,
        batch_size: Option<usize>,
    ) -> anyhow::Result<Vec<EmbedData>> {
        match self {
            Self::Vision(embedder) => embedder.embed_pdf(pdf_path, batch_size).await,
            _ => Err(anyhow!("Model not supported for PDF embedding")),
        }
    }
}
