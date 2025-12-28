use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;

use anyhow::{anyhow, Result};

use crate::embeddings::cloud::cohere::CohereEmbedder;
use crate::embeddings::local::clip::ClipEmbedder;
use crate::embeddings::local::colpali::{ColPaliEmbed, ColPaliEmbedder};
use crate::embeddings::local::vision_encoder::VisionEncoderEmbedder;

use super::text::TextEmbed;
use super::types::{EmbedData, EmbeddingResult};

pub enum VisionEmbedder {
    Clip(Box<ClipEmbedder>),
    VisionEncoder(Box<VisionEncoderEmbedder>),
    ColPali(Box<dyn ColPaliEmbed + Send + Sync>),
    Cohere(CohereEmbedder),
}

impl VisionEmbedder {
    pub fn from_pretrained_hf(
        architecture: &str,
        model_id: &str,
        revision: Option<&str>,
        token: Option<&str>,
    ) -> Result<Self> {
        match architecture {
            "CLIPModel" | "SiglipModel" => Ok(Self::Clip(Box::new(ClipEmbedder::new(
                model_id.to_string(),
                revision,
                token,
            )?))),
            "Dinov2Model" => Ok(Self::VisionEncoder(Box::new(VisionEncoderEmbedder::new(
                model_id, revision, token,
            )?))),
            "ColPali" => Ok(Self::ColPali(Box::new(ColPaliEmbedder::new(
                model_id, revision,
            )?))),
            _ => Err(anyhow!("Model not supported")),
        }
    }

    pub fn from_pretrained_cloud(
        model: &str,
        model_id: &str,
        api_key: Option<String>,
    ) -> Result<Self> {
        match model {
            "cohere-vision" | "CohereVision" => Ok(Self::Cohere(CohereEmbedder::new(
                model_id.to_string(),
                api_key,
            ))),
            _ => Err(anyhow!("Model not supported")),
        }
    }
}

impl TextEmbed for VisionEmbedder {
    async fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>> {
        match self {
            Self::Clip(embedder) => embedder.embed(text_batch, batch_size),
            Self::VisionEncoder(_) => Err(anyhow!("Model not supported for text embedding")),
            Self::ColPali(embedder) => embedder.embed(text_batch, batch_size),
            Self::Cohere(embedder) => embedder.embed(text_batch).await,
        }
    }
}

pub trait EmbedImage {
    fn embed_image<T: AsRef<std::path::Path>>(
        &self,
        image_path: T,
        metadata: Option<HashMap<String, String>>,
    ) -> impl Future<Output = anyhow::Result<EmbedData>>;

    fn embed_image_batch<T: AsRef<std::path::Path>>(
        &self,
        image_paths: &[T],
        batch_size: Option<usize>,
    ) -> impl Future<Output = anyhow::Result<Vec<EmbedData>>>;

    fn embed_pdf<T: AsRef<std::path::Path>>(
        &self,
        pdf_path: T,
        batch_size: Option<usize>,
    ) -> impl Future<Output = anyhow::Result<Vec<EmbedData>>>;
}

impl EmbedImage for VisionEmbedder {
    async fn embed_image<T: AsRef<std::path::Path>>(
        &self,
        image_path: T,
        metadata: Option<HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData> {
        match self {
            Self::Clip(embedder) => embedder.embed_image(image_path, metadata).await,
            Self::ColPali(embedder) => {
                embedder.embed_image(PathBuf::from(image_path.as_ref()), metadata)
            }
            Self::Cohere(embedder) => embedder.embed_image(image_path, metadata).await,
            Self::VisionEncoder(embedder) => embedder.embed_image(image_path, metadata).await,
        }
    }

    async fn embed_image_batch<T: AsRef<std::path::Path>>(
        &self,
        image_paths: &[T],
        batch_size: Option<usize>,
    ) -> anyhow::Result<Vec<EmbedData>> {
        match self {
            Self::Clip(embedder) => embedder.embed_image_batch(image_paths, batch_size).await,
            Self::ColPali(embedder) => embedder.embed_image_batch(
                &image_paths
                    .iter()
                    .map(|p| PathBuf::from(p.as_ref()))
                    .collect::<Vec<_>>(),
            ),
            Self::Cohere(embedder) => embedder.embed_image_batch(image_paths, batch_size).await,
            Self::VisionEncoder(embedder) => {
                embedder.embed_image_batch(image_paths, batch_size).await
            }
        }
    }

    async fn embed_pdf<T: AsRef<std::path::Path>>(
        &self,
        pdf_path: T,
        batch_size: Option<usize>,
    ) -> anyhow::Result<Vec<EmbedData>> {
        match self {
            Self::Cohere(embedder) => embedder.embed_pdf(pdf_path, batch_size).await,
            _ => Err(anyhow!("Model not supported for PDF embedding")),
        }
    }
}
