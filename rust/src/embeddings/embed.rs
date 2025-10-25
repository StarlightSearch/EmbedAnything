use crate::config::{ImageEmbedConfig, TextEmbedConfig};
use crate::embeddings::local::vision_encoder::VisionEncoderEmbedder;
use crate::file_processor::audio::audio_processor::Segment;
use crate::Dtype;

use super::cloud::cohere::CohereEmbedder;
use super::cloud::gemini::GeminiEmbedder;
use super::cloud::openai::OpenAIEmbedder;
use super::local::bert::{BertEmbed, BertEmbedder, SparseBertEmbedder};

use super::local::clip::ClipEmbedder;
use super::local::colpali::{ColPaliEmbed, ColPaliEmbedder};
use super::local::jina::{JinaEmbed, JinaEmbedder};
use super::local::model2vec::Model2VecEmbedder;
use super::local::modernbert::ModernBertEmbedder;
use super::local::qwen3::{Qwen3Embed, Qwen3Embedder};
use super::local::text_embedding::ONNXModel;
use anyhow::anyhow;
use anyhow::Result;
use hf_hub::Repo;
use serde::Deserialize;
use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "ort")]
use {
    super::local::colbert::OrtColbertEmbedder,
    super::local::ort_bert::{OrtBertEmbedder, OrtSparseBertEmbedder},
    super::local::ort_jina::OrtJinaEmbedder,
};

#[derive(Deserialize, Debug, Clone)]
pub enum EmbeddingResult {
    DenseVector(Vec<f32>),
    MultiVector(Vec<Vec<f32>>),
}

impl From<Vec<f32>> for EmbeddingResult {
    fn from(value: Vec<f32>) -> Self {
        EmbeddingResult::DenseVector(value)
    }
}

impl From<Vec<Vec<f32>>> for EmbeddingResult {
    fn from(value: Vec<Vec<f32>>) -> Self {
        EmbeddingResult::MultiVector(value)
    }
}

impl EmbeddingResult {
    pub fn to_dense(&self) -> Result<Vec<f32>, anyhow::Error> {
        match self {
            EmbeddingResult::DenseVector(x) => Ok(x.to_vec()),
            EmbeddingResult::MultiVector(_) => Err(anyhow!(
                "Multi-vector Embedding are not supported for this operation"
            )),
        }
    }

    pub fn to_multi_vector(&self) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        match self {
            EmbeddingResult::MultiVector(x) => Ok(x.to_vec()),
            EmbeddingResult::DenseVector(_) => Err(anyhow!(
                "Dense Embedding are not supported for this operation"
            )),
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct EmbedData {
    pub embedding: EmbeddingResult,
    pub text: Option<String>,
    pub metadata: Option<HashMap<String, String>>,
}

impl EmbedData {
    pub fn new(
        embedding: EmbeddingResult,
        text: Option<String>,
        metadata: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            embedding,
            text,
            metadata,
        }
    }

    pub fn __str__(&self) -> String {
        format!(
            "EmbedData(embedding: {:?}, text: {:?}, metadata: {:?})",
            self.embedding,
            self.text,
            self.metadata.clone()
        )
    }
}

pub trait AudioDecoder {
    fn decode_audio(&mut self, audio_file: &std::path::Path)
        -> Result<Vec<Segment>, anyhow::Error>;
}

pub enum TextEmbedder {
    OpenAI(OpenAIEmbedder),
    Cohere(CohereEmbedder),
    Gemini(GeminiEmbedder),
    Jina(Box<dyn JinaEmbed + Send + Sync>),
    Model2Vec(Box<Model2VecEmbedder>),
    Bert(Box<dyn BertEmbed + Send + Sync>),
    Qwen3(Box<dyn Qwen3Embed + Send + Sync>),
    ColBert(Box<dyn BertEmbed + Send + Sync>),
    ModernBert(Box<dyn BertEmbed + Send + Sync>),
}

impl TextEmbedder {
    pub async fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
        late_chunking: Option<bool>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        match self {
            TextEmbedder::OpenAI(embedder) => embedder.embed(text_batch).await,
            TextEmbedder::Cohere(embedder) => embedder.embed(text_batch).await,
            TextEmbedder::Gemini(embedder) => embedder.embed(text_batch).await,
            TextEmbedder::Model2Vec(embedder) => embedder.embed(text_batch, batch_size),
            TextEmbedder::Jina(embedder) => embedder.embed(text_batch, batch_size, late_chunking),
            TextEmbedder::Bert(embedder) => embedder.embed(text_batch, batch_size, late_chunking),
            TextEmbedder::Qwen3(embedder) => embedder.embed(text_batch, batch_size, late_chunking),
            TextEmbedder::ColBert(embedder) => {
                embedder.embed(text_batch, batch_size, late_chunking)
            }
            TextEmbedder::ModernBert(embedder) => {
                embedder.embed(text_batch, batch_size, late_chunking)
            }
        }
    }

    pub fn from_pretrained_hf(
        architecture: &str,
        model_id: &str,
        revision: Option<&str>,
        token: Option<&str>,
        dtype: Option<Dtype>,
    ) -> Result<Self, anyhow::Error> {
        match architecture {
            "JinaBertForMaskedLM" => Ok(Self::Jina(Box::new(JinaEmbedder::new(
                model_id, revision, token,
            )?))),

            "BertModel" => Ok(Self::Bert(Box::new(BertEmbedder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
                token,
            )?))),
            "BertForMaskedLM" => Ok(Self::Bert(Box::new(SparseBertEmbedder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
                token,
            )?))),
            "StaticModel" => Ok(Self::Model2Vec(Box::new(Model2VecEmbedder::new(
                model_id, token, None,
            )?))),

            "ModernBertForMaskedLM" => Ok(Self::ModernBert(Box::new(ModernBertEmbedder::new(
                model_id.to_string(),
                revision.map(|s| s.to_string()),
                token,
                dtype,
            )?))),
            "Qwen3ForCausalLM" => Ok(Self::Qwen3(Box::new(Qwen3Embedder::new(
                model_id,
                revision.map(|s| s.to_string()),
                token,
                dtype,
            )?))),
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }

    #[cfg(feature = "ort")]
    pub fn from_pretrained_ort(
        model_architecture: &str,
        model_name: Option<ONNXModel>,
        revision: Option<&str>,
        model_id: Option<&str>,
        dtype: Option<Dtype>,
        path_in_repo: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        if model_name.is_some() {
            match model_architecture {
                "Bert" | "bert" => Ok(Self::Bert(Box::new(OrtBertEmbedder::new(
                    model_name,
                    model_id,
                    revision,
                    dtype,
                    path_in_repo,
                )?))),
                "sparse-bert" | "SparseBert" | "SPARSE-BERT" => Ok(Self::Bert(Box::new(
                    OrtSparseBertEmbedder::new(model_name, model_id, revision, path_in_repo)?,
                ))),
                "jina" | "Jina" => Ok(Self::Jina(Box::new(OrtJinaEmbedder::new(
                    model_name,
                    model_id,
                    revision,
                    dtype,
                    path_in_repo,
                )?))),

                _ => Err(anyhow::anyhow!("Model not supported")),
            }
        } else if model_id.is_some() {
            match model_architecture {
                "colbert" | "Colbert" | "COLBERT" => Ok(Self::ColBert(Box::new(
                    OrtColbertEmbedder::new(model_id, revision, path_in_repo)?,
                ))),
                "bert" | "Bert" => Ok(Self::Bert(Box::new(OrtBertEmbedder::new(
                    None,
                    model_id,
                    revision,
                    None,
                    path_in_repo,
                )?))),
                "jina" | "Jina" => Ok(Self::Jina(Box::new(OrtJinaEmbedder::new(
                    None,
                    model_id,
                    revision,
                    dtype,
                    path_in_repo,
                )?))),
                _ => Err(anyhow::anyhow!("Model not supported")),
            }
        } else {
            Err(anyhow::anyhow!(
                "Please provide either model_name or model_id"
            ))
        }
    }

    /// Creates a new instance of a cloud api based `Embedder` with the specified model and API key.
    ///
    /// # Arguments
    ///
    /// * `model` - A string holds the model to be used for embedding. Choose from
    ///      - "openai"
    ///      - "cohere"
    ///
    /// * `model_id` - A string holds the model ID for the model to be used for embedding.
    ///     - For OpenAI, find available models at <https://platform.openai.com/docs/guides/embeddings/embedding-models>
    ///     - For Cohere, find available models at <https://docs.cohere.com/docs/cohere-embed>
    /// * `api_key` - An optional string holds the API key for authenticating requests to the Cohere API. If not provided, it is taken from the environment variable
    ///     - For OpenAI, create environment variable `OPENAI_API_KEY`
    ///     - For Cohere, create environment variable `CO_API_KEY`
    ///
    /// # Returns
    ///
    /// A new instance of `Embedder`.
    pub fn from_pretrained_cloud(
        model: &str,
        model_id: &str,
        api_key: Option<String>,
    ) -> Result<Self, anyhow::Error> {
        match model {
            "openai" | "OpenAI" => Ok(Self::OpenAI(OpenAIEmbedder::new(
                model_id.to_string(),
                api_key,
            ))),
            "cohere" | "Cohere" => Ok(Self::Cohere(CohereEmbedder::new(
                model_id.to_string(),
                api_key,
            ))),
            "gemini" | "Gemini" => Ok(Self::Gemini(GeminiEmbedder::new(api_key))),
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }
}

pub enum VisionEmbedder {
    Clip(Box<ClipEmbedder>),
    VisionEncoder(Box<VisionEncoderEmbedder>),
    ColPali(Box<dyn ColPaliEmbed + Send + Sync>),
    Cohere(CohereEmbedder),
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

impl VisionEmbedder {
    pub fn from_pretrained_hf(
            architecture: &str,
        model_id: &str,
        revision: Option<&str>,
        token: Option<&str>,
    ) -> Result<Self, anyhow::Error> {   
        match architecture {
            "CLIPModel" | "SiglipModel"  => Ok(Self::Clip(Box::new(ClipEmbedder::new(
                model_id.to_string(),
                revision,
                token,
            )?))),
            "Dinov2Model" => Ok(Self::VisionEncoder(Box::new(VisionEncoderEmbedder::new(
                model_id,
                revision,
                token,
            )?))),
            "ColPali" => Ok(Self::ColPali(Box::new(ColPaliEmbedder::new(
                model_id, revision,
            )?))),
   
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }

    pub fn from_pretrained_cloud(
        model: &str,
        model_id: &str,
        api_key: Option<String>,
    ) -> Result<Self, anyhow::Error> {
        match model {
            "cohere-vision" | "CohereVision" => Ok(Self::Cohere(CohereEmbedder::new(
                model_id.to_string(),
                api_key,
            ))),
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }
}

/// This is a builder for the Embedder. You can use it to build an Embedder from either HF or ONNX models.
/// You need to provide atleast the `model_id` or the `onnx_model_id`.
/// ## Example
/// ### Text Embedding Model
/// ```rust
/// use embed_anything::embeddings::embed::EmbedderBuilder;
/// let embedder = EmbedderBuilder::new()
///     .model_architecture("bert")
///     .model_id(Some("sentence-transformers/all-MiniLM-L6-v2"))
///     .revision(None)
///     .from_pretrained_hf()
///     .unwrap();
/// ```
/// ### Vision Embedding Model
/// ```rust
/// use embed_anything::embeddings::embed::EmbedderBuilder;
/// let embedder = EmbedderBuilder::new()
///     .model_architecture("clip")
///     .model_id(Some("openai/clip-vit-base-patch32"))
///     .revision(None)
///     .from_pretrained_hf()
///     .unwrap();
/// ```
///
/// ### Cloud Embedding Model
/// ```rust
/// use embed_anything::embeddings::embed::EmbedderBuilder;
/// let embedder = EmbedderBuilder::new()
///     .model_architecture("openai")
///     .model_id(Some("text-embedding-3-small"))
///     .api_key(Some("your_api_key"))
///     .from_pretrained_cloud()
///     .unwrap();
/// ```
///
/// ### ONNX Embedding Model
/// ```rust,ignore
/// use embed_anything::embeddings::embed::EmbedderBuilder;
/// use embed_anything::embeddings::local::text_embedding::ONNXModel;
/// use embed_anything::Dtype;
/// let embedder = EmbedderBuilder::new()
///     .model_architecture("bert")
///     .onnx_model_id(Some(ONNXModel::AllMiniLML12V2))
///     .revision(None)
///     .dtype(Some(Dtype::F32))
///     .from_pretrained_onnx()
///     .unwrap();
/// ```
#[derive(Default)]
pub struct EmbedderBuilder {
    // The model architecture that you want to use. For cloud models, it is the provider name.
    model_architecture: String,
    // Either HF Model ID or the Cloud Model that youu want to use
    model_id: Option<String>,
    revision: Option<String>,
    // The Hugging Face token
    token: Option<String>,
    // The API key for the cloud model
    api_key: Option<String>,
    path_in_repo: Option<String>,
    // The ONNX Model ID that you want to use
    onnx_model_id: Option<ONNXModel>,
    dtype: Option<Dtype>,
}

impl EmbedderBuilder {
    pub fn new() -> Self {
        Self {
            model_architecture: String::new(),
            model_id: None,
            revision: None,
            token: None,
            api_key: None,
            path_in_repo: None,
            onnx_model_id: None,
            dtype: None,
        }
    }

    // The model architecture that you want to use. For cloud models, it is the provider name.
    pub fn model_architecture(mut self, model_architecture: &str) -> Self {
        self.model_architecture = model_architecture.to_string();
        self
    }

    // The model ID that you want to use. For HF models, it is the model name on Hugging Face Hub.
    pub fn model_id(mut self, model_id: Option<&str>) -> Self {
        self.model_id = model_id.map(|s| s.to_string());
        self
    }

    pub fn revision(mut self, revision: Option<&str>) -> Self {
        self.revision = revision.map(|s| s.to_string());
        self
    }

    /// Provide the Hugging Face token. Useful to access gated models.
    pub fn token(mut self, token: Option<&str>) -> Self {
        self.token = token.map(|s| s.to_string());
        self
    }

    pub fn api_key(mut self, api_key: Option<&str>) -> Self {
        self.api_key = api_key.map(|s| s.to_string());
        self
    }

    pub fn path_in_repo(mut self, path_in_repo: Option<&str>) -> Self {
        self.path_in_repo = path_in_repo.map(|s| s.to_string());
        self
    }

    pub fn onnx_model_id(mut self, onnx_model_id: Option<ONNXModel>) -> Self {
        self.onnx_model_id = onnx_model_id;
        self
    }

    pub fn dtype(mut self, dtype: Option<Dtype>) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn from_pretrained_hf(self) -> Result<Embedder, anyhow::Error> {
        match self.model_id {
            Some(model_id) => Embedder::from_pretrained_hf(
                &model_id,
                self.revision.as_deref(),
                self.token.as_deref(),
                self.dtype,
            ),
            None => Err(anyhow::anyhow!("Model ID is required")),
        }
    }

    pub fn from_pretrained_onnx(self) -> Result<Embedder, anyhow::Error> {
        match (self.onnx_model_id, self.model_id) {
            (None, None) => Err(anyhow::anyhow!(
                "Either model_id or onnx_model_id is required"
            )),
            (Some(_), Some(_)) => Err(anyhow::anyhow!(
                "Only one of model_id or onnx_model_id can be provided"
            )),
            (Some(onnx_model_id), None) => Embedder::from_pretrained_onnx(
                &self.model_architecture,
                Some(onnx_model_id),
                self.revision.as_deref(),
                None,
                self.dtype,
                self.path_in_repo.as_deref(),
            ),
            (None, Some(model_id)) => Embedder::from_pretrained_onnx(
                &self.model_architecture,
                None,
                self.revision.as_deref(),
                Some(model_id.as_str()),
                self.dtype,
                self.path_in_repo.as_deref(),
            ),
        }
    }

    pub fn from_pretrained_cloud(self) -> Result<Embedder, anyhow::Error> {
        Embedder::from_pretrained_cloud(
            &self.model_architecture,
            &self.model_id.unwrap(),
            self.api_key,
        )
    }
}

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
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
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
    ) -> Result<Self, anyhow::Error> {
        let api = hf_hub::api::sync::ApiBuilder::from_env()
            .with_token(token.map(|s| s.to_string()))
            .build()?;
        let api = match revision {
            Some(rev) => api.repo(Repo::with_revision(
                model_id.to_string(),
                hf_hub::RepoType::Model,
                rev.to_string(),
            )),
            None => api.repo(Repo::new(model_id.to_string(), hf_hub::RepoType::Model)),
        };
        let config_filename = api.get("config.json")?;
        let config = std::fs::read_to_string(config_filename)?;
        let config: serde_json::Value = serde_json::from_str(&config)?;

        let architecture = config["architectures"]
            .as_array()
            .ok_or(anyhow::anyhow!("Architecture not found"))?
            .first()
            .ok_or(anyhow::anyhow!("Architecture not found"))?
            .as_str()
            .ok_or(anyhow::anyhow!("Architecture not found"))?;
        match architecture {
            "CLIPModel" | "SiglipModel"  => Ok(Self::Vision(Box::new(
                VisionEmbedder::from_pretrained_hf(architecture, model_id, revision, token)?,
            ))),
            "ColPali" => Ok(Self::Vision(Box::new(
                VisionEmbedder::from_pretrained_hf(architecture, model_id, revision, token)?,
            ))),
            "Dinov2Model" => Ok(Self::Vision(Box::new(
                VisionEmbedder::from_pretrained_hf(architecture, model_id, revision, token)?,
            ))),
            "BertModel" => Ok(Self::Text(TextEmbedder::from_pretrained_hf(
                architecture,
                model_id,
                revision,
                token,
                dtype,
            )?)),
            "JinaBertForMaskedLM" => Ok(Self::Text(TextEmbedder::from_pretrained_hf(
                architecture,
                model_id,
                revision,
                token,
                dtype,
            )?)),
            "StaticModel" => Ok(Self::Text(TextEmbedder::from_pretrained_hf(
                architecture,
                model_id,
                revision,
                token,
                dtype,
            )?)),
            "BertForMaskedLM" => Ok(Self::Text(TextEmbedder::from_pretrained_hf(
                architecture,
                model_id,
                revision,
                token,
                dtype,
            )?)),
            "ModernBertForMaskedLM" => Ok(Self::Text(TextEmbedder::from_pretrained_hf(
                architecture,
                model_id,
                revision,
                token,
                dtype,
            )?)),
            "Qwen3ForCausalLM" => Ok(Self::Text(TextEmbedder::from_pretrained_hf(
                architecture,
                model_id,
                revision,
                token,
                dtype,
            )?)),
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }

    pub fn from_pretrained_cloud(
        model: &str,
        model_id: &str,
        api_key: Option<String>,
    ) -> Result<Self, anyhow::Error> {
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
            _ => Err(anyhow::anyhow!("Model not supported")),
        }
    }

    #[cfg(not(feature = "ort"))]
    pub fn from_pretrained_onnx(
        _model_architecture: &str,
        _model_name: Option<ONNXModel>,
        _revision: Option<&str>,
        _model_id: Option<&str>,
        _dtype: Option<Dtype>,
        _path_in_repo: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        Err(anyhow::anyhow!(
            "The 'ort' feature must be enabled to use the 'from_pretrained_ort' function."
        ))
    }

    #[cfg(feature = "ort")]
    pub fn from_pretrained_onnx(
        model_architecture: &str,
        model_name: Option<ONNXModel>,
        revision: Option<&str>,
        model_id: Option<&str>,
        dtype: Option<Dtype>,
        path_in_repo: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
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

    /// Embeds a list of files.
    ///
    /// # Arguments
    ///
    /// * `files` - A vector of `PathBuf` objects representing the files to embed.
    /// * `embedder` - A reference to the embedding model to use.
    /// * `config` - An optional `TextEmbedConfig` object specifying the configuration for the embedding model.
    /// * `adapter` - An optional callback function to handle the embeddings.
    ///
    /// # Returns
    /// An `Option` containing a vector of `EmbedData` objects representing the embeddings of the files, or `None` if an adapter is used.
    ///
    /// # Errors
    /// Returns a `Result` with an error if the embedding process fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use embed_anything::embed_files_batch;
    /// use std::path::PathBuf;
    /// use std::sync::Arc;
    /// use embed_anything::config::TextEmbedConfig;
    /// use embed_anything::embeddings::embed::EmbedderBuilder;
    /// use embed_anything::embeddings::embed::EmbedData;
    ///
    /// async fn generate_embeddings() {
    ///     let files = vec![PathBuf::from("test_files/test.txt"), PathBuf::from("test_files/test.pdf")];
    ///     let embedder = Arc::new(EmbedderBuilder::new()
    ///         .model_architecture("bert")
    ///         .model_id(Some("jinaai/jina-embeddings-v2-small-en"))
    ///         .from_pretrained_hf()
    ///         .unwrap());
    ///     let config = TextEmbedConfig::default();
    ///     let embeddings = embedder.embed_files_batch(files, Some(&config), None).await.unwrap();
    /// }
    /// ```
    /// This will output the embeddings of the files in the specified directory using the specified embedding model.
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

impl EmbedImage for Embedder {
    async fn embed_image<T: AsRef<std::path::Path>>(
        &self,
        image_path: T,
        metadata: Option<HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData> {
        match self {
            Self::Vision(embedder) => embedder.embed_image(image_path, metadata).await,
            _ => Err(anyhow::anyhow!("Model not supported for vision embedding")),
        }
    }

    async fn embed_image_batch<T: AsRef<std::path::Path>>(
        &self,
        image_paths: &[T],
        batch_size: Option<usize>,
    ) -> anyhow::Result<Vec<EmbedData>> {
        match self {
            Self::Vision(embedder) => embedder.embed_image_batch(image_paths, batch_size).await,
            _ => Err(anyhow::anyhow!("Model not supported for vision embedding")),
        }
    }

    async fn embed_pdf<T: AsRef<std::path::Path>>(
        &self,
        pdf_path: T,
        batch_size: Option<usize>,
    ) -> anyhow::Result<Vec<EmbedData>> {
        match self {
            Self::Vision(embedder) => embedder.embed_pdf(pdf_path, batch_size).await,
            _ => Err(anyhow::anyhow!("Model not supported for PDF embedding")),
        }
    }
}

pub trait TextEmbed {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> impl Future<Output = anyhow::Result<Vec<EmbeddingResult>>>;
}

impl TextEmbed for VisionEmbedder {
    async fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        match self {
            Self::Clip(embedder) => embedder.embed(text_batch, batch_size),
            Self::VisionEncoder(_) => Err(anyhow::anyhow!("Model not supported for text embedding")),
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
            Self::VisionEncoder(embedder) => embedder.embed_image_batch(image_paths, batch_size).await,
        }
    }

    async fn embed_pdf<T: AsRef<std::path::Path>>(
        &self,
        pdf_path: T,
        batch_size: Option<usize>,
    ) -> anyhow::Result<Vec<EmbedData>> {
        match self {
            Self::Cohere(embedder) => embedder.embed_pdf(pdf_path, batch_size).await,
            _ => Err(anyhow::anyhow!("Model not supported for PDF embedding")),
        }
    }
}
