use anyhow::{anyhow, Result};

use super::embedder::Embedder;
use crate::embeddings::local::text_embedding::ONNXModel;
use crate::Dtype;

#[derive(Default)]
pub struct EmbedderBuilder {
    // The model architecture that you want to use. For cloud models, it is the provider name.
    model_architecture: String,
    // Either HF Model ID or the Cloud Model that you want to use
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

    pub fn from_pretrained_hf(self) -> Result<Embedder> {
        match self.model_id {
            Some(model_id) => Embedder::from_pretrained_hf(
                &model_id,
                self.revision.as_deref(),
                self.token.as_deref(),
                self.dtype,
            ),
            None => Err(anyhow!("Model ID is required")),
        }
    }

    pub fn from_pretrained_onnx(self) -> Result<Embedder> {
        match (self.onnx_model_id, self.model_id) {
            (None, None) => Err(anyhow!("Either model_id or onnx_model_id is required")),
            (Some(_), Some(_)) => Err(anyhow!(
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

    pub fn from_pretrained_cloud(self) -> Result<Embedder> {
        Embedder::from_pretrained_cloud(
            &self.model_architecture,
            &self.model_id.unwrap(),
            self.api_key,
        )
    }
}
