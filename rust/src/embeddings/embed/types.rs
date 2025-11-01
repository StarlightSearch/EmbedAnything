use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::collections::HashMap;

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
