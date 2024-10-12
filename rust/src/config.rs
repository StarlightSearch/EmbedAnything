use std::sync::Arc;

use crate::{embeddings::embed::Embedder, text_loader::SplittingStrategy};

#[derive(Clone)]
pub struct TextEmbedConfig {
    pub chunk_size: Option<usize>,
    pub batch_size: Option<usize>,
    pub buffer_size: Option<usize>, // Required for adapter. Default is 100.
    pub splitting_strategy: Option<SplittingStrategy>,
    pub semantic_encoder: Option<Arc<Embedder>>,
}

impl Default for TextEmbedConfig {
    fn default() -> Self {
        Self {
            chunk_size: Some(256),
            batch_size: Some(32),
            buffer_size: Some(100),
            splitting_strategy: None,
            semantic_encoder: None,
        }
    }
}

impl TextEmbedConfig {
    pub fn new(
        chunk_size: Option<usize>,
        batch_size: Option<usize>,
        buffer_size: Option<usize>,
        splitting_strategy: Option<SplittingStrategy>,
        semantic_encoder: Option<Arc<Embedder>>,
    ) -> Self {
        Self {
            chunk_size,
            batch_size,
            buffer_size,
            splitting_strategy,
            semantic_encoder,
        }
    }
}

#[derive(Clone)]
pub struct ImageEmbedConfig {
    pub buffer_size: Option<usize>, // Required for adapter. Default is 100.
}

impl Default for ImageEmbedConfig {
    fn default() -> Self {
        Self {
            buffer_size: Some(100),
        }
    }
}

impl ImageEmbedConfig {
    pub fn new(buffer_size: Option<usize>) -> Self {
        Self { buffer_size }
    }
}
