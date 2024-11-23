use std::sync::Arc;

use crate::{embeddings::embed::Embedder, text_loader::SplittingStrategy};

#[derive(Clone)]
pub struct TextEmbedConfig {
    pub chunk_size: Option<usize>,
    pub overlap_ratio: Option<f32>,
    pub batch_size: Option<usize>,
    pub buffer_size: Option<usize>, // Required for adapter. Default is 100.
    pub splitting_strategy: Option<SplittingStrategy>,
    pub semantic_encoder: Option<Arc<Embedder>>,
    pub use_ocr: Option<bool>,
}

impl Default for TextEmbedConfig {
    fn default() -> Self {
        Self {
            chunk_size: Some(256),
            overlap_ratio: Some(0.0),
            batch_size: Some(32),
            buffer_size: Some(100),
            splitting_strategy: None,
            semantic_encoder: None,
            use_ocr: None,
        }
    }
}

impl TextEmbedConfig {
    pub fn new(
        chunk_size: Option<usize>,
        batch_size: Option<usize>,
        buffer_size: Option<usize>,
        overlap_ratio: Option<f32>,
        splitting_strategy: Option<SplittingStrategy>,
        semantic_encoder: Option<Arc<Embedder>>,
        use_ocr: Option<bool>,
    ) -> Self {
        let config = Self::default()
            .with_chunk_size(chunk_size.unwrap_or(256), overlap_ratio)
            .with_batch_size(batch_size.unwrap_or(32))
            .with_buffer_size(buffer_size.unwrap_or(100))
            .with_ocr(use_ocr.unwrap_or(false));

        match splitting_strategy {
            Some(SplittingStrategy::Semantic) => {
                if semantic_encoder.is_none() {
                    panic!("Semantic encoder is required when using Semantic splitting strategy");
                }
                config
                    .with_semantic_encoder(semantic_encoder.unwrap())
                    .with_splitting_strategy(SplittingStrategy::Semantic)
            }
            Some(strategy) => config.with_splitting_strategy(strategy),
            None => config,
        }
    }

    pub fn with_chunk_size(mut self, size: usize, overlap_ratio: Option<f32>) -> Self {
        self.chunk_size = Some(size);
        self.overlap_ratio = overlap_ratio;
        self
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = Some(size);
        self
    }

    pub fn with_splitting_strategy(mut self, strategy: SplittingStrategy) -> Self {
        self.splitting_strategy = Some(strategy);
        self
    }

    pub fn with_semantic_encoder(mut self, encoder: Arc<Embedder>) -> Self {
        self.semantic_encoder = Some(encoder);
        self
    }

    pub fn with_ocr(mut self, use_ocr: bool) -> Self {
        self.use_ocr = Some(use_ocr);
        self
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
