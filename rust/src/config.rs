use std::sync::Arc;

use crate::{embeddings::embed::Embedder, text_loader::SplittingStrategy};

/// Configuration for text embedding.
///
/// # Example: Creating a new instance
///
/// ```rust
/// use embed_anything::config::TextEmbedConfig;
/// use embed_anything::text_loader::SplittingStrategy;
/// let config = TextEmbedConfig::new(
///     Some(512),
///     Some(128),
///     Some(100),
///     Some(0.0),
///     Some(SplittingStrategy::Sentence),
///     None,
///     Some(true)
/// );
/// ```
///
/// # Example: Overriding a single default
///
/// ```rust
/// use embed_anything::config::TextEmbedConfig;
/// use embed_anything::text_loader::SplittingStrategy;
/// let config = TextEmbedConfig {
///     splitting_strategy: Some(SplittingStrategy::Semantic),
///     ..Default::default()
/// };
/// ```
#[derive(Clone)]
pub struct TextEmbedConfig {
    /// Controls the size of each "chunk" of data that your input text gets split into. Defaults to
    /// 256.
    pub chunk_size: Option<usize>,
    /// Controls the ratio of overlapping data across "chunks" of your input text. Defaults to 0.0,
    /// or no overlap.
    pub overlap_ratio: Option<f32>,
    /// Controls the size of each "batch" of data sent to the embedder. The default value depends
    /// largely on the embedder, but will be set to 32 when using [TextEmbedConfig::default()]
    pub batch_size: Option<usize>,
    /// When using an adapter, this controls the size of the buffer. Defaults to 100.
    pub buffer_size: Option<usize>,
    /// Controls how documents are split into segments. See [SplittingStrategy] for options.
    /// Defaults to [SplittingStrategy::Sentence]
    pub splitting_strategy: Option<SplittingStrategy>,
    /// Allows overriding the embedder used when the splitting strategy is
    /// [SplittingStrategy::Semantic]. Defaults to JINA.
    pub semantic_encoder: Option<Arc<Embedder>>,
    /// When embedding a PDF, controls whether **o**ptical **c**haracter **r**ecognition is used on
    /// the PDF to extract text. This process involves rendering the PDF as a series of images, and
    /// extracting text from the images. Defaults to false.
    pub use_ocr: Option<bool>,
    pub tesseract_path: Option<String>,
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
            tesseract_path: None,
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
        tesseract_path: Option<String>,
    ) -> Self {
        let config = Self::default()
            .with_chunk_size(chunk_size.unwrap_or(256), overlap_ratio)
            .with_batch_size(batch_size.unwrap_or(32))
            .with_buffer_size(buffer_size.unwrap_or(100))
            .with_ocr(use_ocr.unwrap_or(false), tesseract_path.as_deref());

        match splitting_strategy {
            Some(SplittingStrategy::Semantic) => {
                if semantic_encoder.is_none() {
                    panic!("Semantic encoder is required when using Semantic splitting strategy");
                }
                config
                    .with_semantic_encoder(Some(semantic_encoder.unwrap()))
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

    pub fn with_semantic_encoder(mut self, encoder: Option<Arc<Embedder>>) -> Self {
        self.semantic_encoder = encoder;
        self
    }

    /// Use this to do OCR on the documents to extract text. 
    /// Set the path to None if you want to use the default path with tesseract installed on your system. 
    /// You can check if tesseract is installed by running tesseract in your command line. 
    /// If you want to use a custom path, you can set the path to the path of the tesseract executable.
    pub fn with_ocr(mut self, use_ocr: bool, tesseract_path: Option<&str>) -> Self {
        self.use_ocr = Some(use_ocr);
        self.tesseract_path = tesseract_path.map(|p| p.to_string());
        self
    }

    pub fn build(self) -> TextEmbedConfig {
        if self.semantic_encoder.is_none() && self.splitting_strategy.is_some() {
            panic!("Semantic encoder is required when using Semantic splitting strategy");
        }
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
