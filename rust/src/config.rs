//! Configuration structs and enums for embedding operations.
//!
//! Provides configuration options for text and image embedding processes,
//! including chunking strategies, batch sizes, and splitting methods.

use processors_rs::pdf::pdf_processor::PdfBackend;

use crate::embeddings::embed::Embedder;
use std::sync::Arc;

/// Configuration for text embedding.
///
/// # Example: Creating a new instance
///
/// ```rust
/// use embed_anything::config::{TextEmbedConfig, SplittingStrategy};
/// let config = TextEmbedConfig::new(
///     Some(512),
///     Some(128),
///     Some(100),
///     Some(0.0),
///     SplittingStrategy::Sentence,
///     Some(true),
///     None
/// );
/// ```
///
/// # Example: Overriding a single default
///
/// ```rust
/// use embed_anything::config::{TextEmbedConfig, SplittingStrategy};
/// let config = TextEmbedConfig {
///     splitting_strategy: SplittingStrategy::Semantic,
///     ..Default::default()
/// };
/// ```
#[derive(Clone)]
pub struct TextEmbedConfig {
    /// Controls the size of each "chunk" of data that your input text gets split into. Defaults to
    /// 1000 Characters.
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
    pub splitting_strategy: SplittingStrategy,
    /// When embedding a PDF, controls whether **o**ptical **c**haracter **r**ecognition is used on
    /// the PDF to extract text. This process involves rendering the PDF as a series of images, and
    /// extracting text from the images. Defaults to false.
    pub use_ocr: Option<bool>,
    pub tesseract_path: Option<String>,
    /// When embedding a document, controls whether late chunking is used. Use this to take larger context into account for embedding. Defaults to false.
    pub late_chunking: Option<bool>,
    /// When embedding a PDF, controls which backend is used to extract text. Defaults to [PdfBackend::LoPdf]
    pub pdf_backend: PdfBackend,
}

impl Default for TextEmbedConfig {
    fn default() -> Self {
        Self {
            chunk_size: Some(1000),
            overlap_ratio: Some(0.0),
            batch_size: Some(32),
            buffer_size: Some(100),
            splitting_strategy: SplittingStrategy::Sentence,
            late_chunking: None,
            use_ocr: None,
            tesseract_path: None,
            pdf_backend: PdfBackend::LoPdf,
        }
    }
}

#[allow(clippy::too_many_arguments)]
impl TextEmbedConfig {
    pub fn new(
        chunk_size: Option<usize>,
        batch_size: Option<usize>,
        buffer_size: Option<usize>,
        overlap_ratio: Option<f32>,
        splitting_strategy: SplittingStrategy,
        late_chunking: Option<bool>,
        use_ocr: Option<bool>,
        tesseract_path: Option<String>,
    ) -> Self {
        Self::default()
            .with_chunk_size(chunk_size.unwrap_or(1000), overlap_ratio)
            .with_batch_size(batch_size.unwrap_or(32))
            .with_buffer_size(buffer_size.unwrap_or(100))
            .with_ocr(use_ocr.unwrap_or(false), tesseract_path.as_deref())
            .with_pdf_backend("lopdf")
            .with_splitting_strategy(splitting_strategy)
            .with_late_chunking(late_chunking.unwrap_or(false))
            .build()
    }

    pub fn with_chunk_size(mut self, size: usize, overlap_ratio: Option<f32>) -> Self {
        self.chunk_size = Some(size);
        self.overlap_ratio = Some(overlap_ratio.unwrap_or(0.0));
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

    pub fn with_late_chunking(mut self, late_chunking: bool) -> Self {
        self.late_chunking = Some(late_chunking);
        self
    }

    pub fn with_splitting_strategy(mut self, strategy: SplittingStrategy) -> Self {
        self.splitting_strategy = strategy;
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

    /// Set the PDF backend to use. Defaults to "lopdf". Other backends will be supported in the future.
    pub fn with_pdf_backend(mut self, backend: &str) -> Self {
        self.pdf_backend = match backend {
            "lopdf" => PdfBackend::LoPdf,
            _ => PdfBackend::LoPdf,
        };
        self
    }

    pub fn build(self) -> TextEmbedConfig {
        self
    }
}

#[derive(Clone)]
pub enum SplittingStrategy {
    /// Splits text-based content by sentence, resulting in one embedding per sentence.
    Sentence,
    /// Uses an embedder to determine semantic relevance of chunks of text. Produces embeddings that
    /// may be longer, or shorter than a sentence.
    Semantic {
        /// Specifies the embedder used when the splitting semantically.
        semantic_encoder: Arc<Embedder>,
    },
}

#[derive(Clone)]
pub struct ImageEmbedConfig {
    pub buffer_size: Option<usize>, // Required for adapter. Default is 100.
    pub batch_size: Option<usize>,
}

impl Default for ImageEmbedConfig {
    fn default() -> Self {
        Self {
            buffer_size: Some(100),
            batch_size: Some(32),
        }
    }
}

impl ImageEmbedConfig {
    pub fn new(buffer_size: Option<usize>, batch_size: Option<usize>) -> Self {
        Self {
            buffer_size,
            batch_size,
        }
    }
}
