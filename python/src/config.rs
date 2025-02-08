use pyo3::prelude::*;
use embed_anything::config::SplittingStrategy;
use crate::EmbeddingModel;

#[pyclass]
#[derive(Default)]
pub struct TextEmbedConfig {
    pub inner: embed_anything::config::TextEmbedConfig,
}

#[pymethods]
impl TextEmbedConfig {
    #[new]
    #[pyo3(signature = (chunk_size=None, batch_size=None, buffer_size=None, overlap_ratio=None, splitting_strategy=None, semantic_encoder=None, use_ocr=None, tesseract_path=None))]
    pub fn new(
        chunk_size: Option<usize>,
        batch_size: Option<usize>,
        buffer_size: Option<usize>,
        overlap_ratio: Option<f32>,
        splitting_strategy: Option<&str>,
        semantic_encoder: Option<&EmbeddingModel>,
        use_ocr: Option<bool>,
        tesseract_path: Option<&str>,
    ) -> Self {
        let strategy = match splitting_strategy {
            Some(strategy) => match strategy {
                "sentence" => SplittingStrategy::Sentence,
                "semantic" => {
                    if semantic_encoder.is_none() {
                        panic!("Semantic encoder is required when using Semantic splitting strategy");
                    }
                    SplittingStrategy::Semantic {
                        semantic_encoder: semantic_encoder.unwrap().inner.clone()
                    }
                },
                _ => panic!("Unknown strategy provided!"),
            },
            None => SplittingStrategy::Sentence,
        };

        Self {
            inner: embed_anything::config::TextEmbedConfig::default()
                .with_chunk_size(chunk_size.unwrap_or(256), overlap_ratio)
                .with_batch_size(batch_size.unwrap_or(32))
                .with_buffer_size(buffer_size.unwrap_or(100))
                .with_splitting_strategy(strategy)
                .with_ocr(use_ocr.unwrap_or(false), tesseract_path)
        }
    }

    #[getter]
    pub fn chunk_size(&self) -> Option<usize> {
        self.inner.chunk_size
    }

    #[getter]
    pub fn batch_size(&self) -> Option<usize> {
        self.inner.batch_size
    }
}

#[pyclass]
#[derive(Clone, Default)]
pub struct ImageEmbedConfig {
    pub inner: embed_anything::config::ImageEmbedConfig,
}

#[pymethods]
impl ImageEmbedConfig {
    #[new]
    #[pyo3(signature = (buffer_size=None))]
    pub fn new(buffer_size: Option<usize>) -> Self {
        Self {
            inner: embed_anything::config::ImageEmbedConfig::new(buffer_size),
        }
    }

    #[getter]
    pub fn buffer_size(&self) -> Option<usize> {
        self.inner.buffer_size
    }
}
