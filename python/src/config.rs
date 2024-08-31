use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Default)]
pub struct TextEmbedConfig {
    pub inner: embed_anything::config::TextEmbedConfig,
}

#[pymethods]
impl TextEmbedConfig {
    #[new]
    #[pyo3(signature = (chunk_size=None, batch_size=None))]
    pub fn new(chunk_size: Option<usize>, batch_size: Option<usize>) -> Self {
        Self {
            inner: embed_anything::config::TextEmbedConfig::new(chunk_size, batch_size),
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
