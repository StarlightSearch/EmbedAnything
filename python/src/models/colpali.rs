use embed_anything::embeddings::local::colpali::ColPaliEmbedder;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;

use crate::EmbedData;
#[pyclass]
pub struct ColpaliModel {
    pub model: ColPaliEmbedder,
}

#[pymethods]
impl ColpaliModel {

    #[new]
    #[pyo3(signature = (model_id, revision=None))]
    pub fn new(model_id: &str, revision: Option<&str>) -> PyResult<Self> {
        let model = ColPaliEmbedder::new(model_id, revision)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { model })
    }

    #[staticmethod]
    #[pyo3(signature = (model_id, revision=None))]
    pub fn from_pretrained(model_id: &str, revision: Option<&str>) -> PyResult<Self> {
        let model = ColPaliEmbedder::new(model_id, revision)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { model })
    }

    pub fn embed_file(&self, file_path: &str, batch_size: usize) -> PyResult<Vec<EmbedData>> {
        let embed_data = self
            .model
            .embed_file(file_path, batch_size)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(embed_data
            .into_iter()
            .map(|data| EmbedData { inner: data })
            .collect())
    }

    pub fn embed_query(&self, query: &str) -> PyResult<Vec<EmbedData>> {
        let embed_data = self.model.embed_query(query).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(embed_data
            .into_iter()
            .map(|data| EmbedData { inner: data })
            .collect())
    }
}
