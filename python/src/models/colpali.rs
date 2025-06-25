use embed_anything::embeddings::local::colpali::ColPaliEmbed;
use embed_anything::embeddings::local::colpali::ColPaliEmbedder;
use embed_anything::embeddings::local::colpali_ort::OrtColPaliEmbedder;
use embed_anything::embeddings::local::colsmol::ColSmolEmbedder;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;

use crate::EmbedData;
#[pyclass]
pub struct ColpaliModel {
    pub model: Box<dyn ColPaliEmbed + Send + Sync>,
}

#[pymethods]
impl ColpaliModel {
    #[new]
    #[pyo3(signature = (model_id, revision=None))]
    pub fn new(model_id: &str, revision: Option<&str>) -> PyResult<Self> {
        let model_id_small = model_id.to_lowercase();
        let model = if model_id_small.contains("colpali") {
            Box::new(
                ColPaliEmbedder::new(model_id, revision)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ) as Box<dyn ColPaliEmbed + Send + Sync>
        } else if model_id_small.contains("colsmol") {
            Box::new(
                ColSmolEmbedder::new(model_id, revision)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ) as Box<dyn ColPaliEmbed + Send + Sync>
        } else {
            return Err(PyValueError::new_err("Invalid model ID"));
        };
        Ok(Self { model })
    }

    #[staticmethod]
    #[pyo3(signature = (model_id, revision=None))]
    pub fn from_pretrained(model_id: &str, revision: Option<&str>) -> PyResult<Self> {
        let model_id_small = model_id.to_lowercase();
        let model = if model_id_small.contains("colpali") {
            Box::new(
                ColPaliEmbedder::new(model_id, revision)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ) as Box<dyn ColPaliEmbed + Send + Sync>
        } else if model_id_small.contains("colsmol") {
            Box::new(
                ColSmolEmbedder::new(model_id, revision)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ) as Box<dyn ColPaliEmbed + Send + Sync>
        } else {
            return Err(PyValueError::new_err("Invalid model ID"));
        };
        Ok(Self { model })
    }

    #[staticmethod]
    #[pyo3(signature = (model_id, revision=None, path_in_repo=None))]
    pub fn from_pretrained_onnx(
        model_id: &str,
        revision: Option<&str>,
        path_in_repo: Option<&str>,
    ) -> PyResult<Self> {
        let model = OrtColPaliEmbedder::new(model_id, revision, path_in_repo)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            model: Box::new(model),
        })
    }

    pub fn embed_file(&self, file_path: &str, batch_size: usize) -> PyResult<Vec<EmbedData>> {
        let embed_data = self
            .model
            .embed_file(file_path.into(), batch_size)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(embed_data
            .into_iter()
            .map(|data| EmbedData { inner: data })
            .collect())
    }

    pub fn embed_query(&self, query: &str) -> PyResult<Vec<EmbedData>> {
        let embed_data = self
            .model
            .embed_query(query)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(embed_data
            .into_iter()
            .map(|data| EmbedData { inner: data })
            .collect())
    }
}
