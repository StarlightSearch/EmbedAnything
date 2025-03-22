use std::rc::Rc;

use embed_anything::embeddings::get_text_metadata;
use embed_anything::embeddings::local::colbert::{ColbertEmbed, OrtColbertEmbedder};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;

use crate::EmbedData;

#[pyclass]
pub struct ColbertModel {
    pub model: Box<dyn ColbertEmbed + Send + Sync>,
}

#[pymethods]
impl ColbertModel {
    #[new]
    #[pyo3(signature = (hf_model_id=None, revision=None, path_in_repo=None))]
    pub fn new(
        hf_model_id: Option<&str>,
        revision: Option<&str>,
        path_in_repo: Option<&str>,
    ) -> PyResult<Self> {
        let model = OrtColbertEmbedder::new(hf_model_id, revision, path_in_repo)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            model: Box::new(model),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (hf_model_id=None, revision=None, path_in_repo=None))]
    fn from_pretrained_onnx(
        hf_model_id: Option<&str>,
        revision: Option<&str>,
        path_in_repo: Option<&str>,
    ) -> PyResult<Self> {
        let model = OrtColbertEmbedder::new(hf_model_id, revision, path_in_repo)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            model: Box::new(model),
        })
    }

    #[pyo3(signature = (text_batch, batch_size=None, is_doc=true))]
    pub fn embed(
        &self,
        text_batch: Vec<String>,
        batch_size: Option<usize>,
        is_doc: bool,
    ) -> PyResult<Vec<EmbedData>> {
        let text_batch = text_batch.iter().map(|s| s.as_str()).collect::<Vec<&str>>();

        let embed_data = self
            .model
            .embed(&text_batch, batch_size, is_doc)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let embeddings = get_text_metadata(&Rc::new(embed_data), &text_batch, &None)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(embeddings
            .into_iter()
            .map(|data| EmbedData { inner: data })
            .collect())
    }
}
