use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;

#[pyclass]
pub struct Reranker {
    pub model: embed_anything::reranker::model::Reranker,
}

#[pyclass(eq, eq_int)]
#[derive(PartialEq)]

pub enum Dtype {
    F16,
    INT8,
    Q4,
    UINT8,
    BNB4,
    Q4F16,
    F32,
    BF16,
}

#[pyclass]
pub struct RerankerResult {
    pub inner: embed_anything::reranker::model::RerankerResult,
}

#[pyclass]
pub struct DocumentRank {
    pub document: String,
    pub relevance_score: f32,
    pub rank: usize,
}

#[pymethods]
impl DocumentRank {
    #[getter(document)]
    fn document(&self) -> String {
        self.document.clone()
    }

    #[getter(relevance_score)]
    fn relevance_score(&self) -> f32 {
        self.relevance_score
    }

    #[getter(rank)]
    fn rank(&self) -> usize {
        self.rank
    }

    fn __str__(&self) -> String {
        format!(
            "{{\"document\": \"{}\", \"relevance_score\": {}, \"rank\": {}}}",
            self.document, self.relevance_score, self.rank
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "DocumentRank(document={}, relevance_score={}, rank={})",
            self.document, self.relevance_score, self.rank
        )
    }
}

#[pymethods]
impl RerankerResult {
    #[getter(query)]
    fn query(&self) -> String {
        self.inner.query.clone()
    }

    #[getter(documents)]
    fn documents(&self) -> Vec<DocumentRank> {
        self.inner
            .documents
            .clone()
            .into_iter()
            .map(|d| DocumentRank {
                document: d.document,
                relevance_score: d.relevance_score,
                rank: d.rank,
            })
            .collect()
    }

    fn __str__(&self) -> String {
        format!(
            "Query: {}\nDocuments: {}",
            self.query(),
            self.documents()
                .iter()
                .map(|d| format!(
                    "Document: {}, Relevance Score: {}, Rank: {}",
                    d.document, d.relevance_score, d.rank
                ))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

#[pymethods]
impl Reranker {
    #[staticmethod]
    #[pyo3(signature = (model_id, revision=None, dtype=None, path_in_repo=None))]
    pub fn from_pretrained(
        model_id: &str,
        revision: Option<&str>,
        dtype: Option<&Dtype>,
        path_in_repo: Option<&str>,
    ) -> PyResult<Self> {
        let dtype = match dtype {
            Some(Dtype::F16) => embed_anything::Dtype::F16,
            Some(Dtype::INT8) => embed_anything::Dtype::INT8,
            Some(Dtype::Q4) => embed_anything::Dtype::Q4,
            Some(Dtype::UINT8) => embed_anything::Dtype::UINT8,
            Some(Dtype::BNB4) => embed_anything::Dtype::BNB4,
            Some(Dtype::F32) => embed_anything::Dtype::F32,
            _ => embed_anything::Dtype::F32,
        };
        let model =
            embed_anything::reranker::model::Reranker::new(model_id, revision, dtype, path_in_repo)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { model })
    }

    #[pyo3(signature = (query, documents, batch_size))]
    pub fn rerank(
        &self,
        query: Vec<String>,
        documents: Vec<String>,
        batch_size: usize,
    ) -> PyResult<Vec<RerankerResult>> {
        let query_refs: Vec<&str> = query.iter().map(|s| s.as_str()).collect();
        let document_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let results = self
            .model
            .rerank(query_refs, document_refs, batch_size)
            .unwrap();
        Ok(results
            .into_iter()
            .map(|r| RerankerResult { inner: r })
            .collect::<Vec<_>>())
    }

    #[pyo3(signature = (query, documents, batch_size))]
    pub fn compute_scores(
        &self,
        query: Vec<String>,
        documents: Vec<String>,
        batch_size: usize,
    ) -> PyResult<Vec<Vec<f32>>> {
        let query_refs: Vec<&str> = query.iter().map(|s| s.as_str()).collect();
        let document_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let scores = self
            .model
            .compute_scores(query_refs, document_refs, batch_size)
            .unwrap();
        Ok(scores)
    }
}
