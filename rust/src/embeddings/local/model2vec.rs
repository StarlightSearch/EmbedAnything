use anyhow::Error as E;
use model2vec_rs;

use crate::embeddings::embed::EmbeddingResult;

pub struct Model2VecEmbedder {
    pub model: model2vec_rs::model::StaticModel,
}

impl Model2VecEmbedder {
    pub fn new(model_id: &str, token: Option<&str>, path_in_repo: Option<&str>) -> Result<Self, E> {
        let model =
            model2vec_rs::model::StaticModel::from_pretrained(model_id, token, None, path_in_repo)?;
        Ok(Self { model })
    }

    pub fn embed(
        &self,
        text_batch: &[&str],
        _batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, E> {
        let embeddings = self.model.encode(
            text_batch
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
                .as_slice(),
        );
        let embeddings = embeddings
            .iter()
            .map(|e| EmbeddingResult::DenseVector(e.clone()))
            .collect();
        Ok(embeddings)
    }
}
