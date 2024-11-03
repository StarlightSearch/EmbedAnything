use std::path::PathBuf;

use anyhow::Error as E;
use candle_core::Device;
use half::f16;
use ndarray::prelude::*;
use ort::{CUDAExecutionProvider, ExecutionProvider, GraphOptimizationLevel, Session};
use rayon::prelude::*;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::embeddings::embed::{EmbedData, EmbeddingResult};

use super::colpali::ColPaliEmbed;

pub struct OrtColPaliEmbedder {
    pub model: Session,
    pub tokenizer: Tokenizer,
    dummy_input: Array2<i64>,
    device: Device,
}

impl OrtColPaliEmbedder {
    pub fn new(model_id: &str, revision: Option<&str>, max_length: usize) -> Result<Self, E> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo: hf_hub::api::sync::ApiRepo = match revision {
            Some(rev) => api.repo(hf_hub::Repo::with_revision(
                model_id.to_string(),
                hf_hub::RepoType::Model,
                rev.to_string(),
            )),
            None => api.repo(hf_hub::Repo::new(
                model_id.to_string(),
                hf_hub::RepoType::Model,
            )),
        };

        let (_, tokenizer_filename, weights_filename, _) = {
            let config = repo.get("config.json")?;
            let tokenizer = repo.get("tokenizer.json")?;
            let weights = repo.get("model.onnx")?;
            let data = repo.get("model.onnx_data")?;

            (config, tokenizer, weights, data)
        };

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        let trunc = TruncationParams {
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            max_length,
            ..Default::default()
        };

        tokenizer
            .with_padding(Some(pp))
            .with_truncation(Some(trunc))
            .unwrap();

        let cuda = CUDAExecutionProvider::default();

        if !cuda.is_available()? {
            eprintln!("CUDAExecutionProvider is not available");
        } else {
            println!("Session is using CUDAExecutionProvider");
        }

        let threads = std::thread::available_parallelism().unwrap().get();
        let model = Session::builder()?
            .with_execution_providers([CUDAExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(weights_filename)?;

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let dummy_prompt: &str = "Describe the image";

        let dummy_input = tokenize_batch(&tokenizer, &vec![dummy_prompt.to_string()])?;

        Ok(Self {
            model,
            tokenizer,
            dummy_input,
            device,
        })
    }
}

fn tokenize_batch(tokenizer: &Tokenizer, text_batch: &[String]) -> Result<Array2<i64>, E> {
    let token_ids = tokenizer
        .encode_batch(text_batch.to_vec(), true)
        .map_err(E::msg)?
        .iter()
        .map(|tokens| {
            tokens
                .get_ids()
                .iter()
                .map(|&id| id as i64)
                .collect::<Vec<i64>>()
        })
        .collect::<Vec<Vec<i64>>>();

    let token_ids_array = Array2::from_shape_vec(
        (token_ids.len(), token_ids[0].len()),
        token_ids.into_iter().flatten().collect::<Vec<i64>>(),
    )
    .unwrap();

    Ok(token_ids_array)
}

fn get_attention_mask(tokenizer: &Tokenizer, text_batch: &[String]) -> Result<Array2<i64>, E> {
    let attention_mask = tokenizer
        .encode_batch(text_batch.to_vec(), true)
        .map_err(E::msg)?
        .iter()
        .map(|tokens| {
            tokens
                .get_attention_mask()
                .to_vec()
                .iter()
                .map(|x| *x as i64)
                .collect::<Vec<i64>>()
        })
        .collect::<Vec<Vec<i64>>>();
    Ok(Array2::from_shape_vec(
        (attention_mask.len(), attention_mask[0].len()),
        attention_mask.into_iter().flatten().collect::<Vec<i64>>(),
    )
    .unwrap())
}

impl ColPaliEmbed for OrtColPaliEmbedder {
    fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let batch_size = batch_size.unwrap_or(32);
        let encodings: Vec<Vec<Vec<f32>>> = text_batch
            .par_chunks(batch_size)
            .flat_map(|mini_text_batch| -> Result<Vec<Vec<Vec<f32>>>, E> {
                let token_ids: Array2<i64> = tokenize_batch(&self.tokenizer, mini_text_batch)?;
                let attention_mask: Array2<i64> =
                    get_attention_mask(&self.tokenizer, mini_text_batch)?;
                let pixel_values: Array4<f32> = Array4::zeros((mini_text_batch.len(), 3, 448, 448));
                let outputs = self
                    .model
                    .run(ort::inputs![token_ids, pixel_values, attention_mask].unwrap())
                    .unwrap();
                println!(
                    "Outputs: {:?}",
                    outputs["last_hidden_state"]
                        .try_extract_raw_tensor::<f16>()
                        .unwrap()
                );
                let embeddings: Array3<f16> = outputs["last_hidden_state"]
                    .try_extract_tensor::<f16>()?
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix3>()?;
                println!("Embeddings: {:?}", embeddings);

                let e = embeddings
                    .outer_iter()
                    .map(|row| row.outer_iter().map(|x| x.to_vec()).collect::<Vec<_>>())
                    .map(|x| x.iter().map(|y| y.iter().map(|z| z.to_f32()).collect::<Vec<f32>>()).collect::<Vec<Vec<f32>>>())
                    .collect::<Vec<Vec<Vec<f32>>>>();
                Ok(e)
            })
            .flatten()
            .collect::<Vec<_>>();

        Ok(encodings
            .iter()
            .map(|x| EmbeddingResult::MultiVector(x.to_vec()))
            .collect())
    }

    fn embed_query(&self, query: &str) -> anyhow::Result<Vec<EmbedData>> {
        todo!()
    }

    fn embed_file(&self, file_path: PathBuf, batch_size: usize) -> anyhow::Result<Vec<EmbedData>> {
        todo!()
    }

    fn embed_image(
        &self,
        image_path: PathBuf,
        metadata: Option<std::collections::HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData> {
        todo!()
    }

    fn embed_image_batch(&self, image_paths: &[PathBuf]) -> anyhow::Result<Vec<EmbedData>> {
        todo!()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_ort_colpali_model() {
        let model =
            OrtColPaliEmbedder::new("akshayballal/colpali-v1.2-merged-onnx", None, 128).unwrap();
        let embeddings = model
            .embed(&["Hello, world!".to_string()], Some(1))
            .unwrap();
        println!("Embeddings {:?}", embeddings);
    }
}
