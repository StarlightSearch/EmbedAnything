#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use super::embed::TextEmbed;
use crate::embedding_model::normalize_l2;
use anyhow::Error as E;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use hf_hub::{api::sync::Api, Repo};
use tokenizers::{PaddingParams, Tokenizer};

pub struct BertEmbeder {
    pub model: BertModel,
    pub tokenizer: Tokenizer,
}

impl Default for BertEmbeder {
    fn default() -> Self {
        Self::new("sentence-transformers/all-MiniLM-L12-v2".to_string(), None).unwrap()
    }
}
impl BertEmbeder {
    pub fn new(model_id: String, revision: Option<String>) -> Result<Self, E> {
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new().unwrap();
            let api = match revision {
                Some(rev) => api.repo(Repo::with_revision(model_id, hf_hub::RepoType::Model, rev)),
                None => api.repo(hf_hub::Repo::new(
                    model_id.to_string(),
                    hf_hub::RepoType::Model,
                )),
            };
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = api.get("model.safetensors")?;

            (config, tokenizer, weights)
        };
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: Config = serde_json::from_str(&config)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device).unwrap()
        };

        config.hidden_act = HiddenAct::GeluApproximate;

        let model = BertModel::load(vb, &config).unwrap();
        Ok(BertEmbeder { model, tokenizer })
    }

    pub fn tokenize_batch(&self, text_batch: &[String], device: &Device) -> anyhow::Result<Tensor> {
        let tokens = self
            .tokenizer
            .encode_batch(text_batch.to_vec(), true)
            .map_err(E::msg)?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Tensor::new(tokens.as_slice(), device)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        Ok(Tensor::stack(&token_ids, 0)?)
    }

    pub fn embed(&self, text_batch: &[String]) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        let token_ids = self.tokenize_batch(text_batch, &self.model.device).unwrap();
        let token_type_ids = token_ids.zeros_like().unwrap();
        let embeddings = self.model.forward(&token_ids, &token_type_ids).unwrap();
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3().unwrap();
        let embeddings = (embeddings.sum(1).unwrap() / (n_tokens as f64)).unwrap();
        let embeddings = normalize_l2(&embeddings).unwrap();
        let encodings = embeddings.to_vec2::<f32>().unwrap();
        Ok(encodings)
    }
}

impl TextEmbed for BertEmbeder {
    fn embed(&self, text_batch: &[String]) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        self.embed(text_batch)
    }
}

#[cfg(test)]

mod tests {
    use crate::embedding_model::embed_audio;
    use crate::file_processor::audio::audio_processor::Segment;

    use super::*;
    use std::path::PathBuf;
    use std::time::Instant;

    #[test]
    fn test_bert_embeder() {
        let embeder = BertEmbeder::default();
        let text = vec![
            "Rust is a systems programming language".to_string(),
            "It is blazingly fast".to_string(),
        ];
        let start = Instant::now();
        let embeddings = embeder.embed(&text).unwrap();
        println!("Time taken: {:?}", start.elapsed());
        assert_eq!(embeddings.len(), 2);
    }

    #[test]
    fn test_bert_embeder_audio() {
        let embeder = BertEmbeder::default();
        let segments = vec![
            Segment {
                start: 0.0,
                duration: 1.0,
                dr: Default::default(),
            },
            Segment {
                start: 1.0,
                duration: 1.0,
                dr: Default::default(),
            },
        ];
        let audio_file = PathBuf::from("tests/data/sample.wav");
        let start = Instant::now();
        let embeddings = embed_audio(&embeder, segments, audio_file).unwrap();
        println!("Time taken: {:?}", start.elapsed());
        assert_eq!(embeddings.len(), 2);
    }
}
