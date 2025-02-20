use crate::{
    embeddings::{normalize_l2, utils::tokenize_batch},
    models::modernbert::{Config, ModernBert},
    Dtype,
};
use anyhow::Error as E;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::ApiBuilder, Repo};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::embeddings::{embed::EmbeddingResult, select_device};

use super::{
    bert::BertEmbed,
    pooling::{ModelOutput, PooledOutputType, Pooling},
};
pub struct ModernBertEmbedder {
    pub model: ModernBert,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub pooling: Pooling,
}

impl Default for ModernBertEmbedder {
    fn default() -> Self {
        Self::new(
            "nomic-ai/modernbert-embed-base".to_string(),
            None,
            None,
            None,
        )
        .unwrap()
    }
}
impl ModernBertEmbedder {
    pub fn new(
        model_id: String,
        revision: Option<String>,
        token: Option<&str>,
        dtype: Option<Dtype>,
    ) -> Result<Self, E> {
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = ApiBuilder::new()
                .with_token(token.map(|s| s.to_string()))
                .build()
                .unwrap();
            let api = match revision {
                Some(rev) => api.repo(Repo::with_revision(model_id, hf_hub::RepoType::Model, rev)),
                None => api.repo(hf_hub::Repo::new(
                    model_id.to_string(),
                    hf_hub::RepoType::Model,
                )),
            };
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = match api.get("model.safetensors") {
                Ok(safetensors) => safetensors,
                Err(_) => match api.get("pytorch_model.bin") {
                    Ok(pytorch_model) => pytorch_model,
                    Err(e) => {
                        return Err(anyhow::Error::msg(format!(
                            "Model weights not found. The weights should either be a `model.safetensors` or `pytorch_model.bin` file.  Error: {}",
                            e
                        )));
                    }
                },
            };

            (config, tokenizer, weights)
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        let trunc = TruncationParams {
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            max_length: config.max_position_embeddings as usize,
            ..Default::default()
        };

        tokenizer
            .with_padding(Some(pp))
            .with_truncation(Some(trunc))
            .unwrap();

        let device = select_device();
        let dtype = match dtype {
            Some(Dtype::F16) => DType::F16,
            Some(Dtype::F32) => DType::F32,
            _ => DType::F32,
        };
        let vb = if weights_filename.ends_with("model.safetensors") {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, &device)? }
        } else {
            println!("Can't find model.safetensors, loading from pytorch_model.bin");
            VarBuilder::from_pth(&weights_filename, dtype, &device)?
        };

        let model = ModernBert::load(vb, &config)?;
        let tokenizer = tokenizer;

        Ok(ModernBertEmbedder {
            model,
            tokenizer,
            device,
            pooling: Pooling::Mean,
        })
    }
}

impl BertEmbed for ModernBertEmbedder {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let batch_size = batch_size.unwrap_or(32);
        let mut encodings: Vec<EmbeddingResult> = Vec::new();

        for mini_text_batch in text_batch.chunks(batch_size) {
            let (token_ids, attention_mask) =
                tokenize_batch(&self.tokenizer, mini_text_batch, &self.device)?;

            let embeddings: Tensor = self
                .model
                .forward(&token_ids, &attention_mask)?
                .to_dtype(DType::F32)
                .unwrap();

            let attention_mask = PooledOutputType::from(attention_mask);
            let attention_mask = Some(&attention_mask);
            let model_output = ModelOutput::Tensor(embeddings.clone());
            let pooled_output = match self.pooling {
                Pooling::Cls => self.pooling.pool(&model_output, None)?,
                Pooling::Mean => self.pooling.pool(&model_output, attention_mask)?,
            };
            let pooled_output = pooled_output.to_tensor()?;
            let embeddings = normalize_l2(pooled_output)?;
            let batch_encodings = embeddings.to_vec2::<f32>()?;

            encodings.extend(
                batch_encodings
                    .iter()
                    .map(|x| EmbeddingResult::DenseVector(x.to_vec())),
            );
        }
        Ok(encodings)
    }
}
