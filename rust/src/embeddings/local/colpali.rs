use std::collections::HashMap;

use anyhow::Error as E;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{colpali::Model, paligemma};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::embeddings::embed::{EmbedData, EmbedImage};

pub struct ColPaliEmbedder {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub config: paligemma::Config,
    dummy_input: Tensor,
}

impl ColPaliEmbedder {
    pub fn new(model_id: &str, revision: Option<&str>) -> Result<Self, anyhow::Error> {
        let api = hf_hub::api::sync::Api::new()?;
        let api = match revision {
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

        let (config_filename, tokenizer_filename, weights_filename) = {
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
        let config: paligemma::Config = serde_json::from_str(&config)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        let trunc = TruncationParams {
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            max_length: config.text_config.max_position_embeddings as usize,
            ..Default::default()
        };

        tokenizer
            .with_padding(Some(pp))
            .with_truncation(Some(trunc))
            .unwrap();

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let vb = if weights_filename.ends_with("model.safetensors") {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, &device).unwrap()
            }
        } else {
            println!("Loading weights from pytorch_model.bin");
            VarBuilder::from_pth(&weights_filename, dtype, &device).unwrap()
        };

        let model = Model::new(&config, vb)?;
        let dummy_prompt: &str = "Describe the image";

        let dummy_input = tokenize_batch(&tokenizer, vec![dummy_prompt], &device)?;

        Ok(Self {
            model,
            tokenizer,
            config,
            dummy_input,
        })
    }

    fn load_image<T: AsRef<std::path::Path>>(
        &self,
        path: T,
        image_size: usize,
    ) -> anyhow::Result<Tensor> {
        let img = image::ImageReader::open(path)?.decode()?;
        let (height, width) = (image_size, image_size);
        let img = img.resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Triangle,
        );

        let img = img.to_rgb8();

        let img = img.into_raw();
        let img = Tensor::from_vec(
            img,
            (height, width, 3),
            &Device::cuda_if_available(0).unwrap_or(Device::Cpu),
        )?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?;
        // .unsqueeze(0)?;
        Ok(img)
    }
}

impl EmbedImage for ColPaliEmbedder {
    fn embed_image<T: AsRef<std::path::Path>>(
        &self,
        image_path: T,
        metadata: Option<HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData> {
        let pixel_values = self
            .load_image(image_path, self.config.vision_config.image_size)?
            .unsqueeze(0)?;
        let encoding = self
            .model
            .forward_images(&pixel_values, &self.dummy_input)?.to_vec3::<f32>();

        Ok(EmbedData::new(encoding.to_vec(), None, metadata.clone()))
    }
}

fn tokenize_batch(
    tokenizer: &Tokenizer,
    text_batch: Vec<&str>,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let tokens = tokenizer.encode_batch(text_batch, true).map_err(E::msg)?;
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;

    Ok(Tensor::stack(&token_ids, 0)?)
}
