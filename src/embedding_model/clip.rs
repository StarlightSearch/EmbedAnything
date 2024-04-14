#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Error as E;


use candle_core::{DType, Device, Tensor};
use candle_transformers::models::clip;

use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use super::embed::{Embed, EmbedData, EmbedImage};

pub struct ClipEmbeder {
    pub model: clip::ClipModel,
    pub tokenizer: Tokenizer,
}

impl Default for ClipEmbeder {
    fn default() -> Self {
        let api = hf_hub::api::sync::Api::new().unwrap();
        let api = api.repo(hf_hub::Repo::with_revision(
            "openai/clip-vit-base-patch32".to_string(),
            hf_hub::RepoType::Model,
            "refs/pr/15".to_string(),
        ));
        let model_file = api.get("model.safetensors").unwrap();
        let config = clip::ClipConfig::vit_base_patch32();
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_file.clone()], DType::F32, &device).unwrap()
        };
        let model = clip::ClipModel::new(vb, &config).unwrap();
        let tokenizer = Self::get_tokenizer(None).unwrap();
        ClipEmbeder { model, tokenizer }
    }
}

impl ClipEmbeder {
    pub fn new(model: clip::ClipModel, tokenizer: Tokenizer) -> Self {
        ClipEmbeder { model, tokenizer }
    }

    pub fn get_tokenizer(tokenizer: Option<String>) -> anyhow::Result<Tokenizer> {
        let tokenizer = match tokenizer {
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.repo(hf_hub::Repo::with_revision(
                    "openai/clip-vit-base-patch32".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/15".to_string(),
                ));
                api.get("tokenizer.json")?
            }
            Some(file) => file.into(),
        };

        Tokenizer::from_file(tokenizer).map_err(E::msg)
    }

    pub fn tokenize_sequences(
        sequences: Option<Vec<String>>,
        tokenizer: &Tokenizer,
        device: &Device,
    ) -> anyhow::Result<(Tensor, Vec<String>)> {
        let pad_id = *tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .ok_or(E::msg("No pad token"))?;

        let vec_seq = match sequences {
            Some(seq) => seq,
            None => vec![
                "a cycling race".to_string(),
                "a photo of two cats".to_string(),
                "a robot holding a candle".to_string(),
            ],
        };

        let mut tokens = vec![];

        for seq in vec_seq.clone() {
            let encoding = tokenizer.encode(seq, true).map_err(E::msg)?;
            tokens.push(encoding.get_ids().to_vec());
        }

        let max_len = tokens.iter().map(|v| v.len()).max().unwrap_or(0);

        // Pad the sequences to have the same length
        for token_vec in tokens.iter_mut() {
            let len_diff = max_len - token_vec.len();
            if len_diff > 0 {
                token_vec.extend(vec![pad_id; len_diff]);
            }
        }

        let input_ids = Tensor::new(tokens, device)?;

        Ok((input_ids, vec_seq))
    }

    fn load_image<T: AsRef<std::path::Path>>(&self, path: T, image_size: usize) -> anyhow::Result<Tensor> {
        let img = image::io::Reader::open(path)?.decode()?;
        let (height, width) = (image_size, image_size);
        let img = img.resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Triangle,
        );
    
        let img = img.to_rgb8();
    
        let img = img.into_raw();
        let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
            .permute((2, 0, 1))?
            .to_dtype(DType::F32)?
            .affine(2. / 255., -1.)?;
        // .unsqueeze(0)?;
        Ok(img)
    }
    
    fn load_images<T: AsRef<std::path::Path>>(
        &self, 
        paths: &[T],
        image_size: usize,
    ) -> anyhow::Result<Tensor> {
        let mut images = vec![];
    
        for path in paths {
            let tensor = self.load_image(path, image_size)?;
            images.push(tensor);
        }
    
        let images = Tensor::stack(&images, 0)?;
    
        Ok(images)
    }

   


}

impl EmbedImage for ClipEmbeder{
    fn embed_image_batch<T: AsRef<std::path::Path>>(&self, image_paths:&[T]) -> anyhow::Result<Vec<EmbedData>> {
        let config = clip::ClipConfig::vit_base_patch32();

        let images = self.load_images(image_paths, config.image_size).unwrap();
        let encodings = self.model.get_image_features(&images).unwrap().to_vec2::<f32>().unwrap();

        let embeddings = encodings
            .iter()
            .zip(image_paths)
            .map(|(data, path)| EmbedData::new(data.to_vec(), Some(path.as_ref().to_str().unwrap().to_string())))
            .collect::<Vec<_>>();
        Ok(embeddings)

    }
}

impl Embed for ClipEmbeder {
    async fn embed(&self, text_batch: &[String]) -> Result<Vec<EmbedData>, reqwest::Error> {
        let (input_ids, _vec_seq) = ClipEmbeder::tokenize_sequences(
            Some(text_batch.to_vec()),
            &self.tokenizer,
            &Device::Cpu,
        )
        .unwrap();

        let encodings = self
            .model
            .get_text_features(&input_ids)
            .unwrap()
            .to_vec2::<f32>().unwrap();
        let embeddings = encodings
            .iter()
            .zip(text_batch)
            .map(|(data, text)| EmbedData::new(data.to_vec(), Some(text.clone())))
            .collect::<Vec<_>>();
        Ok(embeddings)
    }
}

