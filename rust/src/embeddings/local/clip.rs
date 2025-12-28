#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::{collections::HashMap, fs};

use anyhow::Error as E;

use crate::{
    embeddings::{embed::EmbeddingResult, select_device},
    models::{
        clip::div_l2_norm,
        clip::{self, ClipConfig},
        siglip::{self, Config},
    },
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::dinov2::DinoVisionTransformer;
use tokenizers::{PaddingParams, Tokenizer};

use crate::embeddings::embed::{EmbedData, EmbedImage};

pub enum VisionModel {
    Clip(clip::ClipModel),
    Siglip(siglip::Model),
    Dino(DinoVisionTransformer),
}

impl VisionModel {
    pub fn get_image_features(&self, image: &Tensor) -> Result<Tensor, candle_core::Error> {
        match self {
            VisionModel::Clip(model) => model.get_image_features(image),
            VisionModel::Siglip(model) => model.get_image_features(image),
            VisionModel::Dino(model) => model.forward(image),
        }
    }

    pub fn get_text_features(&self, text: &Tensor) -> Result<Tensor, candle_core::Error> {
        match self {
            VisionModel::Clip(model) => model.get_text_features(text),
            VisionModel::Siglip(model) => model.get_text_features(text),
            VisionModel::Dino(_) => Err(candle_core::Error::msg(
                "Dino model does not support text features",
            )),
        }
    }

    pub fn supports_text(&self) -> bool {
        match self {
            VisionModel::Clip(_) => true,
            VisionModel::Siglip(_) => true,
            VisionModel::Dino(_) => false,
        }
    }
}

pub struct ClipEmbedder {
    pub model: VisionModel,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub max_len: usize,
    pub pad_id: u32,
}
impl Default for ClipEmbedder {
    fn default() -> Self {
        Self::new(
            "openai/clip-vit-base-patch32".to_string(),
            Some("refs/pr/15"),
            None,
        )
        .unwrap()
    }
}

impl ClipEmbedder {
    pub fn new(model_id: String, revision: Option<&str>, token: Option<&str>) -> Result<Self, E> {
        let api = hf_hub::api::sync::ApiBuilder::from_env()
            .with_token(token.map(|s| s.to_string()))
            .build()?;

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

        let device = select_device();

        let vb = match api.get("model.safetensors") {
            Ok(safetensors) => unsafe {
                VarBuilder::from_mmaped_safetensors(&[safetensors], DType::F32, &device)?
            },
            Err(_) => match api.get("pytorch_model.bin") {
                Ok(pytorch_model) => VarBuilder::from_pth(pytorch_model, DType::F32, &device)?,
                Err(e) => {
                    return Err(anyhow::Error::msg(format!(
                        "Model weights not found. The weights should either be a `model.safetensors` or `pytorch_model.bin` file.  Error: {}",
                        e
                    )));
                }
            },
        };
        let config_filename = api.get("config.json")?;
        let config_str = std::fs::read_to_string(config_filename)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;

        let mut tokenizer = Self::get_tokenizer(None, model_id, revision)?;
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };

        tokenizer.with_padding(Some(pp));

        let (model, max_len, pad_id) = if let Some(architectures) = config_json.get("architectures")
        {
            if let Some(arch) = architectures.get(0) {
                match arch.as_str() {
                    Some("CLIPModel") => {
                        let config: ClipConfig = serde_json::from_str(&config_str)?;
                        let vocab = tokenizer.get_vocab(true);
                        let pad_id = *vocab.get("<|endoftext|>").ok_or(E::msg("No pad token"))?;
                        (
                            VisionModel::Clip(clip::ClipModel::new(vb, &config)?),
                            config.text_config.max_position_embeddings,
                            pad_id,
                        )
                    }
                    Some("SiglipModel") => {
                        let config: Config = serde_json::from_str(&config_str)?;
                        (
                            VisionModel::Siglip(siglip::Model::new(&config, vb)?),
                            config.text_config.max_position_embeddings,
                            config.text_config.pad_token_id as u32,
                        )
                    }
                    Some("Dinov2Model") => {
                        let (depth, embed_dim, num_heads) = (
                            config_json
                                .get("num_hidden_layers")
                                .expect("num_hidden_layers not found")
                                .as_u64()
                                .expect("num_hidden_layers not a number")
                                as usize,
                            config_json
                                .get("hidden_size")
                                .expect("hidden_size not found")
                                .as_u64()
                                .expect("hidden_size not a number")
                                as usize,
                            config_json
                                .get("num_attention_heads")
                                .expect("num_attention_heads not found")
                                .as_u64()
                                .expect("num_attention_heads not a number")
                                as usize,
                        );
                        (
                            VisionModel::Dino(DinoVisionTransformer::new(
                                vb,
                                depth as usize,
                                embed_dim as usize,
                                num_heads as usize,
                            )?),
                            0,
                            0,
                        )
                    }
                    _ => return Err(anyhow::Error::msg("Unsupported model architecture")),
                }
            } else {
                return Err(anyhow::Error::msg("No architecture specified in config"));
            }
        } else {
            let config: Config = serde_json::from_str(&config_str)?;
            (
                VisionModel::Siglip(siglip::Model::new(&config, vb)?),
                config.text_config.max_position_embeddings,
                config.text_config.pad_token_id as u32,
            )
        };

        Ok(ClipEmbedder {
            model,
            tokenizer,
            device,
            max_len,
            pad_id,
        })
    }

    pub fn get_tokenizer(
        tokenizer: Option<String>,
        model_id: String,
        revision: Option<&str>,
    ) -> anyhow::Result<Tokenizer> {
        let tokenizer = match tokenizer {
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = match revision {
                    Some(rev) => api.repo(hf_hub::Repo::with_revision(
                        model_id,
                        hf_hub::RepoType::Model,
                        rev.to_string(),
                    )),
                    None => api.repo(hf_hub::Repo::new(model_id, hf_hub::RepoType::Model)),
                };
                api.get("tokenizer.json")?
            }
            Some(file) => file.into(),
        };

        Tokenizer::from_file(tokenizer).map_err(E::msg)
    }

    pub fn tokenize_sequences(
        &self,
        sequences: Option<&[&str]>,
        tokenizer: &Tokenizer,
    ) -> anyhow::Result<(Tensor, Vec<String>)> {
        let pad_id = self.pad_id;

        let vec_seq = sequences.unwrap_or(&[
            "a cycling race",
            "a photo of two cats",
            "a robot holding a candle",
        ]);

        let mut tokens = vec![];

        for seq in vec_seq {
            let encoding = tokenizer.encode(*seq, true).map_err(E::msg)?;
            tokens.push(encoding.get_ids().to_vec());
        }

        let max_len = self.max_len;

        // Pad the sequences to have the same length
        for token_vec in tokens.iter_mut() {
            let len_diff = max_len - token_vec.len();
            if len_diff > 0 {
                token_vec.extend(vec![pad_id; len_diff]);
            }
        }

        let input_ids = Tensor::new(tokens, &self.device)?;

        Ok((input_ids, vec_seq.iter().map(|s| s.to_string()).collect()))
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
        let img = Tensor::from_vec(img, (height, width, 3), &self.device)?
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

    pub fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let mut encodings = Vec::new();

        if !self.model.supports_text() {
            return Err(anyhow::Error::msg(
                "This model does not support text features",
            ));
        }

        let batch_size = batch_size.unwrap_or(32);

        for mini_text_batch in text_batch.chunks(batch_size) {
            let (input_ids, _vec_seq) = self
                .tokenize_sequences(Some(mini_text_batch), &self.tokenizer)
                .unwrap();

            let batch_encodings = self.model.get_text_features(&input_ids)?;
            let normalized_encodings = div_l2_norm(&batch_encodings)?.to_vec2::<f32>()?;
            encodings.extend(
                normalized_encodings
                    .iter()
                    .map(|embedding| EmbeddingResult::DenseVector(embedding.to_vec())),
            );
        }

        Ok(encodings)
    }
}

impl EmbedImage for ClipEmbedder {
    async fn embed_image_batch<T: AsRef<std::path::Path>>(
        &self,
        image_paths: &[T],
        batch_size: Option<usize>,
    ) -> anyhow::Result<Vec<EmbedData>> {
        let config = clip::ClipConfig::vit_base_patch32();

        let mut encodings = Vec::new();
        for image_batch in image_paths.chunks(batch_size.unwrap_or(32)) {
            let images = self
                .load_images(image_batch, config.vision_config.image_size)
                .unwrap();
            let batch_encodings = self.model.get_image_features(&images)?;
            let normalized_encodings = div_l2_norm(&batch_encodings)?.to_vec2::<f32>()?;
            encodings.extend(normalized_encodings);
        }

        let embeddings = encodings
            .iter()
            .zip(image_paths)
            .map(|(data, path)| {
                let mut metadata = HashMap::new();
                metadata.insert(
                    "file_name".to_string(),
                    fs::canonicalize(path)
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string(),
                );

                EmbedData::new(
                    EmbeddingResult::DenseVector(data.to_vec()),
                    Some(path.as_ref().to_str().unwrap().to_string()),
                    Some(metadata),
                )
            })
            .collect::<Vec<_>>();
        Ok(embeddings)
    }

    async fn embed_image<T: AsRef<std::path::Path>>(
        &self,
        image_path: T,
        metadata: Option<HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData> {
        let config = clip::ClipConfig::vit_base_patch32();
        let image = self
            .load_image(&image_path, config.vision_config.image_size)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let encoding = &self.model.get_image_features(&image)?;
        let normalized_encoding = &div_l2_norm(encoding)?.to_vec2::<f32>()?[0];
        Ok(EmbedData::new(
            EmbeddingResult::DenseVector(normalized_encoding.to_vec()),
            None,
            metadata.clone(),
        ))
    }

    async fn embed_pdf<T: AsRef<std::path::Path>>(
        &self,
        _pdf_path: T,
        _batch_size: Option<usize>,
    ) -> anyhow::Result<Vec<EmbedData>> {
        Err(anyhow::anyhow!(
            "PDF embedding not supported for Clip model"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests the tokenize_sequences method.
    #[test]
    fn test_tokenize_sequences() {
        let clip_embedder = ClipEmbedder::default();
        let sequences = &["Hey there how are you?", "EmbedAnything is the best!"];
        let (input_ids, vec_seq) = clip_embedder
            .tokenize_sequences(Some(sequences), &clip_embedder.tokenizer)
            .unwrap();
        assert_eq!(
            vec_seq,
            vec![
                "Hey there how are you?".to_string(),
                "EmbedAnything is the best!".to_string(),
            ]
        );
        assert_eq!(input_ids.shape().clone().into_dims(), &[2, 77]);
    }

    // Tests the load_image method.
    #[test]
    fn test_load_image() {
        let clip_embedder = ClipEmbedder::default();
        let image = clip_embedder
            .load_image("../test_files/clip/cat1.jpg", 224)
            .unwrap();
        assert_eq!(image.shape().clone().into_dims(), &[3, 224, 224]);
    }

    // Tests the load_images method.
    #[test]
    fn test_load_images() {
        let clip_embedder = ClipEmbedder::default();
        let images = clip_embedder
            .load_images(
                &[
                    "../test_files/clip/cat1.jpg",
                    "../test_files/clip/cat2.jpeg",
                ],
                224,
            )
            .unwrap();
        assert_eq!(images.shape().clone().into_dims(), &[2, 3, 224, 224]);
    }

    // Tests the embed_image_batch method.
    #[tokio::test]
    async fn test_embed_image_batch() {
        let clip_embedder = ClipEmbedder::default();
        let embeddings = clip_embedder
            .embed_image_batch(
                &[
                    "../test_files/clip/cat1.jpg",
                    "../test_files/clip/cat2.jpeg",
                ],
                Some(2),
            )
            .await
            .unwrap();
        assert_eq!(embeddings.len(), 2);
    }
}
