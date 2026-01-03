#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::{collections::HashMap, fs};

use anyhow::Error as E;

use crate::{
    embeddings::{embed::EmbeddingResult, select_device},
    models::clip::div_l2_norm,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::embeddings::embed::{EmbedData, EmbedImage};
use crate::models::dinov2::DinoVisionTransformer;

pub enum VisionEncoderModel {
    Dino(DinoVisionTransformer),
}

impl VisionEncoderModel {
    pub fn get_image_features(&self, image: &Tensor) -> Result<Tensor, candle_core::Error> {
        match self {
            VisionEncoderModel::Dino(model) => model.forward(image),
        }
    }
}

pub struct VisionEncoderEmbedder {
    pub model: VisionEncoderModel,
    pub device: Device,
    pub crop_size: usize,
}
impl Default for VisionEncoderEmbedder {
    fn default() -> Self {
        Self::new("facebook/dinov2-small", None, None).unwrap()
    }
}

impl VisionEncoderEmbedder {
    pub fn new(model_id: &str, revision: Option<&str>, token: Option<&str>) -> Result<Self, E> {
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

        let preprocessor_config_filename = api.get("preprocessor_config.json")?;
        let preprocessor_config_str = std::fs::read_to_string(preprocessor_config_filename)?;
        let preprocessor_config_json: serde_json::Value =
            serde_json::from_str(&preprocessor_config_str)?;
        let crop_size = preprocessor_config_json
            .get("crop_size")
            .expect("crop_size not found")
            .get("height")
            .expect("height not found")
            .as_u64()
            .expect("height not a number") as usize;

        let model = if let Some(architectures) = config_json.get("architectures") {
            if let Some(arch) = architectures.get(0) {
                match arch.as_str() {
                    Some("Dinov2Model") => {
                        let (depth, embed_dim, num_heads, img_size, patch_size) = (
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
                            config_json
                                .get("image_size")
                                .expect("image_size not found")
                                .as_u64()
                                .expect("image_size not a number")
                                as usize,
                            config_json
                                .get("patch_size")
                                .expect("patch_size not found")
                                .as_u64()
                                .expect("patch_size not a number")
                                as usize,
                        );

                        VisionEncoderModel::Dino(DinoVisionTransformer::new(
                            vb,
                            depth as usize,
                            embed_dim as usize,
                            num_heads as usize,
                            img_size as usize,
                            patch_size as usize,
                        )?)
                    }
                    _ => return Err(anyhow::Error::msg("Unsupported model architecture")),
                }
            } else {
                return Err(anyhow::Error::msg("No architecture specified in config"));
            }
        } else {
            VisionEncoderModel::Dino(DinoVisionTransformer::new(vb, 12, 384, 6, 518, 14)?)
        };
        Ok(VisionEncoderEmbedder {
            model,
            device,
            crop_size,
        })
    }

    fn load_image<T: AsRef<std::path::Path>>(&self, path: T) -> anyhow::Result<Tensor> {
        let img = image::ImageReader::open(path)?.decode()?;
        let (height, width) = (self.crop_size, self.crop_size);
        let img = img.resize_exact(
            width as u32,
            height as u32,
            image::imageops::FilterType::CatmullRom,
        );

        let img = img.to_rgb8();

        let img = img.into_raw();
        let img = Tensor::from_vec(img, (height, width, 3), &self.device)?
            .permute((2, 0, 1))?
            .to_dtype(DType::F32)?
            .affine(0.00392156862745098, 0.)?; // Rescale factor: 1/255
                                               // ImageNet normalization: (image - mean) / std
        let mean = Tensor::new(&[0.485f32, 0.456f32, 0.406f32], &self.device)?
            .unsqueeze(1)?
            .unsqueeze(2)?; // Shape: [3, 1, 1]
        let std = Tensor::new(&[0.229f32, 0.224f32, 0.225f32], &self.device)?
            .unsqueeze(1)?
            .unsqueeze(2)?; // Shape: [3, 1, 1]

        let img = img.broadcast_sub(&mean)?.broadcast_div(&std)?;
        // .unsqueeze(0)?;
        Ok(img)
    }

    fn load_images<T: AsRef<std::path::Path>>(&self, paths: &[T]) -> anyhow::Result<Tensor> {
        let mut images = vec![];

        for path in paths {
            let tensor = self.load_image(path)?;
            images.push(tensor);
        }

        let images = Tensor::stack(&images, 0)?;

        Ok(images)
    }
}

impl EmbedImage for VisionEncoderEmbedder {
    async fn embed_image_batch<T: AsRef<std::path::Path>>(
        &self,
        image_paths: &[T],
        batch_size: Option<usize>,
    ) -> anyhow::Result<Vec<EmbedData>> {
        let mut encodings = Vec::new();
        for image_batch in image_paths.chunks(batch_size.unwrap_or(32)) {
            let images = self.load_images(image_batch).unwrap();
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
        let image = self.load_image(&image_path).unwrap().unsqueeze(0).unwrap();
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

    // Tests the load_image method.
    #[test]
    fn test_load_image() {
        let vision_encoder_embedder = VisionEncoderEmbedder::default();
        let image = vision_encoder_embedder
            .load_image("../test_files/clip/cat3.jpg")
            .unwrap();
        assert_eq!(image.shape().clone().into_dims(), &[3, 224, 224]);
    }

    // Tests the load_images method.
    #[test]
    fn test_load_images() {
        let vision_encoder_embedder = VisionEncoderEmbedder::default();
        let images = vision_encoder_embedder
            .load_images(&[
                "../test_files/clip/cat1.jpg",
                "../test_files/clip/cat2.jpeg",
            ])
            .unwrap();
        assert_eq!(images.shape().clone().into_dims(), &[2, 3, 224, 224]);
    }

    // Tests the embed_image_batch method.
    #[tokio::test]
    async fn test_embed_image_batch() {
        let vision_encoder_embedder = VisionEncoderEmbedder::default();
        let embeddings = vision_encoder_embedder
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
