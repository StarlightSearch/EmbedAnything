use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

use base64::Engine;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use image::ImageFormat;

use crate::embeddings::embed::{EmbedData, EmbeddingResult};
use crate::embeddings::local::colpali::{get_images_from_pdf, ColPaliEmbed};
use crate::embeddings::select_device;
use crate::models::idefics3::model::{ColIdefics3Model, Idefics3Config};
use crate::models::idefics3::tensor_processing::Idefics3Processor;

pub struct ColSmolEmbedder {
    pub model: RwLock<ColIdefics3Model>,
    pub processor: Idefics3Processor,
    pub device: Device,
}

impl ColSmolEmbedder {
    pub fn new(model_id: &str, revision: Option<&str>) -> Result<Self, anyhow::Error> {
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

        let model_file = repo.get("model.safetensors")?;
        let device = select_device();
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };
        let config_file = repo.get("config.json")?;

        let processor = Idefics3Processor::from_pretrained("akshayballal/colSmol-256M-merged")?;
        let config: Idefics3Config = serde_json::from_slice(&std::fs::read(config_file)?)?;

        let model = ColIdefics3Model::load(&config, false, vb)?;

        Ok(Self {
            model: RwLock::new(model),
            processor,
            device,
        })
    }
}

impl ColPaliEmbed for ColSmolEmbedder {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let mut encodings = Vec::new();
        for mini_text_batch in text_batch.chunks(batch_size.unwrap_or(32)) {
            let (input_ids, attention_mask) = self
                .processor
                .tokenize_batch(mini_text_batch.to_vec(), &self.device)?;
            let batch_encodings = self
                .model
                .write()
                .map_err(|e| anyhow::anyhow!("{}", e))?
                .forward(&input_ids, &attention_mask, &None, &None)?
                .to_dtype(DType::F32)?;

            encodings.extend(
                batch_encodings
                    .to_vec3::<f32>()?
                    .iter()
                    .map(|x| EmbeddingResult::MultiVector(x.to_vec())),
            );
        }
        Ok(encodings)
    }

    fn embed_query(&self, query: &str) -> anyhow::Result<Vec<EmbedData>> {
        let (input_ids, attention_mask) =
            self.processor.tokenize_batch(vec![query], &self.device)?;

        let encoding = self
            .model
            .write()
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .forward(&input_ids, &attention_mask, &None, &None)?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?
            .into_iter()
            .map(|x| EmbeddingResult::MultiVector(x.to_vec()));

        Ok(encoding
            .map(|x| EmbedData::new(x.clone(), None, None))
            .collect::<Vec<_>>())
    }

    fn embed_image(
        &self,
        image_path: PathBuf,
        metadata: Option<HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData> {
        let image = image::open(image_path)?;
        let (input_ids, attention_mask, pixel_values, pixel_attention_mask) =
            self.processor.preprocess(&[image], &self.device)?;
        let encoding = self
            .model
            .write()
            .unwrap()
            .forward(
                &input_ids,
                &attention_mask,
                &Some(pixel_values),
                &pixel_attention_mask,
            )?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?
            .into_iter()
            .map(|x| EmbeddingResult::MultiVector(x.to_vec()))
            .collect::<Vec<_>>();

        Ok(EmbedData::new(encoding[0].clone(), None, metadata))
    }

    fn embed_image_batch(&self, image_paths: &[PathBuf]) -> anyhow::Result<Vec<EmbedData>> {
        let images = image_paths
            .iter()
            .map(image::open)
            .collect::<Result<Vec<_>, _>>()?;
        let (input_ids, attention_mask, pixel_values, pixel_attention_mask) =
            self.processor.preprocess(&images, &self.device)?;
        let encodings = self
            .model
            .write()
            .unwrap()
            .forward(
                &input_ids,
                &attention_mask,
                &Some(pixel_values),
                &pixel_attention_mask,
            )?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?;

        Ok(encodings
            .into_iter()
            .map(|x| EmbedData::new(EmbeddingResult::MultiVector(x), None, None))
            .collect::<Vec<_>>())
    }
    fn embed_file(&self, file_path: PathBuf, batch_size: usize) -> anyhow::Result<Vec<EmbedData>> {
        let pages = get_images_from_pdf(&file_path)?;
        let mut embed_data = Vec::new();
        for (index, batch) in pages.chunks(batch_size).enumerate() {
            let start_page = index * batch_size + 1;
            let end_page = start_page + batch.len();
            let page_numbers = (start_page..=end_page).collect::<Vec<_>>();
            let (input_ids, attention_mask, pixel_values, pixel_attention_mask) =
                self.processor.preprocess(batch, &self.device)?;

            let image_embeddings = self
                .model
                .write()
                .map_err(|e| anyhow::anyhow!("{}", e))?
                .forward(
                    &input_ids,
                    &attention_mask,
                    &Some(pixel_values),
                    &pixel_attention_mask,
                )?
                .to_dtype(DType::F32)?
                .to_vec3::<f32>()?
                .into_iter()
                .map(|x| EmbeddingResult::MultiVector(x.to_vec()));

            // zip the embeddings with the page numbers
            let embed_data_batch = image_embeddings
                .zip(page_numbers.into_iter())
                .zip(batch.iter())
                .map(|((embedding, page_number), page_image)| {
                    let mut metadata = HashMap::new();

                    let mut buf = Vec::new();
                    let mut cursor = std::io::Cursor::new(&mut buf);
                    page_image.write_to(&mut cursor, ImageFormat::Png).unwrap();
                    let engine = base64::engine::general_purpose::STANDARD;
                    let base64_image = engine.encode(&buf);

                    metadata.insert("page_number".to_string(), page_number.to_string());
                    metadata.insert(
                        "file_path".to_string(),
                        file_path.to_str().unwrap_or("").to_string(),
                    );
                    metadata.insert("image".to_string(), base64_image);
                    EmbedData::new(embedding, None, Some(metadata))
                });
            embed_data.extend(embed_data_batch);
        }
        Ok(embed_data)
    }
}
