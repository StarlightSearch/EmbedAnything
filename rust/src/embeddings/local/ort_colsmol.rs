use std::sync::RwLock;
use std::{collections::HashMap, path::PathBuf};

use crate::embeddings::embed::{EmbedData, EmbeddingResult};
use crate::embeddings::local::colpali::ColPaliEmbed;
use crate::models::idefics3::array_processing::Idefics3Processor;
use crate::models::paligemma;
use anyhow::Error as E;
use base64::Engine;
use half::f16;
use image::ImageFormat;
use ndarray::prelude::*;
use ort::execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use rayon::prelude::*;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use super::colpali::get_images_from_pdf;


pub struct OrtColSmolEmbedder {
    pub model: RwLock<Session>,
    pub tokenizer: Tokenizer,
    pub image_size: usize,
    pub num_channels: usize,
    pub processor: Idefics3Processor,
    dummy_input: Array2<i64>,
}

impl OrtColSmolEmbedder {
    pub fn new(
        model_id: &str,
        revision: Option<&str>,
        path_in_repo: Option<&str>,
    ) -> Result<Self, E> {
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

        let mut path_in_repo = path_in_repo.unwrap_or_default().to_string();
        if !path_in_repo.is_empty() {
            path_in_repo.push('/');
        }
        let (_, tokenizer_filename, weights_filename, _, processing_config_filename,) = {
            let config = repo
                .get("config.json")
                .unwrap_or(repo.get("preprocessor_config.json")?);
            let tokenizer = repo.get("tokenizer.json")?;
            let weights = repo.get(format!("{}model.onnx", path_in_repo).as_str())?;
            let data = repo
                .get(format!("{}model.onnx_data", path_in_repo).as_str())
                .ok();
            let processing_config = repo.get("preprocessor_config.json")?;

            (config, tokenizer, weights, data, processing_config)
        };

        let config: paligemma::Config = paligemma::Config::paligemma_3b_448();
        let image_size = config.vision_config.image_size;
        let num_channels = config.vision_config.num_channels;
        
        let mut tokenizer = Tokenizer::from_pretrained(model_id, None).map_err(E::msg)?;

        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        let trunc = TruncationParams {
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            max_length: config.text_config.max_position_embeddings,
            ..Default::default()
        };

        let processor: Idefics3Processor = Idefics3Processor::from_pretrained(model_id)?;

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

        // Get physical core count (excluding hyperthreading)
        let threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        // For CPU-bound workloads like ONNX inference, it's often better to use
        // physical cores rather than logical cores to avoid context switching overhead
        let optimal_threads = std::cmp::max(1, threads / 2);

        let model = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
                CoreMLExecutionProvider::default().build(),
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(optimal_threads)? // Use optimal thread count
            .with_inter_threads(1)? // Set inter-op parallelism to 1 when using GPU
            .commit_from_file(weights_filename)?;

        let dummy_prompt: &str = "<|im_start|>user\n<image>Describe the image.<end_of_utterance>";
        let dummy_input = tokenize(&tokenizer, dummy_prompt.to_string())?;

        Ok(Self {
            model: RwLock::new(model),
            tokenizer,
            image_size,
            processor,
            num_channels,
            dummy_input,
        })
    }
}

fn tokenize_batch(tokenizer: &Tokenizer, text_batch: &[&str]) -> Result<Array2<i64>, E> {
    let token_ids = tokenizer
        .encode_batch_fast(text_batch.to_vec(), true)
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

fn tokenize(tokenizer: &Tokenizer, text: String) -> Result<Array2<i64>, E> {
    let token_ids = tokenizer.encode(text, false).map_err(E::msg)?;
    let token_ids_array = Array2::from_shape_vec(
        (1, token_ids.len()),
        token_ids
            .get_ids()
            .iter()
            .map(|x| *x as i64)
            .collect::<Vec<i64>>(),
    )
    .unwrap();
    Ok(token_ids_array)
}

fn get_attention_mask(tokenizer: &Tokenizer, text_batch: &[&str]) -> Result<Array2<i64>, E> {
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

impl OrtColSmolEmbedder {
    fn run_model(
        &self,
        token_ids: Array2<i64>,
        attention_mask: Array2<i64>,
        pixel_values: Array5<f32>,
        pixel_attention_mask: Array4<i64>,
    ) -> Result<Vec<EmbeddingResult>, E> {
        let mut model_guard = self.model.write().unwrap();
        let output_name = model_guard.outputs.first().unwrap().name.to_string();
        println!("input names: {:?}", model_guard.inputs);

        let token_ids_tensor = ort::value::Value::from_array(token_ids)?;
        let pixel_values_tensor = ort::value::Value::from_array(pixel_values)?;
        let attention_mask_tensor = ort::value::Value::from_array(attention_mask)?;
        let pixel_attention_mask_tensor = ort::value::Value::from_array(pixel_attention_mask)?;
        let outputs = model_guard.run(ort::inputs!["input_ids" => token_ids_tensor, "pixel_values" => pixel_values_tensor, "attention_mask" => attention_mask_tensor, "pixel_attention_mask" => pixel_attention_mask_tensor])?;

        let embeddings = outputs[output_name]
            .try_extract_array::<f16>()?
            .to_owned()
            .into_dimensionality::<ndarray::Ix3>()?;

        let e = embeddings
            .outer_iter()
            .map(|row| {
                EmbeddingResult::MultiVector(
                    row.outer_iter()
                        .map(|x| x.to_vec())
                        .map(|y| y.into_iter().map(|z| z.to_f32()).collect::<Vec<f32>>())
                        .collect(),
                )
            })
            .collect();

        Ok(e)
    }
}

impl ColPaliEmbed for OrtColSmolEmbedder {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let batch_size = batch_size.unwrap_or(32);
        let encodings = text_batch
            .par_chunks(batch_size)
            .flat_map(|mini_text_batch| -> Result<Vec<EmbeddingResult>, E> {
                let token_ids: Array2<i64> = tokenize_batch(&self.tokenizer, mini_text_batch)?;
                let attention_mask: Array2<i64> =
                    get_attention_mask(&self.tokenizer, mini_text_batch)?;
                let pixel_values: Array5<f32> = Array5::zeros((
                    mini_text_batch.len(),
                    1,
                    self.num_channels,
                    self.image_size,
                    self.image_size,
                ));
                let pixel_attention_mask: Array4<i64> = Array4::ones((
                    mini_text_batch.len(),
                    self.num_channels,
                    self.image_size,
                    self.image_size,
                ));
                let e = self.run_model(
                    token_ids,
                    attention_mask,
                    pixel_values,
                    pixel_attention_mask,
                )?;
                Ok(e)
            })
            .flatten()
            .collect::<Vec<_>>();

        Ok(encodings)
    }

    fn embed_query(&self, query: &str) -> anyhow::Result<Vec<EmbedData>> {
        let token_ids = tokenize_batch(&self.tokenizer, &[query])?;
        let attention_mask = get_attention_mask(&self.tokenizer, &[query])?;
        let pixel_values: Array5<f32> =
            Array5::zeros((1, 1, self.num_channels, self.image_size, self.image_size));
        let pixel_attention_mask: Array4<i64> =
            Array4::ones((1, self.num_channels, self.image_size, self.image_size));
        let e = self
            .run_model(
                token_ids,
                attention_mask,
                pixel_values,
                pixel_attention_mask,
            )?
            .into_iter()
            .map(|x| EmbedData::new(x, None, None))
            .collect::<Vec<_>>();
        Ok(e)
    }

    fn embed_file(&self, file_path: PathBuf, batch_size: usize) -> anyhow::Result<Vec<EmbedData>> {
        let pages = get_images_from_pdf(&file_path)?;
        let mut embed_data = Vec::new();
        for (index, batch) in pages.chunks(1).enumerate() {
            let start_page = index * batch_size + 1;
            let end_page = start_page + batch.len();
            let page_numbers = (start_page..=end_page).collect::<Vec<_>>();
            let (input_ids, attention_mask, page_images, pixel_attention_mask) = self.processor.preprocess(&batch)?;

            let image_embeddings = self.run_model(
                input_ids,
                attention_mask,
                page_images.insert_axis(Axis(0)),
                pixel_attention_mask
                    .unwrap()
                    .insert_axis(Axis(0))
                    .mapv(|x| x as i64),
            )?;
            // zip the embeddings with the page numbers
            let embed_data_batch = image_embeddings
                .into_iter()
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

    fn embed_image(
        &self,
        image_path: PathBuf,
        metadata: Option<std::collections::HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData> {
        let image = image::open(&image_path)?;
        let (input_ids, attention_mask, image_array, pixel_attention_mask) = self.processor.preprocess(&[image])?;
   
        println!("image array shape: {:?}", image_array.shape());
        println!("input ids {:?}", input_ids);
        let e = self
            .run_model(
                input_ids,
                attention_mask,
                image_array.insert_axis(Axis(0)),
                pixel_attention_mask
                    .unwrap()
                    .insert_axis(Axis(0))
                    .mapv(|x| x as i64),
            )?
            .into_iter()
            .map(|x| EmbedData::new(x, None, metadata.clone()))
            .collect::<Vec<_>>();
        Ok(e[0].clone())
    }

    fn embed_image_batch(&self, image_paths: &[PathBuf]) -> anyhow::Result<Vec<EmbedData>> {
        let images = image_paths
            .iter()
            .map(|path| image::open(path).unwrap())
            .collect::<Vec<_>>();
        let (input_ids, attention_mask, image_array, pixel_attention_mask) = self.processor.preprocess(&images)?;

        let e = self
            .run_model(
                input_ids,
                attention_mask,
                image_array.insert_axis(Axis(0)),
                pixel_attention_mask
                    .unwrap()
                    .insert_axis(Axis(0))
                    .mapv(|x| x as i64),
            )?
            .into_iter()
            .enumerate()
            .map(|(i, x)| {
                let mut metadata: HashMap<String, String> = HashMap::new();
                metadata.insert(
                    "file_path".to_string(),
                    image_paths[i].to_str().unwrap().to_string(),
                );
                EmbedData::new(x, None, Some(metadata))
            })
            .collect::<Vec<_>>();
        Ok(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lazy_static::lazy_static;
    use std::fs;
    use std::path::Path;
    use std::sync::Mutex;

    lazy_static! {
        static ref MODEL: Mutex<OrtColSmolEmbedder> = Mutex::new(
            OrtColSmolEmbedder::new("onnx-community/colSmol-256M-ONNX", None, None).unwrap()
        );
    }

    const IMAGE_PATH: &str = "temp_image.png";
    const IMAGE_URL: &str =
        "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png";

    async fn download_image() -> anyhow::Result<()> {
        if !Path::new(IMAGE_PATH).exists() {
            let response = reqwest::get(IMAGE_URL).await?;
            let bytes = response.bytes().await?;
            fs::write(IMAGE_PATH, &bytes)?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_colpali_embed_image() -> anyhow::Result<()> {
        download_image().await?;
        let model = MODEL.lock().unwrap();
        let embedding = model.embed_image(PathBuf::from(IMAGE_PATH), None).unwrap();
        assert!(
            !embedding.embedding.to_multi_vector().unwrap().is_empty(),
            "Embedding should not be empty"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_colpali_embed_image_batch() -> anyhow::Result<()> {
        download_image().await?;
        let model = MODEL.lock().unwrap();
        let embeddings = model
            .embed_image_batch(&[PathBuf::from(IMAGE_PATH), PathBuf::from(IMAGE_PATH)])
            .unwrap();
        assert_eq!(embeddings.len(), 2, "There should be two embeddings");
        Ok(())
    }

    #[tokio::test]
    async fn test_colpali_embed_file() -> anyhow::Result<()> {
        let model = MODEL.lock().unwrap();
        let embeddings = model
            .embed_file(PathBuf::from("../test_files/test.pdf"), 1)
            .unwrap();
        assert_eq!(embeddings.len(), 1, "There should be 1 embeddings");
        Ok(())
    }

    #[test]
    fn cleanup() -> anyhow::Result<()> {
        if Path::new(IMAGE_PATH).exists() {
            fs::remove_file(IMAGE_PATH)?;
        }
        Ok(())
    }
}
