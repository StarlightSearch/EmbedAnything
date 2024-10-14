//! This module contains the different embedding models that can be used to generate embeddings for the text data.

use std::{collections::HashMap, rc::Rc};

use candle_core::Tensor;
use embed::{EmbedData, Embedder, EmbeddingResult};

use crate::file_processor::audio::audio_processor::Segment;

pub mod cloud;
pub mod embed;
pub mod local;

use rayon::prelude::*;
pub fn get_text_metadata(
    encodings: &Rc<Vec<EmbeddingResult>>,
    text_batch: &Vec<String>,
    metadata: &Option<HashMap<String, String>>,
) -> anyhow::Result<Vec<EmbedData>> {
    let final_embeddings = encodings
        .par_iter()
        .zip(text_batch)
        .map(|(data, text)| EmbedData::new(data.clone(), Some(text.clone()), metadata.clone()))
        .collect::<Vec<_>>();
    Ok(final_embeddings)
}

pub fn get_audio_metadata<T: AsRef<std::path::Path>>(
    encodings: Vec<EmbeddingResult>,
    segments: Vec<Segment>,
    audio_file: T,
) -> Result<Vec<EmbedData>, anyhow::Error> {
    let final_embeddings = encodings
        .iter()
        .enumerate()
        .map(|(i, data)| {
            let mut metadata = HashMap::new();
            metadata.insert("start".to_string(), segments[i].start.to_string());
            metadata.insert(
                "end".to_string(),
                (segments[i].start + segments[i].duration).to_string(),
            );
            metadata.insert(
                "file_name".to_string(),
                audio_file.as_ref().to_str().unwrap().to_string(),
            );
            metadata.insert("text".to_string(), segments[i].dr.text.clone());
            EmbedData::new(
                data.clone(),
                Some(segments[i].dr.text.clone()),
                Some(metadata),
            )
        })
        .collect::<Vec<_>>();
    Ok(final_embeddings)
}

pub fn text_batch_from_audio(segments: &[Segment]) -> Vec<String> {
    segments
        .iter()
        .map(|segment| segment.dr.text.clone())
        .collect()
}

pub async fn embed_audio<T: AsRef<std::path::Path>>(
    embeder: &Embedder,
    segments: Vec<Segment>,
    audio_file: T,
    batch_size: Option<usize>,
) -> Result<Vec<EmbedData>, anyhow::Error> {
    let text_batch = text_batch_from_audio(&segments);
    let encodings = embeder.embed(&text_batch, batch_size).await?;
    get_audio_metadata(encodings, segments, audio_file)
}

pub fn normalize_l2(v: &Tensor) -> candle_core::Result<Tensor> {
    v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
}
