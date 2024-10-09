use anyhow::Error as E;
use anyhow::Result;

use candle_core::Device;
use candle_core::Tensor;
use hf_hub::{api::sync::Api, Repo};
use ndarray::Array2;
use ndarray::Axis;
use ort::{GraphOptimizationLevel, Session};
use tokenizers::{PaddingParams, Tokenizer};

fn main() -> Result<()> {
    let revision = Some("main".to_string());
    let model_id = "BAAI/bge-small-en-v1.5".to_string();
    let (_config_filename, tokenizer_filename, weights_filename) = {
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
        let weights = api.get("onnx/model.onnx")?;

        (config, tokenizer, weights)
    };

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(weights_filename)?;

    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let pp = PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    };
    tokenizer.with_padding(Some(pp));


    println!("{:?}", model.inputs);

    let outputs  = embed(&tokenizer, &model, &vec!["Hello world".to_string(), "Bye World".to_string()], None)?;
    Ok(())
}

pub fn tokenize_batch(
    tokenizer: &Tokenizer,
    text_batch: &[String],
    device: &Device,
) -> Result<Array2<i64>, E> {
    let tokens = tokenizer
        .encode_batch(text_batch.to_vec(), true)
        .map_err(E::msg)?;
    let token_ids = tokens
        .iter()
        .map(|tokens| tokens.get_ids().iter().map(|&id| id as i64).collect::<Vec<i64>>())
        .collect::<Vec<Vec<i64>>>();

    let token_ids_array = Array2::from_shape_vec(
        (token_ids.len(), token_ids[0].len()),
        token_ids.into_iter().flatten().collect::<Vec<i64>>(),
    )
    .unwrap();
    Ok(token_ids_array)
}

pub fn embed(
    tokenizer: &Tokenizer,
    model: &Session,
    text_batch: &[String],
    batch_size: Option<usize>,
) -> Result<Vec<Vec<f32>>, E> {
    let batch_size = batch_size.unwrap_or(32);
    let mut encodings: Vec<Vec<f32>> = Vec::new();

    for mini_text_batch in text_batch.chunks(batch_size) {
        let token_ids: Array2<i64> = tokenize_batch(tokenizer, mini_text_batch, &Device::Cpu).unwrap();
        let token_type_ids: Array2<i64> = Array2::zeros(token_ids.raw_dim());
        let attention_mask: Array2<i64> = Array2::ones(token_ids.raw_dim());
        let outputs = model.run(ort::inputs![token_ids, token_type_ids, attention_mask]?).unwrap();
        let embeddings = outputs["last_hidden_state"].try_extract_tensor::<f32>().unwrap();
        let shape = embeddings.shape();
        let (_, n_tokens, _hidden_size) = (shape[0], shape[1], shape[2]); 
        let embeddings = embeddings.sum_axis(Axis(1)) / n_tokens as f32;
        let denominator = embeddings.mapv(|x| x*x).sum_axis(Axis(1)).sqrt().insert_axis(Axis(1));
        let embeddings = embeddings.clone()/ denominator;
        println!("{:?}", embeddings);
        let batch_encodings: (Vec<f32>, Option<usize>) = embeddings.into_raw_vec_and_offset();
        let batch_encodings = reshape_into_2d_vector(batch_encodings.0, Some(n_tokens as usize));
        encodings.extend(batch_encodings);
    }

    Ok(encodings)
}

fn reshape_into_2d_vector(
    raw_data: Vec<f32>, 
    dim: Option<usize>
) -> Vec<Vec<f32>> {
    let dim = dim.expect("Dimension must be provided");
    let mut reshaped: Vec<Vec<f32>> = Vec::new();

    for chunk in raw_data.chunks(dim) {
        reshaped.push(chunk.to_vec());
    }

    reshaped
}