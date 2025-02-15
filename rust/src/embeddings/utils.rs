use anyhow::Error as E;
use candle_core::{Device, Tensor};
use ndarray::Array2;
use tokenizers::Tokenizer;

pub fn tokenize_batch(
    tokenizer: &Tokenizer,
    text_batch: &[&str],
    device: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let tokens = tokenizer
        .encode_batch(text_batch.to_vec(), true)
        .map_err(E::msg)?;
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;

    Ok((
        Tensor::stack(&token_ids, 0)?,
        Tensor::stack(&attention_mask, 0)?,
    ))
}

pub fn get_attention_mask(
    tokenizer: &Tokenizer,
    text_batch: &[String],
    device: &Device,
) -> anyhow::Result<Tensor> {
    let tokens = tokenizer
        .encode_batch(text_batch.to_vec(), true)
        .map_err(E::msg)?;

    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok(Tensor::stack(&attention_mask, 0)?)
}

pub fn get_attention_mask_ndarray(
    tokenizer: &Tokenizer,
    text_batch: &[&str],
) -> anyhow::Result<Array2<i64>> {
    let attention_mask = tokenizer
        .encode_batch(text_batch.to_vec(), true)
        .map_err(E::msg)?
        .iter()
        .map(|tokens| {
            tokens
                .get_attention_mask()
                .iter()
                .map(|&id| id as i64)
                .collect::<Vec<i64>>()
        })
        .collect::<Vec<Vec<i64>>>();

    let attention_mask_array = Array2::from_shape_vec(
        (attention_mask.len(), attention_mask[0].len()),
        attention_mask.into_iter().flatten().collect::<Vec<i64>>(),
    )
    .unwrap();
    Ok(attention_mask_array)
}

pub fn tokenize_batch_ndarray(
    tokenizer: &Tokenizer,
    text_batch: &[&str],
) -> anyhow::Result<(Array2<i64>, Array2<i64>)> {
    let tokens = tokenizer
        .encode_batch(text_batch.to_vec(), true)
        .map_err(E::msg)?;
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            tokens
                .get_ids()
                .iter()
                .map(|&id| id as i64)
                .collect::<Vec<i64>>()
        })
        .collect::<Vec<Vec<i64>>>();
    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            tokens
                .get_attention_mask()
                .to_vec()
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
    let attention_mask_array = Array2::from_shape_vec(
        (attention_mask.len(), attention_mask[0].len()),
        attention_mask.into_iter().flatten().collect::<Vec<i64>>(),
    )
    .unwrap();
    Ok((token_ids_array, attention_mask_array))
}

pub fn get_type_ids_ndarray(
    tokenizer: &Tokenizer,
    text_batch: &[&str],
) -> anyhow::Result<Array2<i64>> {
    let token_ids = tokenizer
        .encode_batch(text_batch.to_vec(), true)
        .map_err(E::msg)?
        .iter()
        .map(|tokens| {
            tokens
                .get_type_ids()
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
