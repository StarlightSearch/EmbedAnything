use anyhow::Error as E;
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

pub fn tokenize_batch(
    tokenizer: &Tokenizer,
    text_batch: &[String],
    device: &Device,
) -> anyhow::Result<Tensor> {
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

    Ok(Tensor::stack(&token_ids, 0)?)
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
