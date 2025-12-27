use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use half::f16;
use ndarray::{s, Array2};
use ort::session::Session;

pub fn compute_scores(
    model: &mut Session,
    input_ids: Array2<i64>,
    attention_mask: Array2<i64>,
    position_ids: Array2<i64>,
    false_token_id: u32,
    true_token_id: u32,
) -> Result<Vec<f32>, E> {
    let input_ids_tensor = ort::value::TensorRef::from_array_view(&input_ids)?;
    let attention_mask_tensor = ort::value::TensorRef::from_array_view(&attention_mask)?;
    let position_ids_tensor = ort::value::TensorRef::from_array_view(&position_ids)?;

    let inputs = ort::inputs!["input_ids" => input_ids_tensor, "attention_mask" => attention_mask_tensor, "position_ids" => position_ids_tensor];

    let outputs = model.run(inputs)?;
    let logits = outputs["logits".to_string()]
        .try_extract_array::<f16>()?
        .to_owned()
        .into_dimensionality::<ndarray::Ix3>()?
        .map(|&x| x.to_f32())
        .into_dimensionality::<ndarray::Ix3>()?;

    let logits = logits
        .slice(s![.., input_ids.shape()[1] - 1, ..])
        .to_owned();
    let false_logits = logits.slice(s![.., false_token_id as usize]).to_owned();
    let true_logits = logits.slice(s![.., true_token_id as usize]).to_owned();

    let false_logits_tensor =
        Tensor::from_vec(false_logits.to_vec(), input_ids.shape()[0], &Device::Cpu)?;
    let true_logits_tensor =
        Tensor::from_vec(true_logits.to_vec(), input_ids.shape()[0], &Device::Cpu)?;

    let logits_tensor = Tensor::stack(&[&false_logits_tensor, &true_logits_tensor], 1)?;
    let probs = candle_nn::ops::softmax(&logits_tensor, 1)?;
    let scores = probs.i((.., 1))?;
    Ok(scores.to_vec1::<f32>()?)
}
