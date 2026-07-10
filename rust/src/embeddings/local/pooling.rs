use candle_core::Tensor;
use ndarray::prelude::*;
use ndarray::{Array2, Array3};
use std::ops::Mul;

#[derive(Debug, Clone)]
pub enum Pooling {
    Mean,
    Cls,
    LastToken,
}

#[derive(Debug, Clone)]
pub enum PooledOutputType {
    Tensor(Tensor),
    Array(Array2<f32>),
}

impl PooledOutputType {
    pub fn to_tensor(&self) -> Result<&Tensor, anyhow::Error> {
        match self {
            PooledOutputType::Tensor(tensor) => Ok(tensor),
            PooledOutputType::Array(_) => Err(anyhow::anyhow!("Cannot convert Array to Tensor")),
        }
    }

    pub fn to_array(&self) -> Result<&Array2<f32>, anyhow::Error> {
        match self {
            PooledOutputType::Tensor(_) => Err(anyhow::anyhow!("Cannot convert Tensor to Array")),
            PooledOutputType::Array(array) => Ok(array),
        }
    }
}
impl From<Tensor> for PooledOutputType {
    fn from(value: Tensor) -> Self {
        PooledOutputType::Tensor(value)
    }
}

impl From<Array2<f32>> for PooledOutputType {
    fn from(value: Array2<f32>) -> Self {
        PooledOutputType::Array(value)
    }
}
pub enum ModelOutput {
    Tensor(Tensor),
    Array(Array3<f32>),
}

impl Pooling {
    pub fn pool(
        &self,
        output: &ModelOutput,
        attention_mask: Option<&PooledOutputType>,
    ) -> Result<PooledOutputType, anyhow::Error> {
        match self {
            Pooling::Cls => Self::cls(output),
            Pooling::Mean => Self::mean(output, attention_mask),
            Pooling::LastToken => Self::last_token(output, attention_mask),
        }
    }

    fn cls(output: &ModelOutput) -> Result<PooledOutputType, anyhow::Error> {
        match output {
            ModelOutput::Tensor(tensor) => tensor
                .get_on_dim(1, 0)
                .map(PooledOutputType::Tensor)
                .map_err(|_| anyhow::anyhow!("Cls of empty tensor")),
            ModelOutput::Array(array) => Ok(PooledOutputType::Array(
                array.slice(s![.., 0, ..]).to_owned(),
            )),
        }
    }
    fn last_token(
        output: &ModelOutput,
        attention_mask: Option<&PooledOutputType>,
    ) -> Result<PooledOutputType, anyhow::Error> {
        match output {
            ModelOutput::Tensor(tensor) => {
                let attention_mask = if let Some(mask) = attention_mask {
                    mask.to_tensor()?
                } else {
                    &tensor.ones_like()?
                };

                // check if left padding by taking sum of last attention mask column
                let left_padding = attention_mask
                    .get_on_dim(1, attention_mask.dim(1)? - 1)?
                    .sum_all()?
                    .to_scalar::<u32>()?;

                if left_padding == attention_mask.dim(0)? as u32 {
                    Ok(PooledOutputType::Tensor(
                        tensor.get_on_dim(1, tensor.dim(1)? - 1)?,
                    ))
                } else {
                    let sequence_lengths = attention_mask.sum(1)?.to_vec1::<u32>()?;
                    let batch_size = tensor.dim(0)?;

                    // Create a tensor of indices for the last tokens
                    let indices: Vec<u32> = sequence_lengths.iter().map(|&len| len - 1).collect();
                    let indices = Tensor::from_vec(indices, (batch_size,), tensor.device())?;

                    // Use gather to get all last tokens at once
                    let final_tensor = tensor.gather(&indices, 1)?;
                    Ok(PooledOutputType::Tensor(final_tensor))
                }
            }
            ModelOutput::Array(array) => {
                let attention_mask = attention_mask
                    .ok_or_else(|| {
                        anyhow::anyhow!("Attention mask required for LastToken pooling output")
                    })?
                    .to_array()?;

                let sequence_lengths = attention_mask.sum_axis(Axis(1));
                let batch_size = array.shape()[0];

                let mut final_embeddings = vec![];
                for i in 0..batch_size {
                    let t = array.slice(s![i, .., (sequence_lengths[i] - 1.0) as usize]);
                    final_embeddings.push(t);
                }
                let final_embeddings = ndarray::stack(Axis(1), &final_embeddings)?;
                Ok(PooledOutputType::Array(final_embeddings))
            }
        }
    }

    fn mean(
        output: &ModelOutput,
        attention_mask: Option<&PooledOutputType>,
    ) -> Result<PooledOutputType, anyhow::Error> {
        match output {
            ModelOutput::Tensor(tensor) => {
                let attention_mask = if let Some(mask) = attention_mask {
                    mask.to_tensor()?
                } else {
                    &tensor.ones_like()?
                };

                let expanded_mask = attention_mask
                    .unsqueeze(2)?
                    .expand(&[tensor.dim(0)?, tensor.dim(1)?, tensor.dim(2)?])?
                    .to_dtype(tensor.dtype())?;

                // Per-row token count: sum the mask over the sequence dim only (not over
                // the hidden dim or the whole batch). Shape [batch, hidden]; each row holds
                // that row's number of non-padding tokens.
                let mask_sum = expanded_mask.sum(1)?.clamp(1e-10, f32::MAX)?;

                let result = tensor
                    .mul(&expanded_mask)?
                    .sum(1)?
                    .broadcast_div(&mask_sum)?;

                Ok(PooledOutputType::Tensor(result))
            }
            ModelOutput::Array(output) => {
                let attention_mask = attention_mask
                    .ok_or_else(|| {
                        anyhow::anyhow!("Attention mask required for Mean pooling output")
                    })?
                    .to_array()?;

                let mask_3d = attention_mask.view().insert_axis(Axis(2));

                // Per-row token count (sum over the sequence dim), one denominator per row.
                let counts = attention_mask.sum_axis(Axis(1));

                let mut result = output.view().mul(&mask_3d).sum_axis(Axis(1));
                for (mut row, &count) in result.outer_iter_mut().zip(counts.iter()) {
                    let denom = count.max(1e-10);
                    row.mapv_inplace(|x| x / denom);
                }

                Ok(PooledOutputType::Array(result))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // A batch whose two rows have different real-token counts, so the correct mean
    // divisor differs per row (this is what the `sum_all` bug got wrong):
    //   row 0: tokens [1,2] and [3,4], plus one PADDING token [5,6] (mask 0)
    //   row 1: tokens [10,20], [30,40], [50,60] (all real)
    const HIDDEN: [f32; 12] = [1., 2., 3., 4., 5., 6., 10., 20., 30., 40., 50., 60.];
    // Correct per-row mean over real tokens:
    //   row0 = ([1,2] + [3,4]) / 2 = [2, 3]
    //   row1 = ([10,20] + [30,40] + [50,60]) / 3 = [30, 40]
    const EXPECTED: [[f32; 2]; 2] = [[2., 3.], [30., 40.]];

    fn assert_close(got: &[Vec<f32>]) {
        assert_eq!(got.len(), EXPECTED.len());
        for (row, want) in got.iter().zip(EXPECTED.iter()) {
            for (a, b) in row.iter().zip(want.iter()) {
                assert!((a - b).abs() < 1e-6, "got {got:?} want {EXPECTED:?}");
            }
        }
    }

    #[test]
    fn mean_pool_tensor_divides_by_per_row_token_count() {
        let device = Device::Cpu;
        let hidden = Tensor::from_vec(HIDDEN.to_vec(), (2, 3, 2), &device).unwrap();
        let mask = Tensor::from_vec(vec![1u32, 1, 0, 1, 1, 1], (2, 3), &device).unwrap();
        let pooled = Pooling::Mean
            .pool(
                &ModelOutput::Tensor(hidden),
                Some(&PooledOutputType::from(mask)),
            )
            .unwrap();
        assert_close(&pooled.to_tensor().unwrap().to_vec2::<f32>().unwrap());
    }

    #[test]
    fn mean_pool_ndarray_divides_by_per_row_token_count() {
        let hidden = Array3::from_shape_vec((2, 3, 2), HIDDEN.to_vec()).unwrap();
        let mask = Array2::from_shape_vec((2, 3), vec![1., 1., 0., 1., 1., 1.]).unwrap();
        let pooled = Pooling::Mean
            .pool(
                &ModelOutput::Array(hidden),
                Some(&PooledOutputType::from(mask)),
            )
            .unwrap();
        let rows: Vec<Vec<f32>> = pooled
            .to_array()
            .unwrap()
            .outer_iter()
            .map(|r| r.to_vec())
            .collect();
        assert_close(&rows);
    }
}
