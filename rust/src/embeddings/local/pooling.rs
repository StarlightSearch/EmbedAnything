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

                if left_padding == 1 {
                    return Ok(PooledOutputType::Tensor(
                        tensor.get_on_dim(1, tensor.dim(1)? - 1)?,
                    ));
                } else if left_padding == 0 {
                    let sequence_lengths = attention_mask.sum( 1)?.to_vec1::<u32>()?;
                    let batch_size = tensor.dim(0)?;

                    let mut final_tensor = vec![];
                    for i in 0..batch_size{
                        let t = tensor.get(i)?.get(sequence_lengths[i] as usize)?;
                        final_tensor.push(t);
                    }
                    let final_tensor = Tensor::stack(&final_tensor, 1)?;
                    return Ok(PooledOutputType::Tensor(final_tensor))
                    
                } else {
                    return Err(anyhow::anyhow!("Left padding is not 0 or 1"));
                }
            }
            ModelOutput::Array(array) => {
                let attention_mask = attention_mask
                    .ok_or_else(|| {
                        anyhow::anyhow!("Attention mask required for LastToken pooling output")
                    })?
                    .to_array()?;

                let last_mask_column = attention_mask.slice(s![.., -1]);
                let left_padding = last_mask_column.iter().sum::<f32>();

                if left_padding == 0.0 {
                    return Ok(PooledOutputType::Array(
                        array.slice(s![.., -1, ..]).to_owned(),
                    ));
                }

                Ok(PooledOutputType::Array(
                    array.slice(s![.., -1, ..]).to_owned(),
                ))
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

                let mask_sum = expanded_mask.sum_all()?.clamp(1e-10, f32::MAX)?;

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

                let mask_sum = mask_3d.iter().sum::<f32>();

                let result = output
                    .view()
                    .mul(&mask_3d)
                    .sum_axis(Axis(1))
                    .mapv(|x| x / mask_sum.clamp(1e-10, f32::MAX));

                Ok(PooledOutputType::Array(result.to_owned()))
            }
        }
    }
}
