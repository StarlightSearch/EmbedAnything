use candle_core::Tensor;
use ndarray::prelude::*;
use ndarray::{Array2, Array3};
use std::ops::Mul;

#[derive(Debug, Clone)]
pub enum Pooling {
    Mean,
    Cls,
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
