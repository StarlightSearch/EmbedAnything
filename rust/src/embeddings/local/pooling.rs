use candle_core::Tensor;
use ndarray::prelude::*;
use ndarray::{Array2, Array3};

#[derive(Debug, Clone, Default)]
pub enum Pooling {
    #[default]
    Mean,
    Cls,
}

#[derive(Debug, Clone)]
pub enum PooledOutput {
    Tensor(Tensor),
    Array(Array2<f32>),
}

impl PooledOutput {
    pub fn to_tensor(self) -> Result<Tensor, anyhow::Error> {
        Ok(match self {
            PooledOutput::Tensor(tensor) => tensor,
            PooledOutput::Array(_) => panic!("Not implemented"),
        })
    }

    pub fn to_array(self) -> Result<Array2<f32>, anyhow::Error> {
        Ok(match self {
            PooledOutput::Tensor(_) => panic!("Not implemented"),
            PooledOutput::Array(array) => array,
        })
    }
}

pub enum ModelOutput {
    Tensor(Tensor),
    Array(Array3<f32>),
}

impl Pooling {
    pub fn pool(&self, output: &ModelOutput) -> Result<PooledOutput, anyhow::Error> {
        match self {
            Pooling::Cls => Self::cls(output),
            Pooling::Mean => Self::mean(output),
        }
    }

    fn cls(output: &ModelOutput) -> Result<PooledOutput, anyhow::Error> {
        match output {
            ModelOutput::Tensor(tensor) => tensor
                .get_on_dim(1, 0)
                .map(PooledOutput::Tensor)
                .map_err(|_| anyhow::anyhow!("Cls of empty tensor")),
            ModelOutput::Array(array) => {
                Ok(PooledOutput::Array(array.slice(s![.., 0, ..]).to_owned()))
            }
        }
    }

    fn mean(output: &ModelOutput) -> Result<PooledOutput, anyhow::Error> {
        match output {
            ModelOutput::Tensor(tensor) => tensor
                .mean(1)
                .map(PooledOutput::Tensor)
                .map_err(|_| anyhow::anyhow!("Mean of empty tensor")),
            ModelOutput::Array(array) => array
                .mean_axis(Axis(1))
                .map(PooledOutput::Array)
                .ok_or_else(|| anyhow::anyhow!("Mean of empty array")),
        }
    }
}
