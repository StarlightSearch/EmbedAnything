//! Local embedding model implementations.
//!
//! Self-contained embedding models that run locally without external API calls.
//! Models use either Candle backend or ONNX Runtime for inference.

pub mod bert;
pub mod clip;
#[cfg(feature = "ort")]
pub mod colbert;
pub mod colpali;
#[cfg(feature = "ort")]
pub mod colpali_ort;
pub mod colsmol;
pub mod jina;
pub mod model2vec;
pub mod model_info;
pub mod modernbert;
#[cfg(feature = "ort")]
pub mod ort_bert;
#[cfg(feature = "ort")]
pub mod ort_jina;
pub mod pooling;
pub mod qwen3;
pub mod text_embedding;
pub mod vision_encoder;
