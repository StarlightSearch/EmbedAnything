//! Neural network model implementations for embedding generation.
//!
//! Candle-based implementations of embedding models for local inference
//! with optional GPU acceleration support.
//!
//! # Usage
//!
//! Models are accessed through the embedding API:
//!
//! ```rust
//! use embed_anything::embeddings::embed::EmbedderBuilder;
//!
//! let embedder = EmbedderBuilder::new()
//!     .model_architecture("bert")
//!     .model_id(Some("sentence-transformers/all-MiniLM-L6-v2"))
//!     .from_pretrained_hf()?;
//! ```

pub mod bert;
pub mod clip;
pub mod colpali;
pub mod dinov2;
pub mod gemma;
pub mod idefics3;
pub mod jina_bert;
pub mod llama;
pub mod modernbert;
pub mod paligemma;
pub mod quantized_qwen3;
pub mod qwen3;
pub mod siglip;
pub mod with_tracing;
