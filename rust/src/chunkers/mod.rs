//! Text chunking algorithm implementations.
//!
//! Provides low-level chunking algorithms that support the text splitting
//! strategies available in the embedding pipeline. These algorithms balance
//! context preservation with computational efficiency.
//!
//! # Available Algorithms
//!
//! - **Statistical** - Length-based chunking with statistical overlap
//! - **Cumulative** - Boundary-aware accumulation chunking
//!
//! # Usage
//!
//! These algorithms are used internally by the embedding pipeline.
//! End users configure chunking through [`TextEmbedConfig`].
//!
//! [`TextEmbedConfig`]: crate::config::TextEmbedConfig

pub mod cumulative;
pub mod statistical;
