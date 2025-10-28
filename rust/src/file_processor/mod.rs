//! Audio file processing utilities.
//!
//! Provides specialized processors for extracting and transcribing audio content
//! with temporal segmentation for embedding generation.
//!
//! # Features
//!
//! - **Transcription** - Speech-to-text conversion
//! - **Temporal segmentation** - Time-based audio chunking
//! - **Multiple formats** - Support for various audio file types
//!
//! # Usage
//!
//! Audio processing is handled through the main embedding functions:
//!
//! ```rust
//! use embed_anything::embed_file;
//! # async fn example() -> anyhow::Result<()> {
//! let embeddings = embed_file("audio.wav", &embedder, None, None).await?;
//! # Ok(())
//! # }
//! ```

pub mod audio;
