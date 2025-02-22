#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/StarlightSearch/EmbedAnything/refs/heads/main/docs/assets/icon.ico"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/StarlightSearch/EmbedAnything/refs/heads/main/docs/assets/Square310x310Logo.png"
)]
#![doc(issue_tracker_base_url = "https://github.com/StarlightSearch/EmbedAnything/issues/")]
//! embed_anything is a minimalist, highly performant, lightning-fast, lightweight, multisource,
//! multimodal, and local embedding pipeline.
//!
//! Whether you're working with text, images, audio, PDFs, websites, or other media, embed_anything
//! streamlines the process of generating embeddings from various sources and seamlessly streaming
//! (memory-efficient-indexing) them to a vector database.
//!
//! It supports dense, sparse, [ONNX](https://github.com/onnx/onnx) and late-interaction embeddings,
//! offering flexibility for a wide range of use cases.
//!
//! # Usage
//!
//! ## Creating an [Embedder]
//!
//! To get started, you'll need to create an [Embedder] for the type of content you want to embed.
//! We offer some utility functions to streamline creating embedders from various sources, such as
//! [Embedder::from_pretrained_hf], [Embedder::from_pretrained_onnx], and
//! [Embedder::from_pretrained_cloud]. You can use any of these to quickly create an Embedder like so:
//!
//! ```rust
//! use embed_anything::embeddings::embed::Embedder;
//!
//! // Create a local CLIP embedder from a Hugging Face model
//! let clip_embedder = Embedder::from_pretrained_hf("CLIP", "jina-clip-v2", None, None, None);
//!
//! // Create a cloud OpenAI embedder
//! let openai_embedder = Embedder::from_pretrained_cloud("OpenAI", "gpt-3.5-turbo", Some("my-api-key".to_string()));
//! ```
//!
//! If needed, you can also create an instance of [Embedder] manually, allowing you to create your
//! own embedder! Here's an example of manually creating embedders:
//!
//! ```rust
//! use embed_anything::embeddings::embed::{Embedder, TextEmbedder};
//! use embed_anything::embeddings::local::jina::JinaEmbedder;
//!
//! let jina_embedder = Embedder::Text(TextEmbedder::Jina(Box::new(JinaEmbedder::default())));
//! ```
//!
//! ## Generate embeddings
//!
//! # Example: Embed a text file
//!
//! Let's see how embed_anything can help us generate embeddings from a plain text file:
//!
//! ```rust
//! use embed_anything::embed_file;
//! use embed_anything::embeddings::embed::{Embedder, TextEmbedder};
//! use embed_anything::embeddings::local::jina::JinaEmbedder;
//!
//! // Create an Embedder for text. We support a variety of models out-of-the-box, including cloud-based models!
//! let embedder = Embedder::Text(TextEmbedder::Jina(Box::new(JinaEmbedder::default())));
//! // Generate embeddings for 'path/to/file.txt' using the embedder we just created.
//! let embedding = embed_file("path/to/file.txt", &embedder, None, None);
//! ```

pub mod chunkers;
pub mod config;
pub mod embeddings;
pub mod file_loader;
pub mod file_processor;
pub mod models;
#[cfg(feature = "ort")]
pub mod reranker;
pub mod tesseract;
pub mod text_loader;

use std::{collections::HashMap, fs, path::PathBuf, rc::Rc, sync::Arc};
use std::ffi::OsStr;
use anyhow::Result;
use config::{ImageEmbedConfig, TextEmbedConfig};
use embeddings::{
    embed::{EmbedData, EmbedImage, Embedder, TextEmbedder, VisionEmbedder},
    get_text_metadata,
};
use file_loader::FileParser;
use file_processor::audio::audio_processor::AudioDecoderModel;
use itertools::Itertools;
use rayon::prelude::*;
use text_loader::TextLoader;
use tokio::sync::mpsc;
// Add this at the top of your file

use crate::file_processor::markdown_processor::MarkdownProcessor;

use crate::file_processor::docx_processor::DocxProcessor;
use crate::file_processor::html_processor::HtmlProcessor;
use crate::file_processor::pdf_processor::PdfProcessor;
use crate::file_processor::processor::{Document, DocumentProcessor};
use crate::file_processor::txt_processor::TxtProcessor;
use crate::text_loader::FileLoadingError;
#[cfg(feature = "audio")]
use embeddings::embed_audio;

pub enum Dtype {
    F16,
    INT8,
    Q4,
    UINT8,
    BNB4,
    F32,
    Q4F16,
    QUANTIZED,
}

/// Embeds a list of queries using the specified embedding model.
///
/// # Arguments
///
/// * `query` - A vector of strings representing the queries to embed.
/// * `embedder` - A string specifying the embedding model to use. Valid options are "OpenAI", "Jina", "Clip", and "Bert".
/// * `config` - An optional `EmbedConfig` object specifying the configuration for the embedding model.
/// * 'adapter' - An optional `Adapter` object to send the embeddings to a vector database.
///
/// # Returns
///
/// A vector of `EmbedData` objects representing the embeddings of the queries.
///
/// # Errors
///
/// Returns a `PyValueError` if an invalid embedding model is specified.
///
/// # Example
///
/// ```
/// use embed_anything::embed_query;
/// use embed_anything::embeddings::embed::{Embedder, TextEmbedder};
/// use embed_anything::embeddings::local::jina::JinaEmbedder;
///
/// let query = ["Hello", "World"];
/// let embedder = Embedder::Text(TextEmbedder::Jina(Box::new(JinaEmbedder::default())));
/// let embeddings = embed_query(&query, &embedder, None).unwrap();
/// println!("{:?}", embeddings);
/// ```
pub async fn embed_query(
    query: &[&str],
    embedder: &Embedder,
    config: Option<&TextEmbedConfig>,
) -> Result<Vec<EmbedData>> {
    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);
    let batch_size = config.batch_size;

    let encodings = embedder.embed(query, batch_size).await?;
    let embeddings = get_text_metadata(&Rc::new(encodings), query, &None)?;

    Ok(embeddings)
}

/// Embeds the text from a file using the specified embedding model.
///
/// # Arguments
///
/// * `file_name` - A string specifying the name of the file to embed.
/// * `embedder` - A string specifying the embedding model to use. Valid options are "OpenAI", "Jina", "Clip", and "Bert".
/// * `config` - An optional `EmbedConfig` object specifying the configuration for the embedding model.
/// * 'adapter' - An optional `Adapter` object to send the embeddings to a vector database.
///
/// # Returns
///
/// A vector of `EmbedData` objects representing the embeddings of the file.
///
/// # Errors
///
/// Returns a `PyValueError` if an invalid embedding model is specified.
///
/// # Example
///
/// ```rust
/// use embed_anything::embed_file;
/// use embed_anything::embeddings::embed::{Embedder, TextEmbedder};
/// use embed_anything::embeddings::local::bert::BertEmbedder;
///
/// let file_name = "path/to/file.pdf";
/// let embedder = Embedder::Text(TextEmbedder::from(BertEmbedder::new("sentence-transformers/all-MiniLM-L12-v2".into(), None, None).unwrap()));
/// let embeddings = embed_file(file_name, &embedder, None, None).unwrap();
/// ```
pub async fn embed_file(
    file_name: impl AsRef<std::path::Path>,
    embedder: &Embedder,
    config: Option<TextEmbedConfig>,
    adapter: Option<Box<dyn FnOnce(Vec<EmbedData>) + Send + Sync>>,
) -> Result<Option<Vec<EmbedData>>> {
    match embedder {
        Embedder::Text(embedder) => emb_text(file_name, embedder, config, adapter).await,
        Embedder::Vision(embedder) => Ok(Some(vec![emb_image(file_name, embedder)?])),
    }
}

/// Embeddings of a webpage using the specified embedding model.
///
/// # Arguments
///
/// * `embedder` - The embedding model to use. Supported options are "OpenAI", "Jina", and "Bert".
/// * `webpage` - The webpage to embed.
///
/// # Returns
///
/// The embeddings of the webpage.
///
/// # Errors
///
/// Returns an error if the specified embedding model is invalid.
///
/// # Example
///
/// ```
/// use embed_anything::embed_webpage;
/// use embed_anything::embeddings::embed::{Embedder, TextEmbedder};
/// use embed_anything::embeddings::local::jina::JinaEmbedder;
///
/// let embedder = Embedder::Text(TextEmbedder::Jina(Box::new(JinaEmbedder::default())));
/// let embeddings = embed_webpage("https://en.wikipedia.org/wiki/Embedding".into(), &embedder, None, None).unwrap();
/// ```
pub async fn embed_webpage(
    url: String,
    embedder: &Embedder,
    config: Option<&TextEmbedConfig>,
    // Callback function
    adapter: Option<Box<dyn FnOnce(Vec<EmbedData>) + Send + Sync>>,
) -> Result<Option<Vec<EmbedData>>>
{
    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);
    let chunk_size = config.chunk_size.unwrap_or(1000);

    let website_processor = file_processor::website_processor::WebsiteProcessor::new(chunk_size);
    let document = website_processor.process_website(url.as_ref())?;

    let batch_size = config.batch_size;

    // Convert Vec<String> to Vec<&str> for embedding
    let chunks = document.chunks;
    let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_ref()).collect();

    if let Some(adapter) = adapter {
        let encodings = embedder
            .embed(&chunk_refs, batch_size)
            .await?;
        let embeddings = get_text_metadata(&Rc::new(encodings), &chunk_refs, &None)?;
        adapter(embeddings);
        Ok(None)
    } else {
        let encodings = embedder
            .embed(&chunk_refs, batch_size)
            .await?;
        let embeddings = get_text_metadata(&Rc::new(encodings), &chunk_refs, &None)?;

        Ok(Some(embeddings))
    }
}

/// Embeds an HTML document using the specified embedding model.
///
/// # Arguments
///
/// * `file_name` - The path of the HTML document to embed.
/// * `origin` - The original URL of the document. If specified, links can be resolved and metadata points to the site.
/// * `embedder` - The embedding model to use. Supported options are "OpenAI", "Jina", and "Bert".
///
/// # Returns
///
/// The embeddings of the HTML document.
///
/// # Errors
///
/// Returns an error if the specified embedding model is invalid.
///
/// # Example
///
/// ```
/// use embed_anything::embed_html;
/// use embed_anything::embeddings::embed::{Embedder, TextEmbedder};
/// use embed_anything::embeddings::local::jina::JinaEmbedder;
///
/// async fn get_embeddings() {
///     let embeddings = embed_html(
///         "test_files/test.html",
///         &TextEmbedder::from_pretrained_hf("JINA", "jinaai/jina-embeddings-v2-small-en", None, None, None).unwrap(),
///         None,
///         None,
///     ).await.unwrap();
/// }
/// ```
pub async fn embed_html(
    file_name: impl AsRef<std::path::Path>,
    embedder: &TextEmbedder,
    config: Option<&TextEmbedConfig>,
    // Callback function
    adapter: Option<Box<dyn FnOnce(Vec<EmbedData>) + Send + Sync>>,
) -> Result<Option<Vec<EmbedData>>> {
    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);

    let html_processor = HtmlProcessor::new(config.chunk_size.unwrap_or(1000));

    let bytes = fs::read(&file_name)?;
    let out = String::from_utf8_lossy(&bytes);
    let html = html_processor.process_document(&out);

    let batch_size = config.batch_size;

    let embeddings = embed_document(html, None, embedder, batch_size).await?;

    // Send embeddings to vector database
    if let Some(adapter) = adapter {
        adapter(embeddings);
        Ok(None)
    } else {
        Ok(Some(embeddings))
    }
}

/// Embeds a Markdown document using the specified embedding model.
///
/// # Arguments
///
/// * `file_name` - The path of the HTML document to embed.
/// * `embedder` - The embedding model to use. Supported options are "OpenAI", "Jina", and "Bert".
///
/// # Returns
///
/// The embeddings of the Markdown document.
///
/// # Errors
///
/// Returns an error if the specified embedding model is invalid.
///
/// # Example
///
/// ```
/// use embed_anything::embed_markdown;
/// use embed_anything::embeddings::embed::{Embedder, TextEmbedder};
/// use embed_anything::embeddings::local::jina::JinaEmbedder;
///
/// async fn get_embeddings() {
///     let embeddings = embed_markdown(
///         "# This is some Markdown content",
///         &TextEmbedder::from_pretrained_hf("JINA", "jinaai/jina-embeddings-v2-small-en", None, None, None).unwrap(),
///         None,
///     ).await.unwrap();
/// }
/// ```
pub async fn embed_markdown(
    markdown: &str,
    embedder: &TextEmbedder,
    config: Option<&TextEmbedConfig>,
) -> Result<Vec<EmbedData>> {
    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);

    let md_processor = MarkdownProcessor::new(
        config.chunk_size.unwrap_or(1000)
    );
    let md = md_processor.process_document(markdown);

    embed_document(md, None, embedder, config.batch_size)
}

#[allow(clippy::too_many_arguments)]
async fn emb_text(
    file: impl AsRef<std::path::Path>,
    embedding_model: &TextEmbedder,
    config: Option<TextEmbedConfig>,
    adapter: Option<Box<dyn FnOnce(Vec<EmbedData>) + Send + Sync>>,
) -> Result<Option<Vec<EmbedData>>>
{
    if !file.as_ref().exists() {
        return Err(FileLoadingError::FileNotFound(
            file.as_ref().to_str().unwrap().to_string(),
        )
            .into());
    }
    
    let file_extension = file.as_ref().extension().unwrap();
    let processor = get_processor(file_extension, config.clone())?;

    let bytes = fs::read(&file)?;
    let out = String::from_utf8_lossy(&bytes);
    let document = processor.process_document(out.as_ref());

    let metadata = TextLoader::get_metadata(file).ok();

    let batch_size = config.map(|x| x.batch_size).flatten();

    let embeddings = embed_document(document, metadata, embedding_model, batch_size).await?;

    if let Some(adapter) = adapter {
        adapter(embeddings);
        Ok(None)
    } else {
        Ok(Some(embeddings))
    }
}

fn emb_image<T: AsRef<std::path::Path>>(
    image_path: T,
    embedding_model: &VisionEmbedder,
) -> Result<EmbedData> {
    let mut metadata = HashMap::new();
    metadata.insert(
        "file_name".to_string(),
        fs::canonicalize(&image_path)?.to_str().unwrap().to_string(),
    );
    let embedding = embedding_model
        .embed_image(&image_path, Some(metadata))?;

    Ok(embedding.clone())
}

#[cfg(feature = "audio")]
pub async fn emb_audio<T: AsRef<std::path::Path>>(
    audio_file: T,
    audio_decoder: &mut AudioDecoderModel,
    embedder: &Arc<Embedder>,
    text_embed_config: Option<&TextEmbedConfig>,
) -> Result<Option<Vec<EmbedData>>> {
    use file_processor::audio::audio_processor;

    let segments: Vec<audio_processor::Segment> = audio_decoder.process_audio(&audio_file)?;
    let embeddings = embed_audio(
        embedder,
        segments,
        audio_file,
        text_embed_config
            .unwrap_or(&TextEmbedConfig::default())
            .batch_size,
    )
    .await?;

    Ok(Some(embeddings))
}

#[cfg(not(feature = "audio"))]
pub async fn emb_audio<T: AsRef<std::path::Path>>(
    _audio_file: T,
    _audio_decoder: &mut AudioDecoderModel,
    _embedder: &Arc<Embedder>,
    _text_embed_config: Option<&TextEmbedConfig>,
) -> Result<Option<Vec<EmbedData>>> {
    Err(anyhow::anyhow!(
        "The 'audio' feature is not enabled. Please enable it to use the emb_audio function."
    ))
}

/// Embeds images in a directory using the specified embedding model.
///
/// # Arguments
///
/// * `directory` - A `PathBuf` representing the directory containing the images to embed.
/// * `embedder` - A reference to the embedding model to use.
/// * `config` - An optional `ImageEmbedConfig` object specifying the configuration for the embedding model. Default buffer size is 100.
/// * `adapter` - An optional callback function to handle the embeddings.
///
/// # Returns
/// An `Option` containing a vector of `EmbedData` objects representing the embeddings of the images, or `None` if an adapter is used.
///
/// # Errors
/// Returns a `Result` with an error if the embedding process fails.
///
/// # Example
///
/// ```rust
/// use embed_anything::embed_image_directory;
/// use std::path::PathBuf;
/// use std::sync::Arc;
/// use embed_anything::embeddings::embed::Embedder;
///
/// async fn embed_images() {
///     let directory = PathBuf::from("/path/to/directory");
///     let embedder = Arc::new(Embedder::from_pretrained_hf("clip", "openai/clip-vit-base-patch16", None, None, None).unwrap());
///     let embeddings = embed_image_directory(directory, &embedder, None, None).await.unwrap();
/// }
/// ```
/// This will output the embeddings of the images in the specified directory using the specified embedding model.
///
pub async fn embed_image_directory<T: EmbedImage + Send + Sync + 'static>(
    directory: PathBuf,
    embedding_model: &Arc<T>,
    config: Option<&ImageEmbedConfig>,
    adapter: Option<Box<dyn FnMut(Vec<EmbedData>) + Send + Sync>>,
) -> Result<Option<Vec<EmbedData>>> {
    let mut file_parser = FileParser::new();
    file_parser.get_image_paths(&directory)?;

    let buffer_size = config
        .unwrap_or(&ImageEmbedConfig::default())
        .buffer_size
        .unwrap_or(100);

    let (tx, mut rx) = mpsc::unbounded_channel();
    let (collector_tx, mut collector_rx) = mpsc::unbounded_channel();

    let embedder = embedding_model.clone();

    let pb = indicatif::ProgressBar::new(file_parser.files.len() as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )?,
    );

    let processing_task = tokio::spawn({
        async move {
            // make image buffer
            let mut image_buffer = Vec::with_capacity(buffer_size);
            let mut files_processed: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            while let Some(image) = rx.recv().await {
                image_buffer.push(image);

                if image_buffer.len() == buffer_size {
                    // Ensure embedder is mutable and not wrapped in Arc
                    match process_images(&image_buffer, embedder.clone()).await {
                        Ok(embeddings) => {
                            let files = embeddings
                                .iter()
                                .cloned()
                                .map(|e| e.metadata.unwrap().get("file_name").unwrap().to_string())
                                .collect::<Vec<_>>();

                            let unique_files = files.into_iter().unique().collect::<Vec<_>>();
                            let old_len = files_processed.len() as u64;
                            files_processed.extend(unique_files);
                            let new_len = files_processed.len() as u64;

                            pb.inc(new_len - old_len);

                            if let Err(e) = collector_tx.send(embeddings) {
                                eprintln!("Error sending embeddings to collector: {:?}", e);
                            }
                        }
                        Err(e) => eprintln!("Error processing images: {:?}", e),
                    }

                    image_buffer.clear();
                }
            }

            // Process any remaining images
            if !image_buffer.is_empty() {
                match process_images(&image_buffer, embedder).await {
                    Ok(embeddings) => {
                        let files = embeddings
                            .iter()
                            .cloned()
                            .map(|e| e.metadata.unwrap().get("file_name").unwrap().to_string())
                            .collect::<Vec<_>>();
                        let unique_files = files.into_iter().unique().collect::<Vec<_>>();
                        let old_len = files_processed.len() as u64;
                        files_processed.extend(unique_files);
                        let new_len = files_processed.len() as u64;

                        pb.inc(new_len - old_len);

                        if let Err(e) = collector_tx.send(embeddings) {
                            eprintln!("Error sending embeddings to collector: {:?}", e);
                        }
                    }
                    Err(e) => eprintln!("Error processing images: {:?}", e),
                }
            }
        }
    });

    file_parser.files.par_iter().for_each(|image| {
        if let Err(e) = tx.send(image.clone()) {
            eprintln!("Error sending image: {:?}", e);
        }
    });

    drop(tx);

    let mut all_embeddings = Vec::new();
    let mut adapter = adapter;
    while let Some(embeddings) = collector_rx.recv().await {
        if let Some(adapter) = adapter.as_mut() {
            adapter(embeddings.to_vec());
        } else {
            all_embeddings.extend(embeddings.to_vec());
        }
    }

    // Wait for the spawned task to complete
    processing_task.await?;

    if adapter.is_some() {
        Ok(None)
    } else {
        Ok(Some(all_embeddings))
    }
}

async fn process_images<E: EmbedImage>(
    image_buffer: &[String],
    embedder: Arc<E>,
) -> Result<Arc<Vec<EmbedData>>> {
    let embeddings = embedder.embed_image_batch(image_buffer)?;
    Ok(Arc::new(embeddings))
}

/// Embeds text from files in a directory using the specified embedding model.
///
/// # Arguments
///
/// * `directory` - A `PathBuf` representing the directory containing the files to embed.
/// * `embedder` - A reference to the embedding model to use.
/// * `extensions` - An optional vector of strings representing the file extensions to consider for embedding. If `None`, all files in the directory will be considered.
/// * `config` - An optional `TextEmbedConfig` object specifying the configuration for the embedding model.
/// * `adapter` - An optional callback function to handle the embeddings.
///
/// # Returns
/// An `Option` containing a vector of `EmbedData` objects representing the embeddings of the files, or `None` if an adapter is used.
///
/// # Errors
/// Returns a `Result` with an error if the embedding process fails.
///
/// # Example
///
/// ```rust
/// use embed_anything::embed_directory_stream;
/// use std::path::PathBuf;
/// use std::sync::Arc;
/// use embed_anything::config::TextEmbedConfig;
/// use embed_anything::embeddings::embed::Embedder;
///
/// async fn generate_embeddings() {
///     let directory = PathBuf::from("/path/to/directory");
///     let embedder = Arc::new(Embedder::from_pretrained_hf("clip", "openai/clip-vit-base-patch16", None, None, None).unwrap());
///     let config = Some(&TextEmbedConfig::default());
///     let extensions = Some(vec!["txt".to_string(), "pdf".to_string()]);
///     let embeddings = embed_directory_stream(directory, &embedder, extensions, config, None).await.unwrap();
/// }
/// ```
/// This will output the embeddings of the files in the specified directory using the specified embedding model.
pub async fn embed_directory_stream(
    directory: PathBuf,
    embedder: &Arc<Embedder>,
    extensions: Option<Vec<String>>,
    config: Option<&TextEmbedConfig>,
    adapter: Option<Box<dyn FnMut(Vec<EmbedData>) + Send + Sync>>,
) -> Result<Option<Vec<EmbedData>>> {
    println!("Embedding directory: {:?}", directory);

    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);
    let buffer_size = config.buffer_size.unwrap_or(binding.buffer_size.unwrap());
    let batch_size = config.batch_size;
    let mut file_parser = FileParser::new();
    file_parser.get_text_files(&directory, extensions)?;
    let files = file_parser.files.clone();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let (collector_tx, mut collector_rx) = mpsc::unbounded_channel();

    let embedder = embedder.clone();
    let files: Vec<_> = files.into_iter().collect();
    let pb = indicatif::ProgressBar::new(files.len() as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )?,
    );

    let processing_task = tokio::spawn({
        async move {
            let mut chunk_buffer = Vec::with_capacity(buffer_size);
            let mut metadata_buffer = Vec::with_capacity(buffer_size);
            let mut files_processed: std::collections::HashSet<String> =
                std::collections::HashSet::new();

            while let Some((chunk, metadata)) = rx.recv().await {
                chunk_buffer.push(chunk);
                metadata_buffer.push(metadata);

                if chunk_buffer.len() == buffer_size {
                    match process_chunks(&chunk_buffer, &metadata_buffer, &embedder, batch_size)
                        .await
                    {
                        Ok(embeddings) => {
                            let files = embeddings
                                .iter()
                                .cloned()
                                .map(|e| e.metadata.unwrap().get("file_name").unwrap().to_string())
                                .collect::<Vec<_>>();

                            let unique_files = files.into_iter().unique().collect::<Vec<_>>();
                            let old_len = files_processed.len() as u64;
                            files_processed.extend(unique_files);
                            let new_len = files_processed.len() as u64;

                            pb.inc(new_len - old_len);

                            if let Err(e) = collector_tx.send(embeddings) {
                                eprintln!("Error sending embeddings to collector: {:?}", e);
                            }
                        }
                        Err(e) => eprintln!("Error processing chunks: {:?}", e),
                    }

                    chunk_buffer.clear();
                    metadata_buffer.clear();
                }
            }

            // Process any remaining chunks
            if !chunk_buffer.is_empty() {
                match process_chunks(&chunk_buffer, &metadata_buffer, &embedder, batch_size).await {
                    Ok(embeddings) => {
                        let files = embeddings
                            .iter()
                            .cloned()
                            .map(|e| e.metadata.unwrap().get("file_name").unwrap().to_string())
                            .collect::<Vec<_>>();
                        let unique_files = files.into_iter().unique().collect::<Vec<_>>();
                        let old_len = files_processed.len() as u64;
                        files_processed.extend(unique_files);
                        let new_len = files_processed.len() as u64;

                        pb.inc(new_len - old_len);

                        if let Err(e) = collector_tx.send(embeddings) {
                            eprintln!("Error sending embeddings to collector: {:?}", e);
                        }
                    }
                    Err(e) => eprintln!("Error processing chunks: {:?}", e),
                }
            }
        }
    });

    files.into_iter().for_each(|file| {
        let file_extension = file.as_ref().extension().unwrap();
        let processor = get_processor(file_extension.into(), Some(config.clone())).unwrap();

        let bytes = fs::read(&file).unwrap();
        let out = String::from_utf8_lossy(&bytes);
        let document = processor.process_document(out.as_ref());

        let metadata = TextLoader::get_metadata(file).unwrap();
        let chunks = document.chunks.into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();

        for chunk in chunks {
            if let Err(e) = tx.send((chunk, Some(metadata.clone()))) {
                eprintln!("Error sending chunk: {:?}", e);
            }
        }
    });

    drop(tx);

    let mut all_embeddings = Vec::new();
    let mut adapter = adapter;
    while let Some(embeddings) = collector_rx.recv().await {
        if let Some(adapter) = adapter.as_mut() {
            adapter(embeddings.to_vec());
        } else {
            all_embeddings.extend(embeddings.to_vec());
        }
    }
    // Wait for the spawned task to complete
    processing_task.await?;

    if adapter.is_some() {
        Ok(None)
    } else {
        Ok(Some(all_embeddings))
    }
}

/// Embeds a list of files.
///
/// # Arguments
///
/// * `files` - A vector of `PathBuf` objects representing the files to embed.
/// * `embedder` - A reference to the embedding model to use.
/// * `config` - An optional `TextEmbedConfig` object specifying the configuration for the embedding model.
/// * `adapter` - An optional callback function to handle the embeddings.
///
/// # Returns
/// An `Option` containing a vector of `EmbedData` objects representing the embeddings of the files, or `None` if an adapter is used.
///
/// # Errors
/// Returns a `Result` with an error if the embedding process fails.
///
/// # Example
///
/// ```rust
/// use embed_anything::embed_files_batch;
/// use std::path::PathBuf;
/// use std::sync::Arc;
/// use embed_anything::config::TextEmbedConfig;
/// use embed_anything::embeddings::embed::Embedder;
///
/// async fn generate_embeddings() {
///     let files = vec![PathBuf::from("test_files/test.txt"), PathBuf::from("test_files/test.pdf")];
///     let embedder = Arc::new(Embedder::from_pretrained_hf("bert", "jinaai/jina-embeddings-v2-small-en", None, None, None).unwrap());
///     let config = Some(&TextEmbedConfig::default());
///     let embeddings = embed_files_batch(files, &embedder, config, None).await.unwrap();
/// }
/// ```
/// This will output the embeddings of the files in the specified directory using the specified embedding model.
pub async fn embed_files_batch(
    files: impl IntoIterator<Item = impl AsRef<std::path::Path>>,
    embedder: &Arc<Embedder>,
    config: Option<&TextEmbedConfig>,
    adapter: Option<Box<dyn FnMut(Vec<EmbedData>) + Send + Sync>>,
) -> Result<Option<Vec<EmbedData>>>
{
    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);
    let buffer_size = config.buffer_size.unwrap_or(binding.buffer_size.unwrap());
    let batch_size = config.batch_size;

    let (tx, mut rx) = mpsc::unbounded_channel();
    let (collector_tx, mut collector_rx) = mpsc::unbounded_channel();

    let embedder = embedder.clone();
    let files: Vec<_> = files.into_iter().collect();
    let pb = indicatif::ProgressBar::new(files.len() as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )?,
    );

    let processing_task = tokio::spawn({
        async move {
            let mut chunk_buffer = Vec::with_capacity(buffer_size);
            let mut metadata_buffer = Vec::with_capacity(buffer_size);
            let mut files_processed: std::collections::HashSet<String> =
                std::collections::HashSet::new();

            while let Some((chunk, metadata)) = rx.recv().await {
                chunk_buffer.push(chunk);
                metadata_buffer.push(metadata);

                if chunk_buffer.len() == buffer_size {
                    match process_chunks(&chunk_buffer, &metadata_buffer, &embedder, batch_size)
                        .await
                    {
                        Ok(embeddings) => {
                            let files = embeddings
                                .iter()
                                .cloned()
                                .map(|e| e.metadata.unwrap().get("file_name").unwrap().to_string())
                                .collect::<Vec<_>>();

                            let unique_files = files.into_iter().unique().collect::<Vec<_>>();
                            let old_len = files_processed.len() as u64;
                            files_processed.extend(unique_files);
                            let new_len = files_processed.len() as u64;

                            pb.inc(new_len - old_len);

                            if let Err(e) = collector_tx.send(embeddings) {
                                eprintln!("Error sending embeddings to collector: {:?}", e);
                            }
                        }
                        Err(e) => eprintln!("Error processing chunks: {:?}", e),
                    }

                    chunk_buffer.clear();
                    metadata_buffer.clear();
                }
            }

            // Process any remaining chunks
            if !chunk_buffer.is_empty() {
                match process_chunks(&chunk_buffer, &metadata_buffer, &embedder, batch_size).await {
                    Ok(embeddings) => {
                        let files = embeddings
                            .iter()
                            .cloned()
                            .map(|e| e.metadata.unwrap().get("file_name").unwrap().to_string())
                            .collect::<Vec<_>>();
                        let unique_files = files.into_iter().unique().collect::<Vec<_>>();
                        let old_len = files_processed.len() as u64;
                        files_processed.extend(unique_files);
                        let new_len = files_processed.len() as u64;

                        pb.inc(new_len - old_len);

                        if let Err(e) = collector_tx.send(embeddings) {
                            eprintln!("Error sending embeddings to collector: {:?}", e);
                        }
                    }
                    Err(e) => eprintln!("Error processing chunks: {:?}", e),
                }
            }
        }
    });

    files.into_iter().for_each(|file| {
        let file_extension = file.as_ref().extension().unwrap();
        let processor = get_processor(file_extension.into(), Some(config.clone())).unwrap();

        let bytes = fs::read(&file).unwrap();
        let out = String::from_utf8_lossy(&bytes);
        let document = processor.process_document(out.as_ref());

        let metadata = TextLoader::get_metadata(file).unwrap();
        let chunks = document.chunks.into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();

        for chunk in chunks {
            if let Err(e) = tx.send((chunk, Some(metadata.clone()))) {
                eprintln!("Error sending chunk: {:?}", e);
            }
        }
    });

    drop(tx);

    let mut all_embeddings = Vec::new();
    let mut adapter = adapter;
    while let Some(embeddings) = collector_rx.recv().await {
        if let Some(adapter) = adapter.as_mut() {
            adapter(embeddings.to_vec());
        } else {
            all_embeddings.extend(embeddings.to_vec());
        }
    }
    // Wait for the spawned task to complete
    processing_task.await?;

    if adapter.is_some() {
        Ok(None)
    } else {
        Ok(Some(all_embeddings))
    }
}

pub async fn process_chunks(
    chunks: &[String],
    metadata: &[Option<HashMap<String, String>>],
    embedding_model: &Arc<Embedder>,
    batch_size: Option<usize>,
) -> Result<Arc<Vec<EmbedData>>> {
    let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
    let encodings = embedding_model.embed(&chunk_refs, batch_size).await?;

    // zip encodings with chunks and metadata
    let embeddings = encodings
        .into_iter()
        .zip(chunks)
        .zip(metadata)
        .map(|((encoding, chunk), metadata)| {
            EmbedData::new(encoding.clone(), Some(chunk.clone()), metadata.clone())
        })
        .collect::<Vec<_>>();
    Ok(Arc::new(embeddings))
}

fn get_processor(
    file_extension: &OsStr,
    config: Option<TextEmbedConfig>
) -> Result<impl DocumentProcessor> {
    let config = config.unwrap_or(TextEmbedConfig::default());
    let chunk_size = config.chunk_size.unwrap_or(1000);
    let overlap_ratio = config.overlap_ratio.unwrap_or(0.0);
    let splitting_strategy = config.splitting_strategy.clone();
    let use_ocr = config.use_ocr.unwrap_or(false);
    let tesseract_path = config.tesseract_path.clone();

    match file_extension.to_str().unwrap() {
        //"pdf" => PdfProcessor::new(chunk_size, overlap_ratio, splitting_strategy, use_ocr, tesseract_path),
        "md" => {
            Ok(MarkdownProcessor::new(chunk_size))
        },
        "txt" => {
            Ok(TxtProcessor::new(chunk_size, overlap_ratio, splitting_strategy))
        },
        "docx" => {
            Ok(DocxProcessor::new(chunk_size))
        },
        _ => {
            Err(FileLoadingError::UnsupportedFileType(file_extension.to_str().unwrap().to_string()).into())
        },
    }
}

async fn embed_document(document: Document, metadata: Option<HashMap<String, String>>, embedder: &TextEmbedder, batch_size: Option<usize>) -> Result<Vec<EmbedData>> {
    let chunks = document.chunks;
    let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_ref()).collect();

    let encodings = embedder
        .embed(&chunk_refs, batch_size)
        .await?;

    get_text_metadata(&Rc::new(encodings), &chunk_refs, &metadata)
}
