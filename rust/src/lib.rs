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
//! let clip_embedder = Embedder::from_pretrained_hf("CLIP", "jina-clip-v2", None);
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
use tokio::sync::mpsc; // Add this at the top of your file

#[cfg(feature = "audio")]
use embeddings::embed_audio;
use crate::config::SplittingStrategy;
use crate::file_processor::markdown_processor::MarkdownProcessor;

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
/// let query = vec!["Hello".to_string(), "World".to_string()];
/// let embedder = Embedder::Text(TextEmbedder::Jina(Box::new(JinaEmbedder::default())));
/// let embeddings = embed_query(query, &embedder, None).unwrap();
/// println!("{:?}", embeddings);
/// ```
pub async fn embed_query(
    query: Vec<String>,
    embedder: &Embedder,
    config: Option<&TextEmbedConfig>,
) -> Result<Vec<EmbedData>> {
    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);
    let _chunk_size = config.chunk_size.unwrap_or(256);
    let batch_size = config.batch_size;

    let encodings = embedder.embed(&query, batch_size).await?;
    let embeddings = get_text_metadata(&Rc::new(encodings), &query, &None)?;

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
    config: Option<&TextEmbedConfig>,
    // Callback function
    adapter: Option<Box<dyn FnOnce(Vec<EmbedData>)>>,
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
pub async fn embed_webpage<F>(
    url: String,
    embedder: &Embedder,
    config: Option<&TextEmbedConfig>,
    // Callback function
    adapter: Option<F>,
) -> Result<Option<Vec<EmbedData>>>
where
    F: Fn(Vec<EmbedData>),
{
    let website_processor = file_processor::website_processor::WebsiteProcessor::new();
    let webpage = website_processor.process_website(url.as_ref())?;

    // if let Embedder::Clip(_) = embedder {
    //     return Err(anyhow!("Clip model does not support webpage embedding"));
    // }

    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);
    let chunk_size = config.chunk_size.unwrap_or(256);
    let overlap_ratio = config.overlap_ratio.unwrap_or(0.0);
    let batch_size = config.batch_size;

    let embeddings = webpage
        .embed_webpage(embedder, chunk_size, overlap_ratio, batch_size)
        .await?;

    // Send embeddings to vector database
    if let Some(adapter) = adapter {
        adapter(embeddings);
        Ok(None)
    } else {
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
///         Some("https://example.com/"),
///         &Embedder::from_pretrained_hf("JINA", "jinaai/jina-embeddings-v2-small-en", None).unwrap(),
///         None,
///         None,
///     ).await.unwrap();
/// }
/// ```
pub async fn embed_html(
    file_name: impl AsRef<std::path::Path>,
    origin: Option<impl Into<String>>,
    embedder: &Embedder,
    config: Option<&TextEmbedConfig>,
    // Callback function
    adapter: Option<Box<dyn FnOnce(Vec<EmbedData>)>>,
) -> Result<Option<Vec<EmbedData>>> {
    let html_processor = file_processor::html_processor::HtmlProcessor::new();
    let html = html_processor.process_html_file(file_name.as_ref(), origin)?;

    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);
    let chunk_size = config.chunk_size.unwrap_or(256);
    let overlap_ratio = config.overlap_ratio.unwrap_or(0.0);
    let batch_size = config.batch_size;

    let embeddings = html
        .embed_webpage(embedder, chunk_size, overlap_ratio, batch_size)
        .await?;

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
///         &Embedder::from_pretrained_hf("JINA", "jinaai/jina-embeddings-v2-small-en", None, None, None).unwrap(),
///         None,
///     ).await.unwrap();
/// }
/// ```
pub async fn embed_markdown(
    markdown: impl Into<String>,
    embedder: &Embedder,
    config: Option<&TextEmbedConfig>,
) -> Result<Vec<EmbedData>> {
    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);

    let md_processor = MarkdownProcessor::new();
    let md = md_processor.process_markdown(markdown.into())?;

    md.embed_markdown(
        &embedder,
        config.chunk_size.unwrap_or(256),
        config.batch_size,
    ).await
}

#[allow(clippy::too_many_arguments)]
async fn emb_text(
    file: impl AsRef<std::path::Path>,
    embedding_model: &TextEmbedder,
    config: Option<&TextEmbedConfig>,
    // Callback function
    adapter: Option<Box<dyn FnOnce(Vec<EmbedData>)>>,
) -> Result<Option<Vec<EmbedData>>> {
    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);
    let chunk_size = config.chunk_size.unwrap_or(256);
    let overlap_ratio = config.overlap_ratio.unwrap_or(0.0);
    let batch_size = config.batch_size;
    let splitting_strategy = config.splitting_strategy.clone();
    let use_ocr = config.use_ocr.unwrap_or(false);
    let tesseract_path = config.tesseract_path.clone();
    let text = TextLoader::extract_text(&file, use_ocr, tesseract_path.as_deref())?;
    let textloader = TextLoader::new(chunk_size, overlap_ratio);
    let chunks = textloader
        .split_into_chunks(&text, splitting_strategy)
        .unwrap_or_default();

    let metadata = TextLoader::get_metadata(file).ok();

    if let Some(adapter) = adapter {
        let encodings = embedding_model.embed(&chunks, batch_size).await.unwrap();
        let embeddings = get_text_metadata(&Rc::new(encodings), &chunks, &metadata).unwrap();
        adapter(embeddings);
        Ok(None)
    } else {
        let encodings = embedding_model.embed(&chunks, batch_size).await.unwrap();
        let embeddings = get_text_metadata(&Rc::new(encodings), &chunks, &metadata).unwrap();

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
        .embed_image(&image_path, Some(metadata))
        .unwrap();

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

    let segments: Vec<audio_processor::Segment> = audio_decoder.process_audio(&audio_file).unwrap();
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
///     let embedder = Arc::new(Embedder::from_pretrained_hf("clip", "openai/clip-vit-base-patch16", None).unwrap());
///     let embeddings = embed_image_directory(directory, &embedder, None, None).await.unwrap();
/// }
/// ```
/// This will output the embeddings of the images in the specified directory using the specified embedding model.
///
pub async fn embed_image_directory<T: EmbedImage + Send + Sync + 'static, F>(
    directory: PathBuf,
    embedding_model: &Arc<T>,
    config: Option<&ImageEmbedConfig>,
    adapter: Option<F>,
) -> Result<Option<Vec<EmbedData>>>
where
    F: Fn(Vec<EmbedData>),
{
    let mut file_parser = FileParser::new();
    file_parser.get_image_paths(&directory).unwrap();

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
        )
        .unwrap(),
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
    while let Some(embeddings) = collector_rx.recv().await {
        if let Some(adapter) = &adapter {
            adapter(embeddings.to_vec());
        } else {
            all_embeddings.extend(embeddings.to_vec());
        }
    }

    // Wait for the spawned task to complete
    processing_task.await.unwrap();

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
///     let embedder = Arc::new(Embedder::from_pretrained_hf("clip", "openai/clip-vit-base-patch16", None).unwrap());
///     let config = Some(&TextEmbedConfig::default());
///     let extensions = Some(vec!["txt".to_string(), "pdf".to_string()]);
///     let embeddings = embed_directory_stream(directory, &embedder, extensions, config, None).await.unwrap();
/// }
/// ```
/// This will output the embeddings of the files in the specified directory using the specified embedding model.
pub async fn embed_directory_stream<F>(
    directory: PathBuf,
    embedder: &Arc<Embedder>,
    extensions: Option<Vec<String>>,
    config: Option<&TextEmbedConfig>,
    adapter: Option<F>,
) -> Result<Option<Vec<EmbedData>>>
where
    F: Fn(Vec<EmbedData>),
{
    println!("Embedding directory: {:?}", directory);

    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);
    let chunk_size = config.chunk_size.unwrap_or(binding.chunk_size.unwrap());
    let buffer_size = config.buffer_size.unwrap_or(binding.buffer_size.unwrap());
    let batch_size = config.batch_size;
    let use_ocr = config.use_ocr.unwrap_or(false);
    let tesseract_path = config.tesseract_path.as_deref();
    let overlap_ratio = config.overlap_ratio.unwrap_or(0.0);
    let mut file_parser = FileParser::new();
    file_parser.get_text_files(&directory, extensions)?;
    let files = file_parser.files.clone();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let (collector_tx, mut collector_rx) = mpsc::unbounded_channel();

    let embedder = embedder.clone();
    let pb = indicatif::ProgressBar::new(files.len() as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )
        .unwrap(),
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

    let textloader = TextLoader::new(chunk_size, overlap_ratio);

    file_parser.files.iter().for_each(|file| {
        let text = match TextLoader::extract_text(file, use_ocr, tesseract_path) {
            Ok(text) => text,
            Err(_) => {
                return;
            }
        };
        let chunks = textloader
            .split_into_chunks(&text, SplittingStrategy::Sentence)
            .unwrap_or_else(|| vec![text.clone()])
            .into_iter()
            .filter(|chunk| !chunk.trim().is_empty())
            .collect::<Vec<_>>();
        if chunks.is_empty() {
            return;
        }
        let metadata = TextLoader::get_metadata(file).unwrap();
        for chunk in chunks {
            if let Err(e) = tx.send((chunk, Some(metadata.clone()))) {
                eprintln!("Error sending chunk: {:?}", e);
            }
        }
    });

    drop(tx);

    let mut all_embeddings = Vec::new();
    while let Some(embeddings) = collector_rx.recv().await {
        if let Some(adapter) = &adapter {
            adapter(embeddings.to_vec());
        } else {
            all_embeddings.extend(embeddings.to_vec());
        }
    }
    // Wait for the spawned task to complete
    processing_task.await.unwrap();

    if adapter.is_some() {
        Ok(None)
    } else {
        Ok(Some(all_embeddings))
    }
}

pub async fn process_chunks(
    chunks: &Vec<String>,
    metadata: &Vec<Option<HashMap<String, String>>>,
    embedding_model: &Arc<Embedder>,
    batch_size: Option<usize>,
) -> Result<Arc<Vec<EmbedData>>> {
    let encodings = embedding_model.embed(chunks, batch_size).await?;

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
