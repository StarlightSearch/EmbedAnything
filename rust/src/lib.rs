//! # Embed Anything
//! This library provides a simple interface to embed text and images using various embedding models.
pub mod chunkers;
pub mod config;
pub mod embeddings;
pub mod file_loader;
pub mod file_processor;
pub mod text_loader;

use std::{collections::HashMap, fs, path::PathBuf, sync::Arc};

use anyhow::Result;
use config::{ImageEmbedConfig, TextEmbedConfig};
use embeddings::{
    embed::{EmbedData, EmbedImage, Embeder},
    embed_audio, get_text_metadata,
};
use file_loader::FileParser;
use file_processor::audio::audio_processor::{self, AudioDecoderModel};
use futures::future::join_all;
use itertools::Itertools;
use rayon::prelude::*;
use text_loader::{SplittingStrategy, TextLoader};
use tokio::sync::mpsc; // Add this at the top of your file

/// Embeds a list of queries using the specified embedding model.
///
/// # Arguments
///
/// * `query` - A vector of strings representing the queries to embed.
/// * `embeder` - A string specifying the embedding model to use. Valid options are "OpenAI", "Jina", "Clip", and "Bert".
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
///
/// let query = vec!["Hello".to_string(), "World".to_string()];
/// let embeder = "OpenAI";
/// let openai_config = OpenAIConfig{ model: Some("text-embedding-3-small".to_string()), api_key: None, chunk_size: Some(256) };
/// let config = EmbedConfig{ openai: Some(openai_config), ..Default::default() };
/// let embeddings = embed_query(query, embeder).unwrap();
/// println!("{:?}", embeddings);
/// ```
/// This will output the embeddings of the queries using the OpenAI embedding model.

pub async fn embed_query(
    query: Vec<String>,
    embeder: &Embeder,
    config: Option<&TextEmbedConfig>,
) -> Result<Vec<EmbedData>> {
    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);
    let _chunk_size = config.chunk_size.unwrap_or(256);
    let batch_size = config.batch_size;

    let encodings = embeder.embed(&query, batch_size).await.unwrap();
    let embeddings = get_text_metadata(&encodings, &query, &None)?;

    Ok(embeddings)
}
/// Embeds the text from a file using the specified embedding model.
///
/// # Arguments
///
/// * `file_name` - A string specifying the name of the file to embed.
/// * `embeder` - A string specifying the embedding model to use. Valid options are "OpenAI", "Jina", "Clip", and "Bert".
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
///
/// let file_name = "test_files/test.pdf";
/// let embeder = "Bert";
/// let bert_config = BertConfig{ model_id: Some("sentence-transformers/all-MiniLM-L12-v2".to_string()), revision: None, chunk_size: Some(256) };
/// let embeddings = embed_file(file_name, embeder, config).unwrap();
/// ```
/// This will output the embeddings of the file using the OpenAI embedding model.

pub async fn embed_file<F>(
    file_name: &str,
    embeder: &Embeder,
    config: Option<&TextEmbedConfig>,
    adapter: Option<F>,
) -> Result<Option<Vec<EmbedData>>>
where
    F: Fn(Vec<EmbedData>), // Add Send trait bound here
{
    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);
    let chunk_size = config.chunk_size.unwrap_or(256);
    let batch_size = config.batch_size;
    let splitting_strategy = config
        .splitting_strategy
        .unwrap_or(SplittingStrategy::Sentence);
    let semantic_encoder = config.semantic_encoder.clone();

    if let Embeder::Clip(embeder) = embeder {
        return Ok(Some(vec![emb_image(file_name, embeder).unwrap()]));
    } else {
        let embeddings = emb_text(
            file_name,
            embeder,
            Some(chunk_size),
            batch_size,
            Some(splitting_strategy),
            semantic_encoder,
            adapter,
        )
        .await?;
        Ok(embeddings)
    }
}

/// Embeddings of a webpage using the specified embedding model.
///
/// # Arguments
///
/// * `embeder` - The embedding model to use. Supported options are "OpenAI", "Jina", and "Bert".
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
/// let embeddings = match embeder {
///     "OpenAI" => webpage
///         .embed_webpage(&embedding_model::openai::OpenAIEmbeder::default())
///         .unwrap(),
///     "Jina" => webpage
///         .embed_webpage(&embedding_model::jina::JinaEmbeder::default())
///         .unwrap(),
///     "Bert" => webpage
///         .embed_webpage(&embedding_model::bert::BertEmbeder::default())
///         .unwrap(),
///     _ => {
///         return Err(PyValueError::new_err(
///             "Invalid embedding model. Choose between OpenAI and AllMiniLmL12V2.",
///         ))
///     }
/// };
/// ```

pub async fn embed_webpage<F>(
    url: String,
    embeder: &Embeder,
    config: Option<&TextEmbedConfig>,
    // Callback function
    adapter: Option<F>,
) -> Result<Option<Vec<EmbedData>>>
where
    F: Fn(Vec<EmbedData>),
{
    let website_processor = file_processor::website_processor::WebsiteProcessor::new();
    let webpage = website_processor.process_website(url.as_ref())?;

    // if let Embeder::Clip(_) = embeder {
    //     return Err(anyhow!("Clip model does not support webpage embedding"));
    // }

    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);
    let chunk_size = config.chunk_size.unwrap_or(256);
    let batch_size = config.batch_size;

    let embeddings = webpage
        .embed_webpage(embeder, chunk_size, batch_size)
        .await?;

    // Send embeddings to vector database
    if let Some(adapter) = adapter {
        adapter(embeddings);
        Ok(None)
    } else {
        Ok(Some(embeddings))
    }
}

async fn emb_text<T: AsRef<std::path::Path>, F>(
    file: T,
    embedding_model: &Embeder,
    chunk_size: Option<usize>,
    batch_size: Option<usize>,
    splitting_strategy: Option<SplittingStrategy>,
    semantic_encoder: Option<Arc<Embeder>>,
    adapter: Option<F>,
) -> Result<Option<Vec<EmbedData>>>
where
    F: Fn(Vec<EmbedData>), // Add Send trait bound here
{
    println!("Embedding text file: {:?}", file.as_ref());

    let text = TextLoader::extract_text(file.as_ref().to_str().unwrap())?;
    let textloader = TextLoader::new(chunk_size.unwrap_or(256));
    let chunks = textloader.split_into_chunks(
        &text,
        splitting_strategy.unwrap_or(SplittingStrategy::Sentence),
        semantic_encoder,
    );
    let metadata = TextLoader::get_metadata(file).ok();

    if let Some(adapter) = adapter {
        let futures = chunks.iter().map(|chunks| {
            let chunks = chunks.clone();
            let metadata = metadata.clone();
            async move {
                let encodings = embedding_model
                    .embed(&chunks[..], batch_size)
                    .await
                    .unwrap();
                get_text_metadata(&encodings, &chunks, &metadata).unwrap()
            }
        });
        let embeddings = join_all(futures)
            .await
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        adapter(embeddings);
        Ok(None)
    } else {
        let futures = chunks.iter().map(|chunks| {
            let chunks = chunks.clone();
            let metadata = metadata.clone();
            async move {
                let encodings = embedding_model
                    .embed(&chunks[..], batch_size)
                    .await
                    .unwrap();
                get_text_metadata(&encodings, &chunks, &metadata).unwrap()
            }
        });
        let embeddings = join_all(futures)
            .await
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        Ok(Some(embeddings))
    }
}

fn emb_image<T: AsRef<std::path::Path>, U: EmbedImage>(
    image_path: T,
    embedding_model: &U,
) -> Result<EmbedData> {
    let mut metadata = HashMap::new();
    metadata.insert(
        "file_name".to_string(),
        fs::canonicalize(&image_path)?.to_str().unwrap().to_string(),
    );

    let embedding = embedding_model
        .embed_image(&image_path, Some(metadata))
        .unwrap();

    Ok(embedding)
}

pub async fn emb_audio<T: AsRef<std::path::Path>>(
    audio_file: T,
    audio_decoder: &mut AudioDecoderModel,
    embeder: &Embeder,
    text_embed_config: Option<&TextEmbedConfig>,
) -> Result<Option<Vec<EmbedData>>> {
    let segments: Vec<audio_processor::Segment> = audio_decoder.process_audio(&audio_file).unwrap();
    let embeddings = embed_audio(
        embeder,
        segments,
        audio_file,
        text_embed_config
            .unwrap_or(&TextEmbedConfig::default())
            .batch_size,
    )
    .await?;

    Ok(Some(embeddings))
}

/// Embeds images in a directory using the specified embedding model.
///
/// # Arguments
///
/// * `directory` - A `PathBuf` representing the directory containing the images to embed.
/// * `embeder` - A reference to the embedding model to use.
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
///
/// let directory = PathBuf::from("/path/to/directory");
/// let embeder = Arc::new(Embeder::from_pretrained_hf("clip", "openai/clip-vit-base-patch16", None).unwrap());
/// let embeddings = embed_image_directory(directory, &embeder, None).await.unwrap();
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

    let embeder = embedding_model.clone();

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
                    match process_images(&image_buffer, embeder.clone()).await {
                        Ok(embeddings) => {
                            let files = embeddings
                                .to_vec()
                                .into_iter()
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
                match process_images(&image_buffer, embeder.clone()).await {
                    Ok(embeddings) => {
                        let files = embeddings
                            .to_vec()
                            .into_iter()
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
            adapter(embeddings);
        } else {
            all_embeddings.extend(embeddings);
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
    image_buffer: &Vec<String>,
    embeder: Arc<E>,
) -> Result<Vec<EmbedData>> {
    let embeddings = embeder.embed_image_batch(image_buffer)?;
    Ok(embeddings)
}

/// Embeds text from files in a directory using the specified embedding model.
///
/// # Arguments
///
/// * `directory` - A `PathBuf` representing the directory containing the files to embed.
/// * `embeder` - A reference to the embedding model to use.
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
///
/// let directory = PathBuf::from("/path/to/directory");
/// let embeder = Arc::new(Embeder::from_pretrained_hf("clip", "openai/clip-vit-base-patch16", None).unwrap());
/// let config = Some(TextEmbedConfig::default());
/// let extensions = Some(vec!["txt".to_string(), "pdf".to_string()]);
/// let embeddings = embed_directory_stream(directory, &embeder, extensions, config, None).await.unwrap();
/// ```
/// This will output the embeddings of the files in the specified directory using the specified embedding model.
pub async fn embed_directory_stream<F>(
    directory: PathBuf,
    embeder: &Arc<Embeder>,
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
    let semantic_encoder = config.semantic_encoder.clone();

    let mut file_parser = FileParser::new();
    file_parser.get_text_files(&directory, extensions)?;
    let files = file_parser.files.clone();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let (collector_tx, mut collector_rx) = mpsc::unbounded_channel();

    let embeder = embeder.clone();
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
                    match process_chunks(
                        &chunk_buffer,
                        &metadata_buffer,
                        embeder.clone(),
                        batch_size,
                    )
                    .await
                    {
                        Ok(embeddings) => {
                            let files = embeddings
                                .to_vec()
                                .into_iter()
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
                match process_chunks(&chunk_buffer, &metadata_buffer, embeder.clone(), batch_size)
                    .await
                {
                    Ok(embeddings) => {
                        let files = embeddings
                            .to_vec()
                            .into_iter()
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

    let textloader = TextLoader::new(chunk_size);

    file_parser.files.iter().for_each(|file| {
        let text = TextLoader::extract_text(&file.to_string()).unwrap();
        let chunks = textloader
            .split_into_chunks(&text, SplittingStrategy::Sentence, None)
            .unwrap();

        let metadata = TextLoader::get_metadata(&file).unwrap();
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
    embedding_model: Arc<Embeder>,
    batch_size: Option<usize>,
) -> Result<Arc<Vec<EmbedData>>> {
    let encodings = embedding_model.embed(chunks, batch_size).await?;

    // zip encodings with chunks and metadata
    let embeddings = encodings
        .into_iter()
        .zip(chunks)
        .zip(metadata)
        .map(|((encoding, chunk), metadata)| {
            EmbedData::new(encoding.to_vec(), Some(chunk.clone()), metadata.clone())
        })
        .collect::<Vec<_>>();
    Ok(Arc::new(embeddings))
}
