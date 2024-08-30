//! # Embed Anything
//! This library provides a simple interface to embed text and images using various embedding models.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

pub mod chunkers;
pub mod config;
pub mod embeddings;
pub mod file_loader;
pub mod file_processor;
pub mod text_loader;

use std::{collections::HashMap, fs, path::PathBuf};

use anyhow::{anyhow, Result};
use config::{BertConfig, CloudConfig, EmbedConfig, JinaConfig, TextEmbedConfig};
use embeddings::{
    cloud::{cohere::CohereEmbeder, openai::OpenAIEmbeder},
    embed::{CloudEmbeder, EmbedData, EmbedImage, Embeder, TextEmbed},
    embed_audio, get_text_metadata,
    local::{bert::BertEmbeder, jina::JinaEmbeder},
};
use file_loader::FileParser;
use file_processor::audio::audio_processor;
use text_loader::TextLoader;

fn get_bert_embeder(config: &BertConfig) -> Result<BertEmbeder> {
    let model_id = &config
        .model_id
        .clone()
        .unwrap_or_else(|| "sentence-transformers/all-MiniLM-L12-v2".to_string());
    let revision = &config.revision;
    let embeder = if let Some(revision) = revision {
        BertEmbeder::new(model_id.to_string(), Some(revision.to_string()))?
    } else {
        BertEmbeder::new(model_id.to_string(), None)?
    };
    Ok(embeder)
}

fn get_cloud_embeder(config: &CloudConfig) -> Result<CloudEmbeder> {
    let embeder = match &config.provider {
        Some(provider) => match provider.as_str() {
            "OpenAI" => {
                let model = &config
                    .model
                    .clone()
                    .unwrap_or_else(|| "text-embedding-3-small".to_string());
                let api_key = &config.api_key;
                if let Some(api_key) = api_key {
                    CloudEmbeder::OpenAI(OpenAIEmbeder::new(
                        model.to_string(),
                        Some(api_key.to_string()),
                    ))
                } else {
                    CloudEmbeder::OpenAI(OpenAIEmbeder::new(model.to_string(), None))
                }
            }
            "Cohere" => {
                let model = &config
                    .model
                    .clone()
                    .unwrap_or_else(|| "embed-english-v3.0".to_string());
                let api_key = &config.api_key;
                if let Some(api_key) = api_key {
                    CloudEmbeder::Cohere(CohereEmbeder::new(
                        model.to_string(),
                        Some(api_key.to_string()),
                    ))
                } else {
                    CloudEmbeder::Cohere(CohereEmbeder::new(model.to_string(), None))
                }
            }
            _ => {
                return anyhow::Result::Err(anyhow::anyhow!(
                    "Invalid provider. Choose between OpenAI and Cohere."
                ))
            }
        },
        None => {
            return anyhow::Result::Err(anyhow::anyhow!(
                "Provide the provider for the cloud embedding model.",
            ))
        }
    };

    Ok(embeder)
}

fn get_jina_embeder(config: &JinaConfig) -> Result<JinaEmbeder> {
    let model_id = &config
        .model_id
        .clone()
        .unwrap_or_else(|| "jinaai/jina-embeddings-v2-base-en".to_string());
    let revision = &config.revision;
    let embeder = if let Some(revision) = revision {
        JinaEmbeder::new(model_id.to_string(), Some(revision.to_string()))?
    } else {
        JinaEmbeder::new(model_id.to_string(), None)?
    };
    Ok(embeder)
}

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

pub fn embed_query(
    query: Vec<String>,
    embeder: &Embeder,
    config: TextEmbedConfig,
) -> Result<Vec<EmbedData>> {
    let encodings = embeder.embed(&query, config.batch_size)?;
    let embeddings = get_text_metadata(&encodings, &query, None)?;

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

pub fn embed_file<F>(
    file_name: &str,
    embeder: &Embeder,
    config: Option<&TextEmbedConfig>,
    adapter: Option<F>,
) -> Result<Option<Vec<EmbedData>>>
where
    F: Fn(Vec<EmbedData>),
{
    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);
    let chunk_size = config.chunk_size.unwrap_or(256);
    let batch_size = config.batch_size; 

    let embeddings = match embeder {
        Embeder::OpenAI(embeder) => {
            emb_text(file_name, embeder, Some(chunk_size), None, adapter)?
        }
        Embeder::Cohere(embeder) => {
            emb_text(file_name, embeder, Some(chunk_size), None, adapter)?
        }
        Embeder::Jina(embeder) => {
            emb_text(
                file_name,
                embeder,
                Some(chunk_size),
                batch_size,
                adapter,
            )?
        }
        Embeder::Bert(embeder) => {
            emb_text(
                file_name,
                embeder,
                Some(chunk_size),
                batch_size,
                adapter,
            )?
        }
        Embeder::Clip(embeder) => Some(vec![emb_image(file_name, embeder).unwrap()]),
    };

    Ok(embeddings)
}

/// Embeds the text from files in a directory using the specified embedding model.
///
/// # Arguments
///
/// * `directory` - A `PathBuf` representing the directory containing the files to embed.
/// * `embeder` - A string specifying the embedding model to use. Valid options are "OpenAI", "Jina", "Clip", and "Bert".
/// * `extensions` - An optional vector of strings representing the file extensions to consider for embedding. If `None`, all files in the directory will be considered.
/// * `config` - An optional `EmbedConfig` object specifying the configuration for the embedding model.
/// * 'adapter' - An optional `Adapter` object to send the embeddings to a vector database.
///
/// # Returns
/// A vector of `EmbedData` objects representing the embeddings of the files.
///
/// # Errors
/// Returns a `PyValueError` if an invalid embedding model is specified.
///
/// # Example
///
/// ```rust
/// use embed_anything::embed_directory;
/// use std::path::PathBuf;
///
/// let directory = PathBuf::from("/path/to/directory");
/// let embeder = "OpenAI";
/// let bert_config = BertConfig{ model_id: Some("sentence-transformers/all-MiniLM-L12-v2".to_string()), revision: None, chunk_size: Some(256) };
/// let config = EmbedConfig{ bert: Some(bert_config), ..Default::default() };
/// let extensions = Some(vec!["txt".to_string(), "pdf".to_string()]);
/// let embeddings = embed_directory(directory, embeder, extensions, config).unwrap();
/// ```
/// This will output the embeddings of the files in the specified directory using the OpenAI embedding model.
pub fn embed_directory<F>(
    directory: PathBuf,
    embeder: &Embeder,
    extensions: Option<Vec<String>>,
    config: Option<&TextEmbedConfig>,
    adapter: Option<F>,
) -> Result<Option<Vec<EmbedData>>>
where
    F: Fn(Vec<EmbedData>) + Copy,
{
    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);
    let chunk_size = config.chunk_size.unwrap_or(256);

    let embeddings = match embeder {
        Embeder::Clip(embeder) => emb_image_directory(directory, embeder),
        _ => emb_directory(
            directory,
            embeder,
            extensions,
            Some(chunk_size),
            config.batch_size,
            adapter,
        ),
    }
    .unwrap();
    Ok(embeddings)
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

pub fn embed_webpage<F>(
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

    match embeder {
        Embeder::Clip(_) => panic!("Clip model not supported for webpages"),
        _ => {}
    }

    let binding = TextEmbedConfig::default();
    let config = config.unwrap_or(&binding);
    let chunk_size = config.chunk_size.unwrap_or(256);
    let batch_size = config.batch_size;

    let embeddings = webpage.embed_webpage(embeder, chunk_size, batch_size)?;

    // Send embeddings to vector database
    if let Some(adapter) = adapter {
        adapter(embeddings);
        Ok(None)
    } else {
        Ok(Some(embeddings))
    }
}

fn emb_directory<F>(
    directory: PathBuf,
    embedding_model: &Embeder,
    extensions: Option<Vec<String>>,
    chunk_size: Option<usize>,
    batch_size: Option<usize>,
    adapter: Option<F>,
) -> Result<Option<Vec<EmbedData>>>
where
    F: Fn(Vec<EmbedData>) + Copy,
{
    let mut file_parser = FileParser::new();
    file_parser.get_text_files(&directory, extensions).unwrap();

    match adapter {
        Some(adapter) => {
            let _ = file_parser
                .files
                .into_iter()
                .filter_map(|file| {
                    emb_text(file, embedding_model, chunk_size, batch_size, Some(adapter)).unwrap()
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            Ok(None)
        }
        None => {
            let embeddings = file_parser
                .files
                .into_iter()
                .map(|file| {
                    emb_text(
                        file,
                        embedding_model,
                        chunk_size,
                        batch_size,
                        None::<fn(Vec<EmbedData>)>,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            Ok(Some(embeddings))
        }
    }
}

fn emb_text<T: AsRef<std::path::Path>, F, E: TextEmbed>(
    file: T,
    embedding_model: &E,
    chunk_size: Option<usize>,
    batch_size: Option<usize>,
    adapter: Option<F>,
) -> Result<Option<Vec<EmbedData>>>
where
    F: Fn(Vec<EmbedData>),
{
    let text = TextLoader::extract_text(file.as_ref().to_str().unwrap()).unwrap();
    let textloader = TextLoader::new(chunk_size.unwrap_or(256));
    let chunks = textloader.split_into_chunks(&text);
    let metadata = TextLoader::get_metadata(file).ok();

    if let Some(adapter) = adapter {
        let embeddings = chunks
            .map(|chunks| {
                let encodings = embedding_model.embed(&chunks, batch_size).unwrap();
                get_text_metadata(&encodings, &chunks, metadata).unwrap()
            })
            .ok_or_else(|| anyhow::anyhow!("No text found in file"))?;
        adapter(embeddings);
        Ok(None)
    } else {
        let embeddings = chunks
            .map(|chunks| {
                let encodings = embedding_model.embed(&chunks, batch_size).unwrap();
                get_text_metadata(&encodings, &chunks, metadata).unwrap()
            })
            .ok_or_else(|| anyhow!("No text found in file"))?;

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

pub fn emb_audio<T: AsRef<std::path::Path>>(
    audio_file: T,
    config: &EmbedConfig,
) -> Result<Option<Vec<EmbedData>>> {
    let model_input = if let Some(audio_decoder_config) = &config.audio_decoder {
        audio_processor::build_model(
            audio_decoder_config.decoder_model_id.clone(),
            audio_decoder_config.decoder_revision.clone(),
            audio_decoder_config.quantized.unwrap_or(false),
            audio_decoder_config
                .model_type
                .clone()
                .unwrap_or("tiny-en".to_string())
                .as_str(),
        )
        .unwrap()
    } else {
        // error
        return  anyhow::Result::Err(anyhow::anyhow!(
            "Provide the config for the audio decoder model. Otherwise, use the embed_audio function without the config parameter.",
        ));
    };
    let segments: Vec<audio_processor::Segment> =
        audio_processor::process_audio(&audio_file, model_input).unwrap();

    let embeddings = if let Some(bert_config) = &config.bert {
        let embeder = get_bert_embeder(bert_config).unwrap();
        embed_audio(&embeder, segments, audio_file).unwrap()
    } else if let Some(cloud_config) = &config.cloud {
        let embeder = get_cloud_embeder(cloud_config).unwrap();
        embed_audio(&embeder, segments, audio_file).unwrap()
    } else if let Some(jina_config) = &config.jina {
        let embeder = get_jina_embeder(jina_config).unwrap();
        embed_audio(&embeder, segments, audio_file).unwrap()
    } else {
        // error
        return  anyhow::Result::Err(anyhow::anyhow!(
            "Provide the config for the text embedding model. Otherwise, use the embed_audio function without the config parameter.",
        ));
    };

    Ok(Some(embeddings))
}

fn emb_image_directory<T: EmbedImage>(
    directory: PathBuf,
    embedding_model: &T,
) -> Result<Option<Vec<EmbedData>>> {
    let mut file_parser = FileParser::new();
    file_parser.get_image_paths(&directory).unwrap();

    let embeddings = embedding_model
        .embed_image_batch(&file_parser.files)
        .unwrap();
    Ok(Some(embeddings))
}
