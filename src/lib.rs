//! # Embed Anything
//! This library provides a simple interface to embed text and images using various embedding models.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

pub mod config;
pub mod embedding_model;
pub mod file_loader;
pub mod file_processor;
pub mod text_loader;
use std::path::PathBuf;

use config::{AudioDecoderConfig, BertConfig, ClipConfig, CloudConfig, EmbedConfig, JinaConfig};
use embedding_model::{
    bert::BertEmbeder,
    clip::ClipEmbeder,
    cohere::CohereEmbeder,
    embed::{CloudEmbeder, EmbedData, EmbedImage, Embeder},
    embed_audio, get_text_metadata,
    jina::JinaEmbeder,
    openai::OpenAIEmbeder,
};
use file_loader::FileParser;
use file_processor::audio::audio_processor;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use text_loader::TextLoader;

fn get_bert_embeder(config: &BertConfig) -> PyResult<BertEmbeder> {
    let model_id = &config
        .model_id
        .clone()
        .unwrap_or_else(|| "sentence-transformers/all-MiniLM-L12-v2".to_string());
    let revision = &config.revision;
    let embeder = if let Some(revision) = revision {
        BertEmbeder::new(model_id.to_string(), Some(revision.to_string()))
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    } else {
        BertEmbeder::new(model_id.to_string(), None)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    };
    Ok(embeder)
}

fn get_clip_embeder(config: &ClipConfig) -> PyResult<ClipEmbeder> {
    let model_id = &config
        .model_id
        .clone()
        .unwrap_or_else(|| "openai/clip-vit-base-patch32".to_string());
    let revision = &config.revision;

    let embeder = if let Some(revision) = revision {
        ClipEmbeder::new(model_id.to_string(), Some(revision.to_string()))
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    } else {
        ClipEmbeder::new(model_id.to_string(), None)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    };
    Ok(embeder)
}

fn get_cloud_embeder(config: &CloudConfig) -> PyResult<CloudEmbeder> {
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
                return Err(PyValueError::new_err(
                    "Invalid provider. Choose between OpenAI and Cohere.",
                ))
            }
        },
        None => {
            return Err(PyValueError::new_err(
                "Provide the provider for the cloud embedding model.",
            ))
        }
    };

    Ok(embeder)
}

fn get_jina_embeder(config: &JinaConfig) -> PyResult<JinaEmbeder> {
    let model_id = &config
        .model_id
        .clone()
        .unwrap_or_else(|| "jinaai/jina-embeddings-v2-base-en".to_string());
    let revision = &config.revision;
    let embeder = if let Some(revision) = revision {
        JinaEmbeder::new(model_id.to_string(), Some(revision.to_string()))
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    } else {
        JinaEmbeder::new(model_id.to_string(), None)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    };
    Ok(embeder)
}

fn embed_default(file_name: &str, embeder: &str) -> PyResult<Option<Vec<EmbedData>>> {
    let decoder_config = config::AudioDecoderConfig::new(
        Some("lmz/candle-whisper".to_string()),
        Some("main".to_string()),
        Some("tiny-en".to_string()),
        Some(false),
    );
    let config = EmbedConfig {
        audio_decoder: Some(decoder_config),
        bert: Some(BertConfig {
            model_id: Some("sentence-transformers/all-MiniLM-L12-v2".to_string()),
            revision: None,
            chunk_size: Some(100),
            batch_size: Some(32),
        }),
        ..Default::default()
    };
    match embeder {
        "Cloud"|"OpenAI" => emb_text(file_name, &Embeder::Cloud(CloudEmbeder::OpenAI(embedding_model::openai::OpenAIEmbeder::default())), None, None, None),
        "Jina" => emb_text(file_name, &Embeder::Jina(embedding_model::jina::JinaEmbeder::default()), None, None, None),
        "Bert" => emb_text(file_name, &Embeder::Bert(embedding_model::bert::BertEmbeder::default()), None, None,None),
        "Clip" => Ok(Some(vec![emb_image(file_name, embedding_model::clip::ClipEmbeder::default())?])),
        "Audio" => emb_audio(file_name, &config),
        _ => Err(PyValueError::new_err(
            "Invalid embedding model. Choose between OpenAI, Bert, Jina for text files and Clip for image files.",
        )),
    }
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
/// let openai_config = OpenAIConfig{ model: Some("text-embedding-3-small".to_string()), api_key: None, chunk_size: Some(100) };
/// let config = EmbedConfig{ openai: Some(openai_config), ..Default::default() };
/// let embeddings = embed_query(query, embeder).unwrap();
/// println!("{:?}", embeddings);
/// ```
/// This will output the embeddings of the queries using the OpenAI embedding model.
#[pyfunction]
#[pyo3(signature = (query, embeder, config=None))]
pub fn embed_query(
    query: Vec<String>,
    embeder: &str,
    config: Option<&EmbedConfig>,
) -> PyResult<Vec<EmbedData>> {
    let encodings = if let Some(config) = config {
        if let Some(bert_config) = &config.bert {
            let embeder = get_bert_embeder(bert_config)?;
            embeder.embed(&query, bert_config.batch_size).unwrap()
        } else if let Some(clip_config) = &config.clip {
            let embeder = get_clip_embeder(clip_config)?;
            embeder.embed(&query, clip_config.batch_size).unwrap()
        } else if let Some(cloud_config) = &config.cloud {
            let embeder = get_cloud_embeder(cloud_config)?;
            embeder.embed(&query).unwrap()
        } else if let Some(jina_config) = &config.jina {
            let embeder = get_jina_embeder(jina_config)?;
            embeder.embed(&query, jina_config.batch_size).unwrap()
        } else {
            // error to provide atleat one config

            return Err(PyValueError::new_err(
                "Provide the config for the embedding model. Otherwise, use the embed_query function without the config parameter.",
            ));
        }
    } else {
        match embeder {
            "Cloud" | "OpenAI" => Embeder::Cloud(CloudEmbeder::OpenAI(
                embedding_model::openai::OpenAIEmbeder::default(),
            )),
            "Jina" => Embeder::Jina(embedding_model::jina::JinaEmbeder::default()),
            "Clip" => Embeder::Clip(embedding_model::clip::ClipEmbeder::default()),
            "Bert" => Embeder::Bert(embedding_model::bert::BertEmbeder::default()),
            _ => {
                return Err(PyValueError::new_err(
                    "Invalid embedding model. Choose between OpenAI, Jina, Bert and Clip.",
                ))
            }
        }
        .embed(&query, None)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
    };

    let embeddings = get_text_metadata(&encodings, &query, None)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

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
/// let bert_config = BertConfig{ model_id: Some("sentence-transformers/all-MiniLM-L12-v2".to_string()), revision: None, chunk_size: Some(100) };
/// let embeddings = embed_file(file_name, embeder, config).unwrap();
/// ```
/// This will output the embeddings of the file using the OpenAI embedding model.
#[pyfunction]
#[pyo3(signature = (file_name, embeder, config=None, adapter=None))]
pub fn embed_file(
    file_name: &str,
    embeder: &str,
    config: Option<&EmbedConfig>,
    adapter: Option<PyObject>,
) -> PyResult<Option<Vec<EmbedData>>> {
    let embeddings = if let Some(config) = config {
        if config.audio_decoder.is_some() {
            emb_audio(file_name, config)?
        } else if let Some(bert_config) = &config.bert {
            let embeder = get_bert_embeder(bert_config)?;
            let chunk_size = bert_config.chunk_size.unwrap_or(100);
            emb_text(
                file_name,
                &Embeder::Bert(embeder),
                Some(chunk_size),
                bert_config.batch_size,
                adapter,
            )?
        } else if let Some(clip_config) = &config.clip {
            let embeder = get_clip_embeder(clip_config)?;
            Some(vec![emb_image(file_name, embeder)?])
        } else if let Some(openai_config) = &config.cloud {
            let embeder = get_cloud_embeder(openai_config)?;
            let chunk_size = openai_config.chunk_size.unwrap_or(100);

            emb_text(
                file_name,
                &Embeder::Cloud(embeder),
                Some(chunk_size),
                None,
                adapter,
            )?
        } else if let Some(jina_config) = &config.jina {
            let embeder = get_jina_embeder(jina_config)?;
            let chunk_size = jina_config.chunk_size.unwrap_or(100);
            emb_text(
                file_name,
                &Embeder::Jina(embeder),
                Some(chunk_size),
                jina_config.batch_size,
                adapter,
            )?
        } else {
            return Err(PyValueError::new_err(
                "Provide the config for the embedding model. Otherwise, use the embed_file function without the config parameter.",
            ));
        }
    } else {
        embed_default(file_name, embeder)?
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
/// let bert_config = BertConfig{ model_id: Some("sentence-transformers/all-MiniLM-L12-v2".to_string()), revision: None, chunk_size: Some(100) };
/// let config = EmbedConfig{ bert: Some(bert_config), ..Default::default() };
/// let extensions = Some(vec!["txt".to_string(), "pdf".to_string()]);
/// let embeddings = embed_directory(directory, embeder, extensions, config).unwrap();
/// ```
/// This will output the embeddings of the files in the specified directory using the OpenAI embedding model.
#[pyfunction]
#[pyo3(signature = (directory, embeder, extensions=None, config=None, adapter = None))]
pub fn embed_directory(
    directory: PathBuf,
    embeder: &str,
    extensions: Option<Vec<String>>,
    config: Option<&EmbedConfig>,
    adapter: Option<PyObject>,
) -> PyResult<Option<Vec<EmbedData>>> {
    let embeddings = if let Some(config) = &config {
        if let Some(bert_config) = &config.bert {
            let embeder = get_bert_embeder(bert_config)?;
            let chunk_size = bert_config.chunk_size.unwrap_or(100);
            Ok(emb_directory(
                directory,
                &Embeder::Bert(embeder),
                extensions,
                Some(chunk_size),
                bert_config.batch_size,
                adapter,
            )?)
        } else if let Some(clip_config) = &config.clip {
            let embeder = get_clip_embeder(clip_config)?;
            Ok(emb_image_directory(directory, embeder)?)
        } else if let Some(_openai_config) = &config.cloud {
            let embeder = get_cloud_embeder(_openai_config)?;
            let chunk_size = _openai_config.chunk_size.unwrap_or(100);
            Ok(emb_directory(
                directory,
                &Embeder::Cloud(embeder),
                extensions,
                Some(chunk_size),
                None,
                adapter,
            )?)
        } else if let Some(jina_config) = &config.jina {
            let embeder = get_jina_embeder(jina_config)?;
            let chunk_size = jina_config.chunk_size.unwrap_or(100);
            Ok(emb_directory(
                directory,
                &Embeder::Jina(embeder),
                extensions,
                Some(chunk_size),
                jina_config.batch_size,
                adapter,
            )?)
        } else {
            return Err(PyValueError::new_err(
                "Provide the config for the embedding model. Otherwise, use the embed_directory function without the config parameter.",
            ));
        }
    } else {
        match embeder {
            "Cloud"|"OpenAI" => emb_directory(
                directory,
                &Embeder::Cloud(CloudEmbeder::OpenAI(embedding_model::openai::OpenAIEmbeder::default())),
                extensions,
                None,
                Some(100),
                adapter
            )        ,
                    "Jina" => Ok(emb_directory(directory, &Embeder::Jina(embedding_model::jina::JinaEmbeder::default()), extensions, None,None, adapter)?),
                "Bert" => Ok(emb_directory(directory, &Embeder::Bert(embedding_model::bert::BertEmbeder::default()), extensions, None,None, adapter)?),
                "Clip" => Ok(emb_image_directory(directory, embedding_model::clip::ClipEmbeder::default())?),
                _ => {
                    return Err(PyValueError::new_err(
                        "Invalid embedding model. Choose between OpenAI and Bert for text files and Clip for image files.",
                    ))
                }
    }
    };

    embeddings

    // Send embeddings to vector database
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
#[pyfunction]
#[pyo3(signature = (url, embeder, config=None, adapter = None))]
pub fn embed_webpage(
    url: String,
    embeder: &str,
    config: Option<&EmbedConfig>,
    adapter: Option<PyObject>,
) -> PyResult<Option<Vec<EmbedData>>> {
    let website_processor = file_processor::website_processor::WebsiteProcessor::new();
    let webpage = website_processor
        .process_website(url.as_ref())
        .map_err(|e| PyValueError::new_err(format!("Error processing website: {}", e)))?;

    let embeddings = if let Some(config) = config {
        if let Some(bert_config) = &config.bert {
            let embeder = get_bert_embeder(bert_config)?;
            webpage
                .embed_webpage(
                    &embeder,
                    bert_config.chunk_size.unwrap_or(100),
                    bert_config.batch_size,
                )
                .map_err(|e| PyValueError::new_err(format!("Error embedding webpage: {}", e)))?
        } else if let Some(cloud_config) = &config.cloud {
            let embeder = get_cloud_embeder(cloud_config)?;
            webpage
                .embed_webpage(&embeder,cloud_config.chunk_size.unwrap_or(100), None)
                .map_err(|e| PyValueError::new_err(format!("Error embedding webpage: {}", e)))?
        } else if let Some(jina_config) = &config.jina {
            let embeder = get_jina_embeder(jina_config)?;
            webpage
                .embed_webpage(
                    &embeder,
                    jina_config.chunk_size.unwrap_or(100),
                    jina_config.batch_size,
                )
                .map_err(|e| PyValueError::new_err(format!("Error embedding webpage: {}", e)))?
        } else {
            return Err(PyValueError::new_err(
                "Provide the config for the embedding model. Otherwise, use the embed_webpage function without the config parameter.",
            ));
        }
    } else {
        match embeder {
            "OpenAI" => webpage
                .embed_webpage(
                    &embedding_model::openai::OpenAIEmbeder::default(),
                    100,
                    None,
                )
                .unwrap(),
            "Jina" => webpage
                .embed_webpage(&embedding_model::jina::JinaEmbeder::default(), 100, None)
                .unwrap(),
            "Bert" => webpage
                .embed_webpage(&embedding_model::bert::BertEmbeder::default(), 100, None)
                .unwrap(),
            _ => {
                return Err(PyValueError::new_err(
                    "Invalid embedding model. Choose between OpenAI, Jina, Bert and Clip.",
                ))
            }
        }
    };

    // Send embeddings to vector database
    if let Some(adapter) = adapter {
        Python::with_gil(|py| {
            let conversion_fn = adapter.getattr(py, "convert")?;
            let upsert_fn = adapter.getattr(py, "upsert")?;
            let converted_embeddings = conversion_fn.call1(py, (embeddings.clone(),))?;
            upsert_fn.call1(py, (&converted_embeddings,))?;

            // return none
            Ok(None)
        })
    } else {
        Ok(Some(embeddings))
    }
}

#[pymodule]
fn embed_anything(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed_file, m)?)?;
    m.add_function(wrap_pyfunction!(embed_directory, m)?)?;
    m.add_function(wrap_pyfunction!(embed_query, m)?)?;
    m.add_function(wrap_pyfunction!(embed_webpage, m)?)?;
    m.add_class::<embedding_model::embed::EmbedData>()?;
    m.add_class::<BertConfig>()?;
    m.add_class::<ClipConfig>()?;
    m.add_class::<CloudConfig>()?;
    m.add_class::<EmbedConfig>()?;
    m.add_class::<AudioDecoderConfig>()?;
    m.add_class::<JinaConfig>()?;

    Ok(())
}

fn emb_directory(
    directory: PathBuf,
    embedding_model: &Embeder,
    extensions: Option<Vec<String>>,
    chunk_size: Option<usize>,
    batch_size: Option<usize>,
    adapter: Option<PyObject>,
) -> PyResult<Option<Vec<EmbedData>>> {
    let mut file_parser = FileParser::new();
    file_parser.get_text_files(&directory, extensions).unwrap();

    if let Some(adapter) = adapter {
        Python::with_gil(|py| {
            file_parser
                .files
                .iter()
                .filter_map(|file| {
                    emb_text(
                        file,
                        &embedding_model,
                        chunk_size,
                        batch_size,
                        Some(adapter.clone_ref(py)),
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        });
        Ok(None)
    } else {
        let embeddings = file_parser
            .files
            .par_iter()
            .filter_map(|file| {
                emb_text(file, &embedding_model, chunk_size, batch_size, None).unwrap()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        Ok(Some(embeddings))
    }
}

fn emb_text<T: AsRef<std::path::Path>>(
    file: T,
    embedding_model: &Embeder,
    chunk_size: Option<usize>,
    batch_size: Option<usize>,
    adapter: Option<PyObject>,
) -> PyResult<Option<Vec<EmbedData>>> {
    let text = TextLoader::extract_text(file.as_ref().to_str().unwrap()).unwrap();
    let chunks = TextLoader::split_into_chunks(&text, chunk_size.unwrap_or(100));
    let metadata = TextLoader::get_metadata(file.as_ref().to_str().unwrap()).ok();

    if let Some(adapter) = adapter {
        Python::with_gil(|py| {
            let embeddings = chunks
                .map(|chunks| {
                    let encodings = embedding_model.embed(&chunks, batch_size).unwrap();
                    get_text_metadata(&encodings, &chunks, metadata).unwrap()
                })
                .ok_or_else(|| PyValueError::new_err("No text found in file"))?;

            let conversion_fn = adapter.getattr(py, "convert")?;
            let upsert_fn = adapter.getattr(py, "upsert")?;
            let converted_embeddings = conversion_fn.call1(py, (embeddings,))?;
            upsert_fn.call1(py, (&converted_embeddings,))?;

            // return none
            Ok(None)
        })
    } else {
        let embeddings = chunks
            .map(|chunks| {
                let encodings = embedding_model.embed(&chunks, batch_size).unwrap();
                get_text_metadata(&encodings, &chunks, metadata).unwrap()
            })
            .ok_or_else(|| PyValueError::new_err("No text found in file"))?;

        Ok(Some(embeddings))
    }
}

fn emb_image<T: AsRef<std::path::Path>, U: EmbedImage>(
    image_path: T,
    embedding_model: U,
) -> PyResult<EmbedData> {
    let embedding = embedding_model.embed_image(image_path, None).unwrap();
    Ok(embedding)
}

pub fn emb_audio<T: AsRef<std::path::Path>>(
    audio_file: T,
    config: &EmbedConfig,
) -> PyResult<Option<Vec<EmbedData>>> {
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
        return Err(PyValueError::new_err(
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
        return Err(PyValueError::new_err(
            "Provide the config for the text embedding model. Otherwise, use the embed_audio function without the config parameter.",
        ));
    };

    Ok(Some(embeddings))
}

fn emb_image_directory<T: EmbedImage>(
    directory: PathBuf,
    embedding_model: T,
) -> PyResult<Option<Vec<EmbedData>>> {
    let mut file_parser = FileParser::new();
    file_parser.get_image_paths(&directory).unwrap();

    let embeddings = embedding_model
        .embed_image_batch(&file_parser.files)
        .unwrap();
    Ok(Some(embeddings))
}
