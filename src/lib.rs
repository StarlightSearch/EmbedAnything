//! # Embed Anything
//! This library provides a simple interface to embed text and images using various embedding models.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

pub mod embedding_model;
pub mod file_embed;
pub mod file_processor;
pub mod parser;

use std::path::PathBuf;

use embedding_model::embed::{EmbedData, EmbedImage, Embeder};
use file_embed::FileEmbeder;
use parser::FileParser;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use tokio::runtime::Builder;

/// Embeds a list of queries using the specified embedding model.
///
/// # Arguments
///
/// * `query` - A vector of strings representing the queries to embed.
/// * `embeder` - A string specifying the embedding model to use. Valid options are "OpenAI", "Jina", "Clip", and "Bert".
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
/// let embeddings = embed_query(query, embeder).unwrap();
/// println!("{:?}", embeddings);
/// ```
/// This will output the embeddings of the queries using the OpenAI embedding model.
#[pyfunction]
pub fn embed_query(query: Vec<String>, embeder: &str) -> PyResult<Vec<EmbedData>> {
    let embedding_model = match embeder {
        "OpenAI" => Embeder::OpenAI(embedding_model::openai::OpenAIEmbeder::default()),
        "Jina" => Embeder::Jina(embedding_model::jina::JinaEmbeder::default()),
        "Clip" => Embeder::Clip(embedding_model::clip::ClipEmbeder::default()),
        "Bert" => Embeder::Bert(embedding_model::bert::BertEmbeder::default()),
        _ => {
            return Err(PyValueError::new_err(
                "Invalid embedding model. Choose between OpenAI and AllMiniLmL12V2.",
            ))
        }
    };
    let runtime = Builder::new_multi_thread().enable_all().build().unwrap();

    let embeddings = runtime.block_on(embedding_model.embed(&query)).unwrap();
    Ok(embeddings)
}
/// Embeds the text from a file using the specified embedding model.
///
/// # Arguments
///
/// * `file_name` - A string specifying the name of the file to embed.
/// * `embeder` - A string specifying the embedding model to use. Valid options are "OpenAI", "Jina", "Clip", and "Bert".
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
/// let file_name = "example.pdf";
/// let embeder = "Bert";
/// let embeddings = embed_file(file_name, embeder).unwrap();
/// ```
/// This will output the embeddings of the file using the OpenAI embedding model.
#[pyfunction]
pub fn embed_file(file_name: &str, embeder: &str) -> PyResult<Vec<EmbedData>> {
    let embedding_model = match embeder {
        "OpenAI" => Embeder::OpenAI(embedding_model::openai::OpenAIEmbeder::default()),
        "Jina" => Embeder::Jina(embedding_model::jina::JinaEmbeder::default()),
        "Clip" => Embeder::Clip(embedding_model::clip::ClipEmbeder::default()),
        "Bert" => Embeder::Bert(embedding_model::bert::BertEmbeder::default()),
        _ => {
            return Err(PyValueError::new_err(
                "Invalid embedding model. Choose between OpenAI and AllMiniLmL12V2.",
            ))
        }
    };

    let mut file_embeder = FileEmbeder::new(file_name.to_string());
    let text = file_embeder.extract_text().unwrap();
    file_embeder.split_into_chunks(&text, 100);
    let runtime = Builder::new_multi_thread().enable_all().build().unwrap();
    runtime
        .block_on(file_embeder.embed(&embedding_model))
        .unwrap();
    Ok(file_embeder.embeddings)
}

/// Embeds the text from files in a directory using the specified embedding model.
///
/// # Arguments
///
/// * `directory` - A `PathBuf` representing the directory containing the files to embed.
/// * `embeder` - A string specifying the embedding model to use. Valid options are "OpenAI", "Jina", "Clip", and "Bert".
/// * `extensions` - An optional vector of strings representing the file extensions to consider for embedding. If `None`, all files in the directory will be considered.
///
/// # Returns
///
/// A vector of `EmbedData` objects representing the embeddings of the files.
///
/// # Errors
///
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
/// let extensions = Some(vec!["txt".to_string(), "pdf".to_string()]);
/// let embeddings = embed_directory(directory, embeder, extensions).unwrap();
/// ```
/// This will output the embeddings of the files in the specified directory using the OpenAI embedding model.
#[pyfunction]
pub fn embed_directory(
    directory: PathBuf,
    embeder: &str,
    extensions: Option<Vec<String>>,
) -> PyResult<Vec<EmbedData>> {
    let embeddings = match embeder {
        "OpenAI" => emb(
            directory,
            Embeder::OpenAI(embedding_model::openai::OpenAIEmbeder::default()),
            extensions,
        )
        .unwrap(),
        "Jina" => emb(
            directory,
            Embeder::Jina(embedding_model::jina::JinaEmbeder::default()),
            extensions,
        )
        .unwrap(),
        "Bert" => emb(
            directory,
            Embeder::Bert(embedding_model::bert::BertEmbeder::default()),
            extensions,
        )
        .unwrap(),
        "Clip" => emb_image(directory, embedding_model::clip::ClipEmbeder::default()).unwrap(),

        _ => {
            return Err(PyValueError::new_err(
                "Invalid embedding model. Choose between OpenAI and Bert for text files and Clip for image files.",
            ))
        }
    };

    Ok(embeddings)
}

#[pymodule]
fn embed_anything(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed_file, m)?)?;
    m.add_function(wrap_pyfunction!(embed_directory, m)?)?;
    m.add_function(wrap_pyfunction!(embed_query, m)?)?;
    m.add_class::<embedding_model::embed::EmbedData>()?;
    Ok(())
}

fn emb(
    directory: PathBuf,
    embedding_model: Embeder,
    extensions: Option<Vec<String>>,
) -> PyResult<Vec<EmbedData>> {
    let mut file_parser = FileParser::new();
    file_parser.get_text_files(&directory, extensions).unwrap();

    let embeddings: Vec<EmbedData> = file_parser
        .files
        .par_iter()
        .map(|file| {
            let mut file_embeder = FileEmbeder::new(file.to_string());
            let text = file_embeder.extract_text().unwrap();
            file_embeder.split_into_chunks(&text, 100);
            let runtime = Builder::new_multi_thread().enable_all().build().unwrap();
            runtime
                .block_on(file_embeder.embed(&embedding_model))
                .unwrap();
            file_embeder.embeddings
        })
        .flatten()
        .collect();

    Ok(embeddings)
}

fn emb_image<T: EmbedImage>(directory: PathBuf, embedding_model: T) -> PyResult<Vec<EmbedData>> {
    let mut file_parser = FileParser::new();
    file_parser.get_image_paths(&directory).unwrap();

    let embeddings = embedding_model
        .embed_image_batch(&file_parser.files)
        .unwrap();
    Ok(embeddings)
}
