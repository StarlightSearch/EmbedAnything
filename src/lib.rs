//! # Embed Anything
//! This library provides a simple interface to embed text and images using various embedding models.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

pub mod embedding_model;
pub mod file_processor;
pub mod file_loader;
pub mod text_loader;
use std::path::PathBuf;

use embedding_model::embed::{EmbedData, EmbedImage, Embeder};
use file_loader::FileParser;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use text_loader::TextLoader;
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
    let embeddings = embedding_model.embed(&query, None).unwrap();
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
/// let file_name = "test_files/test.pdf";
/// let embeder = "Bert";
/// let embeddings = embed_file(file_name, embeder).unwrap();
/// ```
/// This will output the embeddings of the file using the OpenAI embedding model.
#[pyfunction]
pub fn embed_file(file_name: &str, embeder: &str) -> PyResult<Vec<EmbedData>> {
    let embeddings = match embeder {
        "OpenAI" => emb_text(file_name, Embeder::OpenAI(embedding_model::openai::OpenAIEmbeder::default()))?,
        "Jina" => emb_text(file_name, Embeder::Jina(embedding_model::jina::JinaEmbeder::default()))?,
        "Bert" => emb_text(file_name, Embeder::Bert(embedding_model::bert::BertEmbeder::default()))?,
        "Clip" => emb_image(file_name, embedding_model::clip::ClipEmbeder::default())?,
        _ => {
            return Err(PyValueError::new_err(
                "Invalid embedding model. Choose between OpenAI and Bert for text files and Clip for image files.",
            ))
        }
    };

    Ok(vec![embeddings])
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
        "OpenAI" => emb_directory(
            directory,
            Embeder::OpenAI(embedding_model::openai::OpenAIEmbeder::default()),
            extensions,
        )
        .unwrap(),
        "Jina" => emb_directory(
            directory,
            Embeder::Jina(embedding_model::jina::JinaEmbeder::default()),
            extensions,
        )
        .unwrap(),
        "Bert" => emb_directory(
            directory,
            Embeder::Bert(embedding_model::bert::BertEmbeder::default()),
            extensions,
        )
        .unwrap(),
        "Clip" => emb_image_directory(directory, embedding_model::clip::ClipEmbeder::default())?,

        _ => {
            return Err(PyValueError::new_err(
                "Invalid embedding model. Choose between OpenAI and Bert for text files and Clip for image files.",
            ))
        }
    };

    Ok(embeddings)
}

#[pyfunction]
pub fn emb_webpage(url: String, embeder: &str) -> PyResult<Vec<EmbedData>> {
    let website_processor = file_processor::website_processor::WebsiteProcessor::new();
    let runtime = Builder::new_multi_thread().enable_all().build().unwrap();
    let webpage = runtime
        .block_on(website_processor.process_website(url.as_ref()))
        .unwrap();

    let embeddings = match embeder {
        "OpenAI" => webpage
            .embed_webpage(&embedding_model::openai::OpenAIEmbeder::default())
            .unwrap(),
        "Jina" => webpage
            .embed_webpage(&embedding_model::jina::JinaEmbeder::default())
            .unwrap(),
        "Bert" => webpage
            .embed_webpage(&embedding_model::bert::BertEmbeder::default())
            .unwrap(),
        _ => {
            return Err(PyValueError::new_err(
                "Invalid embedding model. Choose between OpenAI and AllMiniLmL12V2.",
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
    m.add_function(wrap_pyfunction!(emb_webpage, m)?)?;
    m.add_class::<embedding_model::embed::EmbedData>()?;

    Ok(())
}

fn emb_directory(
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
            let text = TextLoader::extract_text(file).unwrap();
            let chunks = TextLoader::split_into_chunks(&text, 100);
            let metadata = TextLoader::get_metadata(file).ok();
            chunks
                .map(|chunks| embedding_model.embed(&chunks, metadata).unwrap())
                .ok_or_else(|| PyValueError::new_err("No text found in file"))
                .unwrap()
        })
        .flatten()
        .collect();

    Ok(embeddings)
}

fn emb_text<T: AsRef<std::path::Path>>(file: T, embedding_model: Embeder) -> PyResult<EmbedData> {
    let text = TextLoader::extract_text(file.as_ref().to_str().unwrap()).unwrap();
    let chunks = TextLoader::split_into_chunks(&text, 100);
    let metadata = TextLoader::get_metadata(file.as_ref().to_str().unwrap()).ok();

    let embeddings = chunks
        .map(|chunks| embedding_model.embed(&chunks, metadata).unwrap())
        .ok_or_else(|| PyValueError::new_err("No text found in file"))?;

    Ok(embeddings[0].clone())
}

fn emb_image<T: AsRef<std::path::Path>, U: EmbedImage>(
    image_path: T,
    embedding_model: U,
) -> PyResult<EmbedData> {
    let embedding = embedding_model.embed_image(image_path, None).unwrap();
    Ok(embedding)
}

fn emb_image_directory<T: EmbedImage>(
    directory: PathBuf,
    embedding_model: T,
) -> PyResult<Vec<EmbedData>> {
    let mut file_parser = FileParser::new();
    file_parser.get_image_paths(&directory).unwrap();

    let embeddings = embedding_model
        .embed_image_batch(&file_parser.files)
        .unwrap();
    Ok(embeddings)
}
