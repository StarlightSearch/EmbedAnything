pub mod embed;
pub mod file_embed;
pub mod pdf_processor;
pub mod parser;

use std::path::PathBuf;

use embed::{EmbedData, Embeder};
use file_embed::FileEmbeder;
use parser::FileParser;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

/// Embeds the text from a file using the OpenAI API.
#[pyfunction]
fn embed_file(file_name: &str, embeder: &str) -> PyResult<Vec<EmbedData>> {

    let embedding_model = match embeder {
        "OpenAI"=> Embeder::OpenAI(embed::OpenAIEmbeder::default()),
        "AllMiniLmL12V2" => Embeder::AllMiniLmL12V2(embed::AllMiniLmL12V2Embeder::new()),
        _ => return Err(PyValueError::new_err("Invalid embedding model. Choose between OpenAI and AllMiniLmL12V2.")),
        
    
    };

    let mut file_embeder = FileEmbeder::new(file_name.to_string());
    let text = file_embeder.extract_text().unwrap();
    file_embeder.split_into_chunks(&text, 100);
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(file_embeder.embed(&embedding_model))
        .unwrap();
    Ok(file_embeder.embeddings)
}

#[pyfunction]
fn embed_directory(directory: PathBuf, embeder: &str) -> PyResult<Vec<EmbedData>> {

    let embedding_model = match embeder {
        "OpenAI" => Embeder::OpenAI(embed::OpenAIEmbeder::default()),
        "AllMiniLmL12V2" => Embeder::AllMiniLmL12V2(embed::AllMiniLmL12V2Embeder::new()),
        _ => return Err(PyValueError::new_err("Invalid embedding model. Choose between OpenAI and AllMiniLmL12V2.")),
    };

    let mut file_parser = FileParser::new();

    let embeddings: Vec<EmbedData> = file_parser
        .files
        .par_iter()
        .map(|file| {
            let mut file_embeder = FileEmbeder::new(file.to_string());
            let text = file_embeder.extract_text().unwrap();
            file_embeder.split_into_chunks(&text, 100);
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(file_embeder.embed(&embedding_model))
                .unwrap();
            file_embeder.embeddings
        })
        .flatten()
        .collect();
    
    Ok(embeddings)
    
}

/// A Python module implemented in Rust.
#[pymodule]
fn embed_anything(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed_file, m)?)?;
    m.add_function(wrap_pyfunction!(embed_directory, m)?)?;
    m.add_class::<embed::EmbedData>()?;
    Ok(())
}
