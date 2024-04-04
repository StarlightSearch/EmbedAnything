pub mod embed;
pub mod file_embed;
pub mod pdf_processor;

use embed::EmbedData;
use file_embed::FileEmbeder;
use pyo3::prelude::*;

/// Embeds the text from a file using the OpenAI API.
#[pyfunction]
#[pyo3(signature=(file_name="hello"))]
fn embed_file(file_name:&str) -> PyResult<(Vec<EmbedData>)>{
    let embeder = embed::Embeder::OpenAI(embed::OpenAIEmbeder::default());
    let mut file_embeder = FileEmbeder::new(file_name.to_string());
    let text = file_embeder.extract_text().unwrap();
    file_embeder.split_into_chunks(&text, 100);
    tokio::runtime::Runtime::new().unwrap().block_on(file_embeder.embed(&embeder)).unwrap();
    Ok(file_embeder.embeddings)
}

/// A Python module implemented in Rust.
#[pymodule]
fn embed_anything(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed_file, m)?)?;
    m.add_class::<embed::EmbedData>()?;
    Ok(())
}
