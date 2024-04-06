use std::path::PathBuf;

use embed_anything::{
    embedding_model::{
        self,
        embed::{EmbedData, EmbedImage, Embeder},
    },
    file_embed::FileEmbeder,
    parser::FileParser,
};
use pyo3::{exceptions::PyValueError, PyResult};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

fn main() {
    //    let out =  embed_file("test_files/TUe_SOP_AI_2.pdf", "Bert").unwrap();
    let out = embed_directory(PathBuf::from("test_files"), "Bert").unwrap();

    println!("{:?}", out);
}

fn embed_file(file_name: &str, embeder: &str) -> PyResult<Vec<EmbedData>> {
    let embedding_model = match embeder {
        "OpenAI" => Embeder::OpenAI(embedding_model::openai::OpenAIEmbeder::default()),
        "Jina" => Embeder::Jina(embedding_model::jina::JinaEmbeder::default().unwrap()),
        "Clip" => Embeder::Clip(embedding_model::clip::ClipEmbeder::default().unwrap()),
        "Bert" => Embeder::Bert(embedding_model::bert::BertEmbeder::default().unwrap()),
        _ => {
            return Err(PyValueError::new_err(
                "Invalid embedding model. Choose between OpenAI and AllMiniLmL12V2.",
            ))
        }
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

fn emb(directory: PathBuf, embedding_model: Embeder) -> PyResult<Vec<EmbedData>> {
    let mut file_parser = FileParser::new();
    file_parser.get_pdf_files(&directory).unwrap();

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

fn emb_image<T:EmbedImage>(directory: PathBuf, embedding_model: T) -> PyResult<Vec<EmbedData>> {
    let mut file_parser = FileParser::new();
    file_parser.get_image_paths(&directory).unwrap();

    println!("Getting files");
    println!("{:?}", file_parser.files);

    // let embeder = embedding_model::clip::ClipEmbeder::default().unwrap();

    let embeddings = embedding_model
        .embed_image_batch(&file_parser.files)
        .unwrap();
    Ok(embeddings)
}
fn embed_directory(directory: PathBuf, embeder: &str) -> PyResult<Vec<EmbedData>> {
    let embeddings = match embeder {
        "OpenAI" => emb(
            directory,
            Embeder::OpenAI(embedding_model::openai::OpenAIEmbeder::default()),
        ).unwrap(),
        "Jina" => emb(
            directory,
            Embeder::Jina(embedding_model::jina::JinaEmbeder::default().unwrap()),
        ).unwrap(),
        "Bert" => emb(
            directory,
            Embeder::Bert(embedding_model::bert::BertEmbeder::default().unwrap()),
        ).unwrap(),
        "Clip" => emb_image(
            directory,
            embedding_model::clip::ClipEmbeder::default().unwrap(),
        ).unwrap(),

        _ => {
            return Err(PyValueError::new_err(
                "Invalid embedding model. Choose between OpenAI and AllMiniLmL12V2.",
            ))
        }
    };

    Ok(embeddings)
}