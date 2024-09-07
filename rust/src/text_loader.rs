use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    fs,
};

use crate::file_processor::{markdown_processor::MarkdownProcessor, txt_processor::TxtProcessor};
use anyhow::Error;
use chrono::{DateTime, Local};
use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::Tokenizer;

use super::file_processor::pdf_processor::PdfProcessor;
use std::path::PathBuf;

impl Default for TextLoader {
    fn default() -> Self {
        Self::new(256)
    }
}

#[derive(Debug)]
pub enum FileLoadingError {
    FileNotFound(String),
    UnsupportedFileType(String),
}
impl Display for FileLoadingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileLoadingError::FileNotFound(file) => write!(f, "File not found: {}", file),
            FileLoadingError::UnsupportedFileType(file) => {
                write!(f, "Unsupported file type: {}", file)
            }
        }
    }
}

impl From<FileLoadingError> for Error {
    fn from(error: FileLoadingError) -> Self {
        match error {
            FileLoadingError::FileNotFound(file) => {
                Error::msg(format!("File not found: {:?}", file))
            }
            FileLoadingError::UnsupportedFileType(file) => Error::msg(format!(
                "Unsupported file type: {:?}. Currently supported file types are: pdf, md, txt",
                file
            )),
        }
    }
}

#[derive(Debug)]
pub struct TextLoader {
    pub splitter: TextSplitter<Tokenizer>,
}
impl TextLoader {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            splitter: TextSplitter::new(
                ChunkConfig::new(chunk_size)
                    .with_sizer(Tokenizer::from_pretrained("bert-base-cased", None).unwrap()),
            ),
        }
    }
    pub fn split_into_chunks(&self, text: &str) -> Option<Vec<String>> {
        if text.is_empty() {
            return None;
        }
        let chunks: Vec<String> = self
            .splitter
            .chunks(text)
            .map(|chunk| chunk.to_string())
            .collect();
        Some(chunks)
    }

    pub fn extract_text(file: &str) -> Result<String, Error> {
        if !PathBuf::from(file).exists() {
            return Err(FileLoadingError::FileNotFound(file.to_string()).into());
        }
        let file_extension = file.split('.').last().unwrap();
        match file_extension {
            "pdf" => PdfProcessor::extract_text(&PathBuf::from(file)),
            "md" => MarkdownProcessor::extract_text(&PathBuf::from(file)),
            "txt" => TxtProcessor::extract_text(&PathBuf::from(file)),
            _ => Err(FileLoadingError::UnsupportedFileType(file.to_string()).into()),
        }
    }

    pub fn get_metadata<T: AsRef<std::path::Path>>(
        file: T,
    ) -> Result<HashMap<String, String>, Error> {
        let metadata = fs::metadata(&file).unwrap();
        let mut metadata_map = HashMap::new();
        metadata_map.insert(
            "created".to_string(),
            format!("{}", DateTime::<Local>::from(metadata.created()?)),
        );
        metadata_map.insert(
            "modified".to_string(),
            format!("{}", DateTime::<Local>::from(metadata.modified()?)),
        );

        metadata_map.insert(
            "file_name".to_string(),
            fs::canonicalize(file)?.to_str().unwrap().to_string(),
        );
        Ok(metadata_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::{embed::EmbedImage, local::clip::ClipEmbeder};
    use std::path::PathBuf;

    #[test]
    fn test_metadata() {
        let file_path = PathBuf::from("test_files/test.pdf");
        let metadata = TextLoader::get_metadata(file_path.to_str().unwrap()).unwrap();

        // assert the fields that are present
        assert!(metadata.contains_key("created"));
        assert!(metadata.contains_key("modified"));
        assert!(metadata.contains_key("file_name"));
    }

    #[test]
    fn test_image_embeder() {
        let file_path = PathBuf::from("test_files/clip/cat1.jpg");
        let embeder = ClipEmbeder::default();
        let emb_data = embeder.embed_image(file_path, None).unwrap();
        assert_eq!(emb_data.embedding.len(), 512);
    }
}
