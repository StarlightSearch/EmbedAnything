use std::{collections::HashMap, fmt::Debug, fs};

use anyhow::Error;
use chrono::{DateTime, Local};
use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::Tokenizer;

use crate::file_processor::markdown_processor::MarkdownProcessor;

use super::file_processor::pdf_processor::PdfProcessor;
use std::path::PathBuf;

impl Default for TextLoader {
    fn default() -> Self {
        Self::new(256)
    }
}

#[derive(Debug)]
pub struct TextLoader {
    splitter: TextSplitter<Tokenizer>,
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
        match file.split('.').last().unwrap() {
            "pdf" => PdfProcessor::extract_text(&PathBuf::from(file)),
            "md" => MarkdownProcessor::extract_text(&PathBuf::from(file)),
            _ => Err(Error::msg("Unsupported file type")),
        }
    }

    pub fn get_metadata(file: &str) -> Result<HashMap<String, String>, Error> {
        let metadata = fs::metadata(file)?;
        let mut metadata_map = HashMap::new();
        metadata_map.insert(
            "created".to_string(),
            format!("{}", DateTime::<Local>::from(metadata.created()?)),
        );
        metadata_map.insert(
            "modified".to_string(),
            format!("{}", DateTime::<Local>::from(metadata.modified()?)),
        );
        metadata_map.insert("file_name".to_string(), file.to_string());
        Ok(metadata_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding_model::{clip::ClipEmbeder, embed::EmbedImage};
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
        let emb_data = embeder.embed_image(&file_path, None).unwrap();
        assert_eq!(emb_data.embedding.len(), 512);
    }
}
