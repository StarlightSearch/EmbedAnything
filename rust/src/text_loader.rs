use std::{collections::HashMap, fmt::Debug, fs};
use rayon::prelude::*;
use anyhow::Error;
use chrono::{DateTime, Local};
use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::Tokenizer;

use crate::file_processor::{markdown_processor::MarkdownProcessor, txt_processor::TxtProcessor};

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
        Some(self
            .splitter
            .chunks(text)
            .par_bridge() // Convert to parallel iterator
            .map(|chunk| {
                let mut result = String::with_capacity(chunk.len());
                let mut chars = chunk.chars().peekable();
                let mut last_char = ' ';

                // Remove consecutive newlines and replace them with a single newline
                while let Some(c) = chars.next() {
                    match c {
                        '\n' if last_char == '\n' => {
                            result.push('\n');
                            result.push('\n');
                            // Skip any additional consecutive newlines
                            while chars.peek() == Some(&'\n') {
                                chars.next();
                            }
                        }
                        '\n' => {
                            if !result.ends_with(' ') {
                                result.push(' ');
                            }
                        }
                        ' ' if !result.ends_with(' ') => result.push(' '),
                        ' ' => {} // Skip consecutive spaces
                        _ => result.push(c),
                    }
                    last_char = c;
                }

                result.trim().to_string()
            })
            .collect())
    }

    pub fn extract_text(file: &str) -> Result<String, Error> {
        match file.split('.').last().unwrap() {
            "pdf" => PdfProcessor::extract_text(&PathBuf::from(file)),
            "md" => MarkdownProcessor::extract_text(&PathBuf::from(file)),
            "txt" => TxtProcessor::extract_text(&PathBuf::from(file)),
            _ => Err(Error::msg("Unsupported file type")),
        }
    }

    pub fn get_metadata<T:AsRef<std::path::Path>>(file: T) -> Result<HashMap<String, String>, Error> {
        let metadata = fs::metadata(&file)?;
        let mut metadata_map = HashMap::new();
        metadata_map.insert(
            "created".to_string(),
            format!("{}", DateTime::<Local>::from(metadata.created()?)),
        );
        metadata_map.insert(
            "modified".to_string(),
            format!("{}", DateTime::<Local>::from(metadata.modified()?)),
        );
        
        metadata_map.insert("file_name".to_string(), fs::canonicalize(file)?.to_str().unwrap().to_string());
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
