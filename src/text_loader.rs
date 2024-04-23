use std::fmt::Debug;

use anyhow::Error;

use crate::file_processor::markdown_processor::MarkdownProcessor;

use super::file_processor::pdf_processor::PdfProcessor;
use std::path::PathBuf;

impl Default for TextLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct TextLoader;
impl TextLoader {
    pub fn new() -> Self {
        Self {}
    }
    pub fn split_into_chunks(text: &str, chunk_size: usize) -> Option<Vec<String>> {
        let mut chunk = Vec::new();
        let mut chunks = Vec::new();

        if text.is_empty() {
            return None;
        }
        if text.len() < chunk_size {
            chunks.push(text.to_owned());
            return Some(chunks);
        }

        let sentences: Vec<&str> = text.split_terminator('.').collect();

        for sentence in sentences {
            let sentence_with_period = format!("{}.", sentence);
            let words: Vec<String> = sentence_with_period
                .split_whitespace()
                .map(|word| word.to_owned())
                .collect();

            chunk.extend(words);

            if chunk.len() >= chunk_size {
                chunks.push(chunk.join(" "));
                chunk.clear();
            }
        }
        Some(chunks)
    }

    pub fn extract_text(file: &str) -> Result<String, Error> {
        match file.split('.').last().unwrap() {
            "pdf" => PdfProcessor::extract_text(&PathBuf::from(file)),
            "md" => MarkdownProcessor::extract_text(&PathBuf::from(file)),
            _ => Err(Error::msg("Unsupported file type")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding_model::{
        bert::BertEmbeder,
        clip::ClipEmbeder,
        embed::{EmbedImage, Embeder},
    };
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_file_embeder() {
        let file_path = PathBuf::from("test_files/test.pdf");
        let text = PdfProcessor::extract_text(&file_path).unwrap();
        let embeder = Embeder::Bert(BertEmbeder::default());
        let chunks = TextLoader::split_into_chunks(&text, 100);
        let embeddings = match chunks {
            Some(chunks) => {
                assert_eq!(chunks.len(), 5);

                embeder.embed(&chunks, None).unwrap()
            }
            None => panic!("No chunks found"),
        };
        assert_eq!(embeddings.len(), 5);
    }

    #[tokio::test]
    async fn test_image_embeder() {
        let file_path = PathBuf::from("test_files/clip/cat1.jpg");
        let embeder = ClipEmbeder::default();
        let emb_data = embeder.embed_image(&file_path, None).unwrap();
        assert_eq!(emb_data.embedding.len(), 512);
    }
}
