use std::fmt::Debug;
use std::io::Error;

use crate::embedding_model::embed::{EmbedData, Embeder};

use super::pdf_processor::PdfProcessor;

#[derive(Debug)]
pub struct FileEmbeder {
    pub file: String,
    pub chunks: Vec<String>,
    pub embeddings: Vec<EmbedData>,
}

impl FileEmbeder {
    pub fn new(file: String) -> Self {
        Self {
            file,
            chunks: Vec::new(),
            embeddings: Vec::new(),
        }
    }
    pub fn split_into_chunks(&mut self, text: &str, chunk_size: usize) {
        let mut chunk = Vec::new();
        let sentences: Vec<&str> = text.split_terminator('.').collect();

        for sentence in sentences {
            let sentence_with_period = format!("{}.", sentence);
            let words: Vec<String> = sentence_with_period
                .split_whitespace()
                .map(|word| word.to_owned())
                .collect();

            chunk.extend(words);

            if chunk.len() >= chunk_size {
                self.chunks.push(chunk.join(" "));
                chunk.clear();
            }
        }
    }

    pub async fn embed(&mut self, embeder:&Embeder) -> Result<(), reqwest::Error> {
        self.embeddings = embeder.embed(&self.chunks).await?;
        Ok(())
    }

    pub fn extract_text(&self) -> Result<String, Error> {
        PdfProcessor::extract_text(&self.file)
    }
}
