use std::fmt::Debug;
use std::io::Error;

use crate::embedding_model::embed::{EmbedData, Embeder};

use super::pdf_processor::PdfProcessor;
use std::path::PathBuf;

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
        PdfProcessor::extract_text(&PathBuf::from(&self.file))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding_model::{bert::BertEmbeder, clip::ClipEmbeder, embed::{EmbedImage, Embeder}};
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_file_embeder() {
        let file_path = PathBuf::from("test_files/test.pdf");
        let text = PdfProcessor::extract_text(&file_path).unwrap();
        let embeder = Embeder::Bert(BertEmbeder::default());
        let mut file_embeder = FileEmbeder::new(file_path.to_string_lossy().to_string());
        file_embeder.split_into_chunks(&text, 100);
        file_embeder.embed(&embeder).await.unwrap();
        assert_eq!(file_embeder.chunks.len(), 5);
        assert_eq!(file_embeder.embeddings.len(), 5);
    }

    #[tokio::test]
    async fn test_image_embeder() {
        let file_path = PathBuf::from("test_files/clip/cat1.jpg");
        let embeder = ClipEmbeder::default();
        let emb_data = embeder.embed_image(&file_path).unwrap();
        assert_eq!(emb_data.embedding.len(), 512);}
}
