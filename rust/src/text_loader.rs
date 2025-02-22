use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    fs
    ,
};

use crate::chunkers::statistical::StatisticalChunker;
use anyhow::Error;
use chrono::{DateTime, Local};
use text_splitter::{Characters, ChunkConfig, TextSplitter};

use crate::config::SplittingStrategy;
use crate::file_processor::processor::Document;
use rayon::prelude::*;

impl Default for TextLoader {
    fn default() -> Self {
        Self::new(1000, 0.0)
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
                "Unsupported file type: {:?}. Currently supported file types are: pdf, md, txt, docx",
                file
            )),
        }
    }
}

#[derive(Debug)]
pub struct TextLoader {
    pub splitter: TextSplitter<Characters>,
}
impl TextLoader {
    pub fn new(chunk_size: usize, overlap_ratio: f32) -> Self {
        Self {
            splitter: TextSplitter::new(
                ChunkConfig::new(chunk_size)
                    .with_overlap(chunk_size * overlap_ratio as usize)
                    .unwrap()
            ),
        }
    }
    pub fn split_into_chunks(
        &self,
        text: &str,
        splitting_strategy: SplittingStrategy,
    ) -> Option<Vec<String>> {
        if text.is_empty() {
            return None;
        }

        // Remove single newlines but keep double newlines
        let cleaned_text = text
            .replace("\n\n", "{{DOUBLE_NEWLINE}}")
            .replace("\n", " ")
            .replace("{{DOUBLE_NEWLINE}}", "\n\n");
        let chunks: Vec<String> = match splitting_strategy {
            SplittingStrategy::Sentence => self
                .splitter
                .chunks(&cleaned_text)
                .par_bridge()
                .map(|chunk| chunk.to_string())
                .collect(),
            SplittingStrategy::Semantic { semantic_encoder } => {
                let chunker = StatisticalChunker {
                    encoder: semantic_encoder,
                    ..Default::default()
                };

                tokio::task::block_in_place(|| {
                    tokio::runtime::Runtime::new()
                        .unwrap()
                        .block_on(async { chunker.chunk(&cleaned_text, 64).await })
                })
            }
        };

        Some(chunks.iter().map(|s| s.to_string()).collect())
    }

    pub fn get_metadata<T: AsRef<std::path::Path>>(
        file: T,
    ) -> Result<HashMap<String, String>, Error> {
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
    use crate::embeddings::{embed::EmbedImage, local::clip::ClipEmbedder};
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
    fn test_image_embedder() {
        let file_path = PathBuf::from("test_files/clip/cat1.jpg");
        let embedder = ClipEmbedder::default();
        let emb_data = embedder.embed_image(file_path, None).unwrap();
        assert_eq!(emb_data.embedding.to_dense().unwrap().len(), 512);
    }
}
