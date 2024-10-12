use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    fs,
    sync::Arc,
};

use crate::{embeddings::embed::Embedder, file_processor::{markdown_processor::MarkdownProcessor, txt_processor::TxtProcessor}};
use crate::{
    chunkers::statistical::StatisticalChunker,
    embeddings::{embed::TextEmbedder, local::jina::JinaEmbedder},
};
use anyhow::Error;
use chrono::{DateTime, Local};
use text_splitter::{Characters, ChunkConfig, TextSplitter};

use super::file_processor::pdf_processor::PdfProcessor;
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub enum SplittingStrategy {
    Sentence,
    Semantic,
}

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
    pub splitter: TextSplitter<Characters>,
}
impl TextLoader {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            // splitter: TextSplitter::new(ChunkConfig::new(chunk_size).with_sizer(
            //     Tokenizer::from_pretrained("BEE-spoke-data/cl100k_base-mlm", None).unwrap(),
            // )),
            splitter: TextSplitter::new(ChunkConfig::new(chunk_size)),
        }
    }
    pub fn split_into_chunks(
        &self,
        text: &str,
        splitting_strategy: SplittingStrategy,
        semantic_encoder: Option<Arc<Embedder>>,
    ) -> Option<Vec<String>> {
        if text.is_empty() {
            return None;
        }
        let chunks: Vec<String> = match splitting_strategy {
            SplittingStrategy::Sentence => self
                .splitter
                .chunks(text)
                .par_bridge()
                .map(|chunk| chunk.to_string())
                .collect(),
            SplittingStrategy::Semantic => {
                let embeder =
                    semantic_encoder.unwrap_or(Arc::new(Embedder::Text(TextEmbedder::Jina(JinaEmbedder::default()))));
                let chunker = StatisticalChunker {
                    encoder: embeder,
                    ..Default::default()
                };

                tokio::task::block_in_place(|| {
                    tokio::runtime::Runtime::new()
                        .unwrap()
                        .block_on(async { chunker.chunk(text, 64).await })
                })
            }
        };

        Some(chunks)
    }

    pub fn extract_text<T: AsRef<std::path::Path>>(file: &T) -> Result<String, Error> {
        if !file.as_ref().exists() {
            return Err(FileLoadingError::FileNotFound(
                file.as_ref().to_str().unwrap().to_string(),
            )
            .into());
        }
        let file_extension = file.as_ref().extension().unwrap();
        match file_extension.to_str().unwrap() {
            "pdf" => PdfProcessor::extract_text(file),
            "md" => MarkdownProcessor::extract_text(file),
            "txt" => TxtProcessor::extract_text(file),
            _ => Err(FileLoadingError::UnsupportedFileType(
                file.as_ref()
                    .extension()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string(),
            )
            .into()),
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
    fn test_image_embeder() {
        let file_path = PathBuf::from("test_files/clip/cat1.jpg");
        let mut embeder = ClipEmbedder::default();
        let emb_data = embeder.embed_image(file_path, None).unwrap();
        assert_eq!(emb_data.embedding.to_dense().unwrap().len(), 512);
    }
}
