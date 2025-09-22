//! Text loading and chunking utilities.
//!
//! Provides functionality for loading text content from files and splitting
//! it into manageable chunks for embedding generation.

use std::{collections::HashMap, fmt::Debug, fs};

use anyhow::Error;
use chrono::{DateTime, Local};
use text_splitter::{Characters, ChunkConfig, TextSplitter};

impl Default for TextLoader {
    fn default() -> Self {
        Self::new(1000, 0.0)
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
                    .unwrap(),
            ),
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
        let file_path = PathBuf::from("../test_files/test.pdf");
        let metadata = TextLoader::get_metadata(file_path.to_str().unwrap()).unwrap();

        // assert the fields that are present
        assert!(metadata.contains_key("created"));
        assert!(metadata.contains_key("modified"));
        assert!(metadata.contains_key("file_name"));
    }

    #[tokio::test]
    async fn test_image_embedder() {
        let file_path = PathBuf::from("../test_files/clip/cat1.jpg");
        let embedder = ClipEmbedder::default();
        let emb_data = embedder.embed_image(file_path, None).await.unwrap();
        assert_eq!(emb_data.embedding.to_dense().unwrap().len(), 512);
    }
}
