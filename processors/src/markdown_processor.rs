use crate::processor::{Document, DocumentProcessor};
use text_splitter::{Characters, ChunkConfig, ChunkConfigError, MarkdownSplitter};

/// A struct that provides functionality to process Markdown files.
pub struct MarkdownProcessor {
    splitter: MarkdownSplitter<Characters>,
}

impl MarkdownProcessor {
    pub fn new(chunk_size: usize, overlap: usize) -> Result<MarkdownProcessor, ChunkConfigError> {
        let splitter_config = ChunkConfig::new(chunk_size).with_overlap(overlap)?;
        let splitter = MarkdownSplitter::new(splitter_config);
        Ok(MarkdownProcessor { splitter })
    }
}

impl DocumentProcessor for MarkdownProcessor {
    fn process_document(&self, content: &str) -> anyhow::Result<Document> {
        let chunks = self
            .splitter
            .chunks(content)
            .map(|x| x.to_string())
            .collect();
        Ok(Document { chunks })
    }
}
