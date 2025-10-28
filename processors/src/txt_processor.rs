use crate::markdown_processor::MarkdownProcessor;
use crate::processor::{Document, DocumentProcessor};
use text_splitter::ChunkConfigError;

/// A struct for processing PDF files.
pub struct TxtProcessor {
    markdown_processor: MarkdownProcessor,
}

impl TxtProcessor {
    pub fn new(chunk_size: usize, overlap: usize) -> Result<TxtProcessor, ChunkConfigError> {
        let markdown_processor = MarkdownProcessor::new(chunk_size, overlap)?;
        Ok(TxtProcessor { markdown_processor })
    }
}

impl DocumentProcessor for TxtProcessor {
    fn process_document(&self, content: &str) -> anyhow::Result<Document> {
        self.markdown_processor.process_document(content)
    }
}
