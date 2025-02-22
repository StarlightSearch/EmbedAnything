use crate::file_processor::markdown_processor::{MarkdownDocument, MarkdownProcessor};
use crate::file_processor::processor::DocumentProcessor;

/// A struct for processing PDF files.
pub struct DocxProcessor {
    markdown_processor: MarkdownProcessor,
}

impl DocxProcessor {
    pub fn new(chunk_size: usize) -> Self {
        DocxProcessor {
            markdown_processor: MarkdownProcessor::new(chunk_size),
        }
    }
}

impl DocumentProcessor for DocxProcessor {
    type DocumentType = MarkdownDocument;

    fn process_document(&self, content: &str) -> Self::DocumentType {
        self.markdown_processor.process_document(content)
    }
}
