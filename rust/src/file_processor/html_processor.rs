use crate::file_processor::markdown_processor::{MarkdownDocument, MarkdownProcessor};
use crate::file_processor::processor::DocumentProcessor;
use htmd::{HtmlToMarkdown, HtmlToMarkdownBuilder};

/// A Struct for processing HTML files.
pub struct HtmlProcessor {
    markdown_processor: MarkdownProcessor,
    converter: HtmlToMarkdown,
}

impl HtmlProcessor {
    pub fn new(chunk_size: usize) -> HtmlProcessor {
        let converter = HtmlToMarkdownBuilder::new()
            .build();
        HtmlProcessor {
            markdown_processor: MarkdownProcessor::new(chunk_size),
            converter,
        }
    }
}

impl DocumentProcessor for HtmlProcessor {
    type DocumentType = MarkdownDocument;

    fn process_document(&self, content: &str) -> Self::DocumentType {
        let md = self.converter.convert(content).unwrap();
        self.markdown_processor.process_document(&md)
    }
}
