use crate::markdown_processor::MarkdownProcessor;
use crate::processor::{Document, DocumentProcessor};
use anyhow::Result;
use htmd::{HtmlToMarkdown, HtmlToMarkdownBuilder};
use text_splitter::ChunkConfigError;

pub struct HtmlDocument {
    pub content: String,
    pub origin: Option<String>,
}

/// A Struct for processing HTML files.
pub struct HtmlProcessor {
    markdown_processor: MarkdownProcessor,
    html_to_markdown: HtmlToMarkdown,
}

impl HtmlProcessor {
    pub fn new(chunk_size: usize, overlap: usize) -> Result<HtmlProcessor, ChunkConfigError> {
        let markdown_processor = MarkdownProcessor::new(chunk_size, overlap)?;
        let html_to_markdown = HtmlToMarkdownBuilder::new().build();
        Ok(HtmlProcessor {
            markdown_processor,
            html_to_markdown,
        })
    }
}

impl DocumentProcessor for HtmlProcessor {
    fn process_document(&self, content: &str) -> Result<Document> {
        let content = self.html_to_markdown.convert(content)?;
        self.markdown_processor.process_document(&content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::FileProcessor;

    #[test]
    fn test_process_html_file() {
        let html_processor = HtmlProcessor::new(128, 0).unwrap();
        let html_file = "../test_files/test.html";
        let result = html_processor.process_file(html_file);
        assert!(result.is_ok());
    }

    #[test]
    fn test_process_html_file_err() {
        let html_processor = HtmlProcessor::new(128, 0).unwrap();
        let html_file = "../test_files/some_file_that_doesnt_exist.html";
        let result = html_processor.process_file(html_file);
        assert!(result.is_err());
    }
}
