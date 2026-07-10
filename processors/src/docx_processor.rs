use crate::markdown_processor::MarkdownProcessor;
use crate::processor::{Document, DocumentProcessor, FileProcessor};
use docx_parser::MarkdownDocument;
use std::path::Path;
use text_splitter::ChunkConfigError;

/// A struct for processing PDF files.
pub struct DocxProcessor {
    markdown_processor: MarkdownProcessor,
}

impl DocxProcessor {
    pub fn new(chunk_size: usize, overlap: usize) -> Result<DocxProcessor, ChunkConfigError> {
        let markdown_processor = MarkdownProcessor::new(chunk_size, overlap)?;
        Ok(DocxProcessor { markdown_processor })
    }
}

impl FileProcessor for DocxProcessor {
    fn process_file(&self, path: impl AsRef<Path>) -> anyhow::Result<Document> {
        // `docx-parser::MarkdownDocument::from_file` uses `panic!` instead of returning
        // `Result` when the file is missing, corrupt, or not a valid DOCX/ZIP archive.
        // We catch that panic here and convert it into a proper anyhow error so callers
        // get a clean Err(…) rather than a process-level abort.
        let path = path.as_ref().to_owned();
        let docs =
            std::panic::catch_unwind(move || MarkdownDocument::from_file(&path)).map_err(|e| {
                let msg = if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "unknown panic".to_string()
                };
                anyhow::anyhow!("docx_parser panicked while opening file: {}", msg)
            })?;
        let markdown = docs.to_markdown(false);
        self.markdown_processor.process_document(&markdown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_extract_text() {
        let txt_file = "../test_files/test.docx";
        let processor = DocxProcessor::new(128, 0).unwrap();

        let text = processor.process_file(txt_file).unwrap();
        assert!(text
            .chunks
            .contains(&"This is a docx file test".to_string()));
    }

    // Returns an error if the file path is invalid.
    #[test]
    #[should_panic(
        expected = "Error processing file: IO(Os { code: 2, kind: NotFound, message: \"No such file or directory\" })"
    )]
    fn test_extract_text_invalid_file_path() {
        let invalid_file_path = "this_file_definitely_does_not_exist.docx";
        let processor = DocxProcessor::new(128, 0).unwrap();
        processor.process_file(invalid_file_path).unwrap();
    }
}
