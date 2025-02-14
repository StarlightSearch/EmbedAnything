use anyhow::Error;
use anyhow::Result;
use text_splitter::{ChunkConfig, MarkdownSplitter};
use crate::embeddings::embed::{EmbedData, Embedder};

/// A struct that provides functionality to process Markdown files.
pub struct MarkdownProcessor;

impl MarkdownProcessor {

    pub fn new() -> MarkdownProcessor { Self }

    /// Extracts the contents of some Markdown text.
    ///
    /// # Arguments
    ///
    /// * `markdown` - The Markdown text to be extracted.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the extracted content as an `MarkdownDocument` if successful,
    /// or an `Error` if an error occurred during any part of the process.
    pub fn process_markdown(
        &self,
        markdown: impl Into<String>,
    ) -> Result<MarkdownDocument> {
        Ok(MarkdownDocument{ content: markdown.into() })
    }

    /// Extracts the contents of a Markdown file.
    ///
    /// # Arguments
    ///
    /// * `file_path` - The path to the md file.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the extracted content as a `MarkdownDocument` if successful,
    /// or an `Error` if an error occurred during any part of the process.
    pub fn process_markdown_file(
        &self,
        file_path: impl AsRef<std::path::Path>,
    ) -> Result<MarkdownDocument> {
        let bytes = std::fs::read(file_path)?;
        let out = String::from_utf8_lossy(&bytes);
        self.process_markdown(out)
    }

    /// Extracts the text content from a Markdown file.
    ///
    /// # Arguments
    ///
    /// * `file_path` - The path to the Markdown file.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the extracted text content as a `String` if successful,
    /// or an `Error` if an error occurred while reading the file or converting the Markdown.
    pub fn extract_text<T: AsRef<std::path::Path>>(file_path: &T) -> Result<String, Error> {
        let bytes = std::fs::read(file_path)?;
        let out = String::from_utf8_lossy(&bytes).to_string();
        let content = markdown_to_text::convert(&out);
        Ok(content)
    }
}

pub struct MarkdownDocument {
    content: String,
}

impl MarkdownDocument {
    pub async fn embed_markdown(
        &self,
        embedder: &Embedder,
        chunk_size: usize,
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbedData>> {
        let splitter_config = ChunkConfig::new(chunk_size);
        let splitter = MarkdownSplitter::new(splitter_config);
        splitter.chunks(&self.content)
            .map(|part| {
                embedder.embed(&[part.to_string()], batch_size)
            })
            .flatten()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_extract_text() {
        let file_path = "test_files/test.md";

        let result = MarkdownProcessor::extract_text(&file_path).unwrap();
        assert_eq!(result, "Hello, world!\n\nHow are you\n\nI am good");
    }

    // returns Err if file does not exist
    #[test]
    fn test_extract_text_file_not_exist() {
        let file_path = "nonexistent_file.md";

        let result = MarkdownProcessor::extract_text(&file_path);
        assert!(result.is_err());
    }
}
