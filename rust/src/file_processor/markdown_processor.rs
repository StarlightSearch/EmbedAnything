use anyhow::Error;
use anyhow::Result;
use text_splitter::{ChunkConfig, MarkdownSplitter};

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
        chunk_size: usize,
    ) -> Result<MarkdownDocument> {
        let splitter_config = ChunkConfig::new(chunk_size);
        let splitter = MarkdownSplitter::new(splitter_config);
        let segments = splitter.chunks(markdown).into_iter()
            .map(|x| x.to_string())
            .collect();
        Ok(MarkdownDocument {
            segments,
        })
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
        chunk_size: usize,
    ) -> Result<MarkdownDocument> {
        let bytes = std::fs::read(file_path)?;
        let out = String::from_utf8_lossy(&bytes);
        self.process_markdown(out, chunk_size)
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
    pub segments: Vec<String>,
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
