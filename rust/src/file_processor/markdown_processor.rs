use anyhow::Error;

/// A struct that provides functionality to process Markdown files.
pub struct MarkdownProcessor;

impl MarkdownProcessor {
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
        Self::process_markdown(&MarkdownProcessor, out)
    }

    /// Processes a Markdown-formatted String into a String ready for embedding.
    ///
    /// # Arguments
    ///
    /// * `markdown` - The Markdown text.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the extracted text content as a `String` if successful,
    /// or an `Error` if an error occurred while reading the file or converting the Markdown.
    pub fn process_markdown(&self, markdown: String) -> Result<String, Error> {
        let content = markdown_to_text::convert(&markdown);
        Ok(content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_extract_text() {
        let file_path = "../test_files/test.md";

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
