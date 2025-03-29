use crate::file_processor::markdown_processor::MarkdownProcessor;
use anyhow::Result;
use htmd::HtmlToMarkdown;

pub struct HtmlDocument {
    pub content: String,
    pub origin: Option<String>,
}

/// A Struct for processing HTML files.
pub struct HtmlProcessor {
    markdown_processor: MarkdownProcessor,
}

impl Default for HtmlProcessor {
    fn default() -> Self {
        Self::new(MarkdownProcessor)
    }
}

impl HtmlProcessor {
    pub fn new(markdown_processor: MarkdownProcessor) -> Self {
        Self { markdown_processor}
    }

    /// Extracts the contents of an HTML file.
    ///
    /// # Arguments
    ///
    /// * `file_path` - The path to the HTML file.
    /// * `origin` - The original URL of the HTML page, if any. This is required for extracting links.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the extracted content as an `HtmlDocument` if successful,
    /// or an `Error` if an error occurred during any part of the process.
    pub fn process_html_file(
        &self,
        file_path: impl AsRef<std::path::Path>,
        origin: Option<&str>,
    ) -> Result<HtmlDocument> {
        // check if https is in the website. If not, add it.
        let bytes = std::fs::read(file_path)?;
        let out = String::from_utf8_lossy(&bytes);
        self.process_html(out, origin)
    }

    /// Extracts the contents of an HTML text.
    ///
    /// # Arguments
    ///
    /// * `html` - The HTML text to be extracted.
    /// * `origin` - The original URL of the HTML, if any. This is required for extracting links.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the extracted content as an `HtmlDocument` if successful,
    /// or an `Error` if an error occurred during any part of the process.
    pub fn process_html(
        &self,
        html: impl Into<String>,
        origin: Option<impl Into<String>>,
    ) -> Result<HtmlDocument> {
        let content = HtmlToMarkdown::new().convert(&html.into())?;
        let result = self.markdown_processor.process_markdown(content)?;
        Ok(HtmlDocument {
            content: result,
            origin: origin.map(Into::into),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_html_file() {
        let html_processor = HtmlProcessor::default();
        let html_file = "../test_files/test.html";
        let result = html_processor.process_html_file(html_file, Some("https://example.com/"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_process_html_file_err() {
        let html_processor = HtmlProcessor::default();
        let html_file = "../test_files/some_file_that_doesnt_exist.html";
        let result = html_processor.process_html_file(html_file, Some("https://example.com/"));
        assert!(result.is_err());
    }
}
