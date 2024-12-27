use anyhow::Error;
use docx_parser::MarkdownDocument;
/// A struct for processing PDF files.
pub struct DocxProcessor;

impl DocxProcessor {
    /// Extracts text from a PDF file.
    ///
    /// # Arguments
    ///
    /// * `file_path` - The path to the PDF file.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the extracted text as a `String` if successful,
    /// or an `Error` if an error occurred during the extraction process.
    pub fn extract_text<T: AsRef<std::path::Path>>(file_path: &T) -> Result<String, Error> {
        let docs = MarkdownDocument::from_file(file_path);
        let markdown = docs.to_markdown(false);

        Ok(markdown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_extract_text() {
        // let temp_dir = TempDir::new("example").unwrap();
        // let txt_file = temp_dir.path().join("test.txt");

        // File::create(&txt_file).unwrap();
        let txt_file = "test_files/test.docx";

        DocxProcessor::extract_text(&txt_file).unwrap_err();

    }

    // Returns an error if the file path is invalid.
    #[test]
    fn test_extract_text_invalid_file_path() {
        let invalid_file_path = "invalid.txt";

        DocxProcessor::extract_text(&invalid_file_path).unwrap_err();
    }
}
