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
        let content = markdown_to_text::convert(&markdown);
        Ok(content)
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
        let txt_file = "../test_files/test.docx";

        let text = DocxProcessor::extract_text(&txt_file).unwrap();
        assert!(text.contains("This is a docx file test"));
    }

    // Returns an error if the file path is invalid.
    #[test]
    #[should_panic(expected = "Error processing file: IO(Os { code: 2, kind: NotFound, message: \"No such file or directory\" })")]
    fn test_extract_text_invalid_file_path() {
        let invalid_file_path = "this_file_definitely_does_not_exist.docx";
        DocxProcessor::extract_text(&invalid_file_path).unwrap();
    }
}
