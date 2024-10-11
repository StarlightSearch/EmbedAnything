use anyhow::Error;

/// A struct for processing PDF files.
pub struct PdfProcessor;

impl PdfProcessor {
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
    pub fn extract_text<T: AsRef<std::path::Path>>(file_path: T) -> Result<String, Error> {
        pdf_extract::extract_text(file_path).map_err(|e| anyhow::anyhow!(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempdir::TempDir;

    #[test]
    fn test_extract_text() {
        let temp_dir = TempDir::new("example").unwrap();
        let pdf_file = temp_dir.path().join("test.pdf");

        File::create(pdf_file).unwrap();

        let pdf_file = "test_files/test.pdf";
        let text = PdfProcessor::extract_text(&pdf_file).unwrap();
        assert_eq!(text.len(), 4271);
    }
}
