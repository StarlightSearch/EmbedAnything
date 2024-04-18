use std::path::PathBuf;

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
    pub fn extract_text(file_path: &PathBuf) -> Result<String, Error> {
        let bytes = std::fs::read(file_path).unwrap();
        let out = pdf_extract::extract_text_from_mem(&bytes).unwrap();
        Ok(out)
    }
}
