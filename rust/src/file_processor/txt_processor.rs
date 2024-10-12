use anyhow::Error;

/// A struct for processing PDF files.
pub struct TxtProcessor;

impl TxtProcessor {
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
        let bytes = std::fs::read(file_path)?;
        let out = String::from_utf8_lossy(&bytes);
        Ok(out.to_string())
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
        let txt_file = temp_dir.path().join("test.txt");

        File::create(&txt_file).unwrap();

        let text = TxtProcessor::extract_text(&txt_file).unwrap();
        assert_eq!(text, "");

        let txt_file = "test_files/test.txt";
        let text = TxtProcessor::extract_text(&txt_file).unwrap();
        assert_eq!(
            text,
            "This is a test file to see how txt embedding works !\n"
        );
    }

    // Returns an error if the file path is invalid.
    #[test]
    fn test_extract_text_invalid_file_path() {
        let invalid_file_path = "invalid.txt";

        let result = TxtProcessor::extract_text(&invalid_file_path);
        assert!(result.is_err());
    }
}
