use std::io::Error;

pub struct PdfProcessor;

impl PdfProcessor {
    pub fn extract_text(file_path: &str) -> Result<String, Error> {
        let bytes = std::fs::read(file_path).unwrap();
        let out = pdf_extract::extract_text_from_mem(&bytes).unwrap();
        Ok(out)
    }
}
