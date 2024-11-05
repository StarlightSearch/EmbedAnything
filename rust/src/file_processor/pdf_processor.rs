use anyhow::Error;
use image::DynamicImage;
use pdf2image::{Pages, RenderOptionsBuilder, PDF};
use rusty_tesseract::{self, Args, Image};

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
    pub fn extract_text<T: AsRef<std::path::Path>>(
        file_path: T,
        use_ocr: bool,
    ) -> Result<String, Error> {
        if use_ocr {
            extract_text_with_ocr(&file_path)
        } else {
            pdf_extract::extract_text(file_path).map_err(|e| anyhow::anyhow!(e))
        }
    }
}

fn get_images_from_pdf<T: AsRef<std::path::Path>>(
    file_path: &T,
) -> Result<Vec<DynamicImage>, Error> {
    let pdf = PDF::from_file(file_path)?;
    let page_count = pdf.page_count();
    let pages = pdf.render(
        Pages::Range(1..=page_count),
        RenderOptionsBuilder::default().build()?,
    )?;
    Ok(pages)
}

fn extract_text_from_image(image: &DynamicImage, args: &Args) -> Result<String, Error> {
    let image = Image::from_dynamic_image(image).unwrap();
    let text = rusty_tesseract::image_to_string(&image, args).unwrap();
    Ok(text)
}

fn extract_text_with_ocr<T: AsRef<std::path::Path>>(file_path: &T) -> Result<String, Error> {
    let images = get_images_from_pdf(file_path)?;
    let texts: Result<Vec<String>, Error> = images
        .iter()
        .map(|image| extract_text_from_image(image, &Args::default()))
        .collect();
    Ok(texts.unwrap().join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempdir::TempDir;
    use text_cleaner::clean::Clean;

    #[test]
    fn test_extract_text() {
        let temp_dir = TempDir::new("example").unwrap();
        let pdf_file = temp_dir.path().join("test.pdf");

        File::create(pdf_file).unwrap();

        let pdf_file = "test_files/test.pdf";
        let text = PdfProcessor::extract_text(&pdf_file, false).unwrap();
        assert_eq!(text.len(), 4271);
    }

    #[test]
    fn test_extract_text_with_ocr() {
        let pdf_file = "../test_files/test.pdf";
        let path = std::path::Path::new(pdf_file);

        // Check if the path exists
        if !path.exists() {
            panic!("File does not exist: {}", path.display());
        }

        // Print the absolute path
        println!("Absolute path: {}", path.canonicalize().unwrap().display());

        let text = extract_text_with_ocr(&pdf_file)
            .unwrap()
            .remove_leading_spaces()
            .remove_trailing_spaces()
            .remove_empty_lines();
        println!("Text: {}", text);
    }
}
