use crate::markdown_processor::MarkdownProcessor;
use crate::pdf::tesseract::input::{Args, Image};
use crate::processor::{Document, DocumentProcessor, FileProcessor};
use anyhow::Error;
use image::DynamicImage;
use pdf2image::{Pages, RenderOptionsBuilder, PDF};
use std::path::Path;
use text_splitter::ChunkConfigError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfBackend {
    LoPdf,
}

/// A struct for processing PDF files.
pub struct PdfProcessor {
    markdown_processor: MarkdownProcessor,
    ocr_config: OcrConfig,
    backend: PdfBackend,
}

pub struct OcrConfig {
    pub use_ocr: bool,
    pub tesseract_path: Option<String>,
}

impl PdfProcessor {
    pub fn new(
        chunk_size: usize,
        overlap: usize,
        ocr_config: OcrConfig,
        backend: PdfBackend,
    ) -> Result<PdfProcessor, ChunkConfigError> {
        let markdown_processor = MarkdownProcessor::new(chunk_size, overlap)?;
        Ok(PdfProcessor {
            markdown_processor,
            ocr_config,
            backend,
        })
    }
}

impl FileProcessor for PdfProcessor {
    fn process_file(&self, path: impl AsRef<Path>) -> anyhow::Result<Document> {
        let content = if self.ocr_config.use_ocr {
            let tesseract_path = self.ocr_config.tesseract_path.as_deref();
            extract_text_with_ocr(&path, tesseract_path)?
        } else {
            match self.backend {
                PdfBackend::LoPdf => {
                    pdf_extract::extract_text(path.as_ref()).map_err(|e| anyhow::anyhow!(e))?
                }
            }
        };

        self.markdown_processor.process_document(&content)
    }
}

fn get_images_from_pdf<T: AsRef<Path>>(file_path: &T) -> Result<Vec<DynamicImage>, Error> {
    let pdf = PDF::from_file(file_path)?;
    let page_count = pdf.page_count();
    let pages = pdf.render(
        Pages::Range(1..=page_count),
        RenderOptionsBuilder::default().build()?,
    )?;
    Ok(pages)
}

fn extract_text_from_image(image: &DynamicImage, args: &Args) -> Result<String, Error> {
    let image = Image::from_dynamic_image(image)?;
    let text = crate::pdf::tesseract::command::image_to_string(&image, args)?;
    Ok(text)
}

fn extract_text_with_ocr<T: AsRef<Path>>(
    file_path: &T,
    tesseract_path: Option<&str>,
) -> Result<String, Error> {
    let images = get_images_from_pdf(file_path)?;
    let texts: Result<Vec<String>, Error> = images
        .iter()
        .map(|image| extract_text_from_image(image, &Args::default().with_path(tesseract_path)))
        .collect();

    // Join the texts and clean up empty lines
    let text = texts?.join("\n");
    let cleaned_text = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<&str>>()
        .join("\n");

    Ok(cleaned_text)
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
        let processor = PdfProcessor::new(
            128,
            0,
            OcrConfig {
                use_ocr: false,
                tesseract_path: None,
            },
            PdfBackend::LoPdf,
        )
        .unwrap();

        File::create(pdf_file).unwrap();

        let pdf_file = "../test_files/test.pdf";
        let text = processor.process_file(pdf_file).unwrap();
        assert_eq!(text.chunks.len(), 4271);
    }

    #[test]
    fn test_extract_text_with_ocr() {
        let pdf_file = "../test_files/test.pdf";
        let path = Path::new(pdf_file);

        // Check if the path exists
        if !path.exists() {
            panic!("File does not exist: {}", path.display());
        }

        // Print the absolute path
        println!("Absolute path: {}", path.canonicalize().unwrap().display());

        let text = extract_text_with_ocr(&pdf_file, None).unwrap();

        println!("Text: {}", text);
    }
}
