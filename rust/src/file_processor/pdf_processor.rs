use std::path::Path;
use crate::tesseract::input::{Args, Image};
use anyhow::Error;
use image::DynamicImage;
use pdf2image::{Pages, RenderOptionsBuilder, PDF};
use crate::config::SplittingStrategy;
use crate::file_processor::processor::{Document, DocumentProcessor, FileProcessor};
use crate::file_processor::txt_processor::TxtProcessor;

/// A struct for processing PDF files.
pub struct PdfProcessor {
    txt_processor: TxtProcessor,
    use_ocr: bool,
    tesseract_path: Option<String>,
}

impl PdfProcessor {
    pub fn new(
        chunk_size: usize,
        overlap_ratio: f32,
        splitting_strategy: SplittingStrategy,
        use_ocr: bool,
        tesseract_path: Option<impl Into<String>>,
    ) -> PdfProcessor {
        PdfProcessor {
            txt_processor: TxtProcessor::new(chunk_size, overlap_ratio, splitting_strategy),
            use_ocr,
            tesseract_path: tesseract_path.map(Into::into),
        }
    }

    fn get_images_from_pdf<T: AsRef<Path>>(
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
        let image = Image::from_dynamic_image(image)?;
        let text = crate::tesseract::command::image_to_string(&image, args)?;
        Ok(text)
    }

    fn extract_text_with_ocr<T: AsRef<Path>>(
        file_path: &T,
        tesseract_path: Option<String>,
    ) -> Result<String, Error> {
        let images = Self::get_images_from_pdf(file_path)?;
        let texts: Result<Vec<String>, Error> = images
            .iter()
            .map(|image| Self::extract_text_from_image(image, &Args::default().with_path(tesseract_path.clone())))
            .collect();
        Ok(texts?.join("\n"))
    }

}

impl FileProcessor for PdfProcessor {

    fn process_file(&self, path: impl AsRef<Path>) -> anyhow::Result<Document> {
        let text = if self.use_ocr {
            Self::extract_text_with_ocr(&path, self.tesseract_path.clone())?
        } else {
            pdf_extract::extract_text(path).map_err(|e| anyhow::anyhow!(e))?
        };
        Ok(self.txt_processor.process_document(&text))
    }
}
