use crate::embeddings::embed::{EmbedData, Embedder};
use crate::embeddings::get_text_metadata;
use crate::text_loader::{SplittingStrategy, TextLoader};
use anyhow::Result;
use scraper::{Html, Selector};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use url::Url;

#[derive(Debug)]
pub struct HtmlDocument {
    pub origin: Option<String>,
    pub title: Option<String>,
    pub headers: Option<Vec<String>>,
    pub paragraphs: Option<Vec<String>>,
    pub codes: Option<Vec<String>>,
    pub links: Option<HashSet<String>>,
}

impl HtmlDocument {
    pub async fn embed_webpage(
        &self,
        embedder: &Embedder,
        chunk_size: usize,
        overlap_ratio: f32,
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbedData>> {
        let mut embed_data = Vec::new();

        if let Some(paragraphs) = &self.paragraphs {
            embed_data.extend(
                self.embed_tag(
                    "p",
                    paragraphs,
                    embedder,
                    chunk_size,
                    overlap_ratio,
                    batch_size,
                )
                .await?,
            );
        }

        if let Some(headers) = &self.headers {
            embed_data.extend(
                self.embed_tag(
                    "h1",
                    headers,
                    embedder,
                    chunk_size,
                    overlap_ratio,
                    batch_size,
                )
                .await?,
            );
        }

        if let Some(codes) = &self.codes {
            embed_data.extend(
                self.embed_tag(
                    "code",
                    codes,
                    embedder,
                    chunk_size,
                    overlap_ratio,
                    batch_size,
                )
                .await?,
            );
        }

        Ok(embed_data)
    }

    pub async fn embed_tag(
        &self,
        tag: &str,
        tag_content: &[String],
        embedder: &Embedder,
        chunk_size: usize,
        overlap_ratio: f32,
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbedData>> {
        let mut embed_data = Vec::new();

        for content in tag_content {
            let textloader = TextLoader::new(chunk_size, overlap_ratio);
            let chunks =
                match textloader.split_into_chunks(content, SplittingStrategy::Sentence, None) {
                    Some(chunks) => chunks,
                    None => continue,
                };

            if chunks.is_empty() {
                continue;
            }

            let tag_type = match tag {
                "h1" => "header",
                "h2" => "subheader",
                "h3" => "subsubheader",
                "p" => "paragraph",
                "code" => "code",
                _ => "paragraph",
            };

            let metadata = json!({
                "url": self.origin,
                "type": tag_type,
                "full_text": content,
            });

            let metadata_hashmap: HashMap<String, String> = serde_json::from_value(metadata)?;

            let encodings = embedder.embed(&chunks, batch_size).await?;
            let embeddings =
                get_text_metadata(&Rc::new(encodings), &chunks, &Some(metadata_hashmap))?;
            embed_data.extend(embeddings);
        }

        Ok(embed_data)
    }
}

/// A Struct for processing HTML files.
pub struct HtmlProcessor;

impl Default for HtmlProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl HtmlProcessor {
    pub fn new() -> Self {
        Self {}
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
        origin: Option<impl Into<String>>,
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
        // check if https is in the website. If not, add it.
        let document = Html::parse_document(&html.into());
        let headers = self.get_text_from_tag("h1,h2,h3", &document)?;
        let paragraphs = self.get_text_from_tag("p", &document)?;
        let codes = self.get_text_from_tag("code", &document)?;
        let origin = origin.map(Into::into);
        let links = match &origin {
            Some(origin) => Some(self.extract_links(&origin.clone(), &document)?),
            None => None,
        };
        let title = self.get_title(&document)?;
        let web_page = HtmlDocument {
            origin,
            title,
            headers: Some(headers),
            paragraphs: Some(paragraphs),
            codes: Some(codes),
            links,
        };

        Ok(web_page)
    }

    fn get_text_from_tag(&self, tag: &str, document: &Html) -> Result<Vec<String>> {
        let selector = Selector::parse(tag).expect("invalid selector for tag");
        Ok(document
            .select(&selector)
            .map(|element| element.text().collect::<String>().trim().to_string())
            .collect())
    }

    fn extract_links(&self, website: &str, document: &Html) -> Result<HashSet<String>> {
        let mut links = HashSet::new();
        let base_url = Url::parse(website)?;

        for element in document.select(&Selector::parse("a").expect("invalid selector for link")) {
            if let Some(href) = element.value().attr("href") {
                let mut link_url = base_url.join(href)?;
                // Normalize URLs, remove fragments and ensure they are absolute.
                link_url.set_fragment(None);
                links.insert(link_url.to_string());
            }
        }

        Ok(links)
    }

    fn get_title(&self, document: &Html) -> Result<Option<String>> {
        if let Some(title_element) = document
            .select(&Selector::parse("title").expect("invalid selector for title"))
            .next()
        {
            Ok(Some(title_element.text().collect::<String>()))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_html_file() {
        let html_processor = HtmlProcessor::new();
        let html_file = "test_files/test.html";
        let result = html_processor.process_html_file(html_file, Some("https://example.com/"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_process_html_file_err() {
        let html_processor = HtmlProcessor::new();
        let html_file = "test_files/some_file_that_doesnt_exist.html";
        let result = html_processor.process_html_file(html_file, Some("https://example.com/"));
        assert!(result.is_err());
    }
}
