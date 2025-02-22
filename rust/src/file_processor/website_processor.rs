use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use anyhow::Result;
use serde_json::json;

use crate::{
    embeddings::{
        embed::{EmbedData, Embedder},
        get_text_metadata,
    },
    file_processor::html_processor::HtmlProcessor,
    text_loader::TextLoader,
};
use crate::config::SplittingStrategy;
use crate::file_processor::markdown_processor::MarkdownDocument;
use crate::file_processor::processor::DocumentProcessor;

#[derive(Debug)]
pub struct WebPage {
    pub url: String,
    pub title: Option<String>,
    pub headers: Option<Vec<String>>,
    pub paragraphs: Option<Vec<String>>,
    pub codes: Option<Vec<String>>,
    pub links: Option<HashSet<String>>,
}

impl WebPage {
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
                match textloader.split_into_chunks(content, SplittingStrategy::Sentence) {
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
                "url": self.url,
                "type": tag_type,
                "full_text": content,
            });

            let metadata_hashmap: HashMap<String, String> = serde_json::from_value(metadata)?;
            let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
            let encodings = embedder.embed(&chunk_refs, batch_size).await?;
            let embeddings =
                get_text_metadata(&Rc::new(encodings), &chunk_refs, &Some(metadata_hashmap))?;
            embed_data.extend(embeddings);
        }

        Ok(embed_data)
    }
}

impl Default for WebPage {
    fn default() -> Self {
        Self {
            url: "".to_string(),
            title: None,
            headers: None,
            paragraphs: None,
            codes: None,
            links: None,
        }
    }
}
pub struct WebsiteProcessor {
    html_processor: HtmlProcessor,
}

impl WebsiteProcessor {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            html_processor: HtmlProcessor::new(chunk_size),
        }
    }

    pub fn process_website(&self, website: &str) -> Result<MarkdownDocument> {
        // check if https is in the website. If not, add it.
        let website = if website.starts_with("http") {
            website
        } else {
            &format!("https://{}", website)
        };

        let response = reqwest::blocking::get(website)?.text()?;
        let result_md = self.html_processor.process_document(&response);

        Ok(result_md)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_website() {
        let website_processor = WebsiteProcessor::new(512);
        let website = "https://www.scrapingbee.com/blog/web-scraping-rust/";
        let result = website_processor.process_website(website);
        assert!(result.is_ok());
    }
}
