use std::collections::{HashMap, HashSet};

use anyhow::Result;
use scraper::{Html, Selector};
use serde_json::json;
use url::Url;

use crate::{
    embedding_model::{
        embed::{EmbedData, TextEmbed},
        get_text_metadata,
    },
    text_loader::TextLoader,
};

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
    pub fn embed_webpage<T: TextEmbed>(&self, embeder: &T, chunk_size: usize) -> Result<Vec<EmbedData>> {
        let mut embed_data = Vec::new();

        if let Some(paragraphs) = &self.paragraphs {
            embed_data.extend(self.embed_tag("p", paragraphs, embeder, chunk_size)?);
        }

        if let Some(headers) = &self.headers {
            embed_data.extend(self.embed_tag("h1", headers, embeder, chunk_size)?);
        }

        if let Some(codes) = &self.codes {
            embed_data.extend(self.embed_tag("code", codes, embeder, chunk_size)?);
        }

        Ok(embed_data)
    }

    pub fn embed_tag<T: TextEmbed>(
        &self,
        tag: &str,
        tag_content: &[String],
        embeder: &T,
        chunk_size: usize,
    ) -> Result<Vec<EmbedData>> {
        let mut embed_data = Vec::new();

        for content in tag_content {
            let chunks = match TextLoader::split_into_chunks(content, chunk_size) {
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

            let encodings = embeder.embed(&chunks)?;
            let embeddings = get_text_metadata(&encodings, &chunks, Some(metadata_hashmap))?;
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

impl Default for WebsiteProcessor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct WebsiteProcessor;

impl WebsiteProcessor {
    pub fn new() -> Self {
        Self {}
    }

    pub fn process_website(&self, website: &str) -> Result<WebPage> {
        let response = reqwest::blocking::get(website)?.text()?;
        let document = Html::parse_document(&response);
        let headers = self.get_text_from_tag("h1,h2,h3", &document)?;
        let paragraphs = self.get_text_from_tag("p", &document)?;
        let codes = self.get_text_from_tag("code", &document)?;
        let links = self.extract_links(website, &document)?;
        let title = self.get_title(&document)?;
        let web_page = WebPage {
            url: website.to_string(),
            title,
            headers: Some(headers),
            paragraphs: Some(paragraphs),
            codes: Some(codes),
            links: Some(links),
        };

        Ok(web_page)
    }

    fn get_text_from_tag(&self, tag: &str, document: &Html) -> Result<Vec<String>, anyhow::Error> {
        let selector = Selector::parse(tag).unwrap();
        Ok(document
            .select(&selector)
            .map(|element| element.text().collect::<String>().trim().to_string())
            .collect())
    }

    fn extract_links(&self, website: &str, document: &Html) -> Result<HashSet<String>> {
        let mut links = HashSet::new();
        let base_url = Url::parse(website)?;

        for element in document.select(&Selector::parse("a").unwrap()) {
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
        if let Some(title_element) = document.select(&Selector::parse("title").unwrap()).next() {
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
    fn test_process_website() {
        let website_processor = WebsiteProcessor::new();
        let website = "https://www.scrapingbee.com/blog/web-scraping-rust/";
        let result = website_processor.process_website(website);
        assert!(result.is_ok());
    }
}
