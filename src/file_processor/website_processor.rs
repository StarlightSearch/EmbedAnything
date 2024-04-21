use std::collections::{HashMap, HashSet};

use anyhow::{Error, Ok};
use regex::Regex;
use scraper::Selector;
use serde_json::json;
use text_cleaner::clean::Clean;

use crate::{
    embedding_model::embed::{EmbedData, TextEmbed},
    file_embed::FileEmbeder,
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
    pub async fn embed_webpage<T: TextEmbed>(&self, embeder: &T) -> Result<Vec<EmbedData>, Error>{
        let mut embed_data = Vec::new();
        let paragraph_embeddings = if let Some(paragraphs) = &self.paragraphs {
            self.embed_tag::<T>("p", paragraphs.to_vec(), &embeder).await.unwrap_or(Vec::new())
        } else {
            Vec::new()
        };

        let header_embeddings = if let Some(headers) = &self.headers {
            self.embed_tag::<T>("h1", headers.to_vec(), &embeder).await.unwrap_or(Vec::new())
        } else {
            Vec::new()
        };

        let code_embeddings = if let Some(codes) = &self.codes {
            self.embed_tag::<T>("code", codes.to_vec(), &embeder).await.unwrap_or(Vec::new())
        } else {
            Vec::new()
        };

        embed_data.extend(paragraph_embeddings);
        embed_data.extend(header_embeddings);
        embed_data.extend(code_embeddings);
        Ok(embed_data)
    }

    pub async fn embed_tag<T: TextEmbed>(&self,tag: &str,  tag_content: Vec<String>, embeder: &T) -> Result<Vec<EmbedData>, Error> {
        let mut embed_data = Vec::new();
        for content in tag_content {
            let mut file_embeder = FileEmbeder::new(self.url.to_string());

            let chunks = match file_embeder.split_into_chunks(&content, 1000) {
                Some(chunks) => chunks,
                None => continue,
            };

            match chunks.len() {
                0 => continue,
                _ => (),
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

            let metadata_hashmap: HashMap<String, String> =
                serde_json::from_value(metadata).unwrap();
      

            let embeddings = embeder
                .embed(&chunks, Some(metadata_hashmap))
                .await
                .unwrap_or(Vec::new());
            for embedding in embeddings {
                embed_data.push(embedding);
               
            }
        }
        Ok(embed_data)
    }
}

/// A struct for processing websites.
pub struct WebsiteProcesor;


impl WebsiteProcesor {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn process_website(&self, website: &str) -> Result<WebPage, Error> {
        let response = reqwest::get(website).await?.text().await?;
        let document = scraper::Html::parse_document(&response);
        let headers = self.get_text_from_tag("h1,h2,h3", &document)?;
        let paragraphs = self.get_text_from_tag("p", &document)?;
        let codes = self.get_text_from_tag("code", &document)?;
        let links = self.extract_links(website, &document)?;
        let binding = self.get_text_from_tag("h1", &document)?;
        let title = binding.first();
        let web_page = WebPage {
            url: website.to_string(),
            title: title.map(|s| s.to_string()),
            headers: Some(headers),
            paragraphs: Some(paragraphs),
            codes: Some(codes),
            links: Some(links),
        };

        Ok(web_page)
    }

    pub fn get_text_from_tag(
        &self,
        tag: &str,
        document: &scraper::Html,
    ) -> Result<Vec<String>, Error> {
        let selector = Selector::parse(tag).map_err(|e| Error::msg(e.to_string()))?;
        Ok(document
            .select(&selector)
            .map(|element| element.text().collect::<String>().trim())
            .collect::<Vec<_>>())
    }

    pub fn extract_links(
        &self,
        website: &str,
        document: &scraper::Html,
    ) -> Result<HashSet<String>, Error> {
        let mut links = HashSet::new();
        let _ = document
            .select(&Selector::parse("a").unwrap())
            .map(|element| {
                let link = element.value().attr("href").unwrap_or_default().to_string();
                let regex: Regex = Regex::new(
                    r"^((https?|ftp|smtp):\/\/)?(www.)?[a-z0-9]+\.[a-z]+(\/[a-zA-Z0-9#]+\/?)*$",
                )
                .unwrap();
                // Check if the link is a valid URL using regex. If not append the website URL to the beginning of the link.
                if !regex.is_match(&link) {
                    links.insert(format!("{}{}", website, link));
                } else {
                    links.insert(link);
                }
            });

        Ok(links)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_process_website() {
        let website_processor = WebsiteProcesor;
        let website = "https://www.scrapingbee.com/blog/web-scraping-rust/";
        let result = website_processor.process_website(website);
        assert!(result.await.is_ok());
    }
}
