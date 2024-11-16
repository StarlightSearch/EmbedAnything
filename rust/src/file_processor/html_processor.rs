use anyhow::Result;
use std::collections::HashSet;
use scraper::{Html, Selector};
use url::Url;

#[derive(Debug)]
pub struct HtmlDocument {
    pub title: Option<String>,
    pub headers: Option<Vec<String>>,
    pub paragraphs: Option<Vec<String>>,
    pub codes: Option<Vec<String>>,
    pub links: Option<HashSet<String>>,
}

/// A Struct for processing HTML files.
pub struct HtmlProcessor;

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
    pub fn process_html_file(&self, file_path: impl AsRef<std::path::Path>, origin: Option<impl Into<String>>) -> Result<HtmlDocument> {
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
    pub fn process_html(&self, html: impl Into<String>, origin: Option<impl Into<String>>) -> Result<HtmlDocument> {
        // check if https is in the website. If not, add it.
        let document = Html::parse_document(&html.into());
        let headers = self.get_text_from_tag("h1,h2,h3", &document)?;
        let paragraphs = self.get_text_from_tag("p", &document)?;
        let codes = self.get_text_from_tag("code", &document)?;
        let links = match origin {
            Some(origin) => Some(self.extract_links(origin, &document)?),
            None => None
        };
        let title = self.get_title(&document)?;
        let web_page = HtmlDocument {
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

    fn extract_links(&self, website: impl Into<String>, document: &Html) -> Result<HashSet<String>> {
        let mut links = HashSet::new();
        let base_url = Url::parse(&website.into())?;

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
        if let Some(title_element) = document.select(&Selector::parse("title").expect("invalid selector for title")).next() {
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
