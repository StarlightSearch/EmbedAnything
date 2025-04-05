use anyhow::Result;

use crate::file_processor::html_processor::HtmlProcessor;

#[derive(Debug)]
pub struct WebPage {
    pub url: String,
    pub content: String,
}

impl Default for WebsiteProcessor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct WebsiteProcessor {
    html_processor: HtmlProcessor,
}

impl WebsiteProcessor {
    pub fn new() -> Self {
        Self {
            html_processor: HtmlProcessor::default(),
        }
    }

    pub fn process_website(&self, website: &str) -> Result<WebPage> {
        // check if https is in the website. If not, add it.
        let website = if website.starts_with("http") {
            website
        } else {
            &format!("https://{}", website)
        };

        let response = reqwest::blocking::get(website)?.text()?;
        let html_document = self.html_processor.process_html(response, Some(website))?;

        let web_page = WebPage {
            url: html_document.origin.unwrap(), // We always have Some as per the above code
            content: html_document.content,
        };

        Ok(web_page)
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
