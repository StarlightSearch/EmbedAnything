use std::path::Path;

pub trait DocumentProcessor {
    fn process_document(&self, content: &str) -> anyhow::Result<Document>;
}

pub trait FileProcessor {
    fn process_file(&self, path: impl AsRef<Path>) -> anyhow::Result<Document>;
}

pub trait UrlProcessor {
    fn process_url(&self, url: &str) -> anyhow::Result<Document>;
}

impl<T: DocumentProcessor> FileProcessor for T {
    fn process_file(&self, path: impl AsRef<Path>) -> anyhow::Result<Document> {
        let bytes = std::fs::read(path)?;
        let out = String::from_utf8_lossy(&bytes);
        self.process_document(&out)
    }
}

impl<T: DocumentProcessor> UrlProcessor for T {
    fn process_url(&self, url: &str) -> anyhow::Result<Document> {
        let client = reqwest::blocking::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .default_headers({
                let mut headers = reqwest::header::HeaderMap::new();
                headers.insert(
                    reqwest::header::ACCEPT,
                    reqwest::header::HeaderValue::from_static("text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"),
                );
                headers.insert(
                    reqwest::header::ACCEPT_LANGUAGE,
                    reqwest::header::HeaderValue::from_static("en-US,en;q=0.9"),
                );
                headers.insert(
                    reqwest::header::ACCEPT_ENCODING,
                    reqwest::header::HeaderValue::from_static("gzip, deflate, br"),
                );
                headers.insert(
                    reqwest::header::CONNECTION,
                    reqwest::header::HeaderValue::from_static("keep-alive"),
                );
                headers.insert(
                    reqwest::header::UPGRADE_INSECURE_REQUESTS,
                    reqwest::header::HeaderValue::from_static("1"),
                );
                headers
            })
            .build()?;
        
        let content = client.get(url).send()?.text()?;
        self.process_document(&content)
    }
}

pub struct Document {
    pub chunks: Vec<String>,
}
