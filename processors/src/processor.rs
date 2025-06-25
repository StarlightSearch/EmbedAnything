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
        let content = reqwest::blocking::get(url)?.text()?;
        self.process_document(&content)
    }
}

pub struct Document {
    pub chunks: Vec<String>,
}
