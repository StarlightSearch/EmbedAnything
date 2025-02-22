pub trait DocumentProcessor {
    type DocumentType: Document;

    fn process_document(&self, content: &str) -> Self::DocumentType;
}

pub trait FileProcessor {
    type DocumentType: Document;

    fn process_file(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<Self::DocumentType>;
}

pub trait Document {
    fn chunks(&self) -> impl Iterator<Item = String>;
}
