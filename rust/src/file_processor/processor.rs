pub trait DocumentProcessor {

    fn process_document(&self, content: &str) -> Document;
}

pub trait FileProcessor {

    fn process_file(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<Document>;
}

pub struct Document {
    pub chunks: Vec<String>
}
