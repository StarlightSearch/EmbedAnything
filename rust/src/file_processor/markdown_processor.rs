use text_splitter::{Characters, ChunkConfig, MarkdownSplitter};
use crate::file_processor::processor::{Document, DocumentProcessor};

/// A struct that provides functionality to process Markdown files.
pub struct MarkdownProcessor {
    splitter: MarkdownSplitter<Characters>
}

impl MarkdownProcessor {
    pub fn new(chunk_size: usize) -> MarkdownProcessor {
        let splitter_config = ChunkConfig::new(chunk_size);
        let splitter = MarkdownSplitter::new(splitter_config);
        MarkdownProcessor {
            splitter
        }
    }
}

impl DocumentProcessor for MarkdownProcessor {

    fn process_document(&self, content: &str) -> Document {
        let chunks = self.splitter.chunks(content).into_iter()
            .map(|x| x.to_string())
            .collect();
        Document {
            chunks
        }
    }
}
