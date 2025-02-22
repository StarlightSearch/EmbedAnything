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
    type DocumentType = MarkdownDocument;

    fn process_document(&self, content: &str) -> Self::DocumentType {
        let iterator = self.splitter.chunks(content).into_iter()
            .map(|x| x.to_string());
        MarkdownDocument {
            segment_iterator: iterator.collect(),
        }
    }
}

pub struct MarkdownDocument {
    segment_iterator: Box<dyn Iterator<Item = String>>
}

impl Document for MarkdownDocument {
    fn chunks(&self) -> impl Iterator<Item=String> {
        &self.segment_iterator
    }
}