use rayon::iter::ParallelBridge;
use text_splitter::{Characters, ChunkConfig, TextSplitter};
use crate::chunkers::statistical::StatisticalChunker;
use crate::config::SplittingStrategy;
use crate::file_processor::processor::{Document, DocumentProcessor};

/// A struct for processing PDF files.
pub struct TxtProcessor {
    splitter: TextSplitter<Characters>,
    splitting_strategy: SplittingStrategy,
}

impl TxtProcessor {
    pub fn new(
        chunk_size: usize,
        overlap_ratio: f32,
        splitting_strategy: SplittingStrategy,
    ) -> TxtProcessor {
        let splitter = TextSplitter::new(
            ChunkConfig::new(chunk_size)
                .with_overlap(chunk_size * overlap_ratio as usize)
                .unwrap()
        );
        TxtProcessor { splitter, splitting_strategy }
    }
}

impl DocumentProcessor for TxtProcessor {
    type DocumentType = TxtDocument;

    fn process_document(&self, content: &str) -> Self::DocumentType {
        // Remove single newlines but keep double newlines
        let cleaned_text = content
            .replace("\n\n", "{{DOUBLE_NEWLINE}}")
            .replace("\n", " ")
            .replace("{{DOUBLE_NEWLINE}}", "\n\n");
        let chunks: dyn Iterator<Item = String> = match &self.splitting_strategy {
            SplittingStrategy::Sentence => self
                .splitter
                .chunks(&cleaned_text)
                .par_bridge()
                .map(|chunk| chunk.to_string())
                .collect(),
            SplittingStrategy::Semantic { semantic_encoder } => {
                let chunker = StatisticalChunker {
                    encoder: semantic_encoder,
                    ..Default::default()
                };

                tokio::task::block_in_place(|| {
                    tokio::runtime::Runtime::new()
                        .unwrap()
                        .block_on(async { chunker.chunk(&cleaned_text, 64).await })
                })
            }
        };

        TxtDocument {
            segment_iterator: chunks,
        }
    }
}

pub struct TxtDocument {
    segment_iterator: Box<dyn Iterator<Item = String>>,
}

impl Document for TxtDocument {
    fn chunks(&self) -> impl Iterator<Item=String> {
        &self.segment_iterator
    }
}
