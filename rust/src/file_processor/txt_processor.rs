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

    fn process_document(&self, content: &str) -> Document {
        // Remove single newlines but keep double newlines
        let cleaned_text = content
            .replace("\n\n", "{{DOUBLE_NEWLINE}}")
            .replace("\n", " ")
            .replace("{{DOUBLE_NEWLINE}}", "\n\n");
        let chunks: Vec<String> = match self.splitting_strategy.clone() {
            SplittingStrategy::Sentence => self
                .splitter
                .chunks(&cleaned_text)
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

        Document {
            chunks
        }
    }
}
