use embed_anything::config::TextEmbedConfig;
use embed_anything::embeddings::embed::{EmbedData, Embedder, TextEmbedder};
use embed_anything::file_processor::docx_processor::DocxProcessor;
use embed_anything::text_loader::SplittingStrategy;
use embed_anything::{embed_directory_stream, embed_file};
use std::collections::HashSet;
use std::sync::Arc;
use std::{path::PathBuf, time::Instant};

#[tokio::main]
async fn main() {
    let model = Arc::new(Embedder::Text(
        TextEmbedder::from_pretrained_hf("jina", "jinaai/jina-embeddings-v2-small-en", None)
            .unwrap(),
    ));
    let config = TextEmbedConfig::default()
        .with_chunk_size(256, Some(0.3))
        .with_batch_size(32)
        .with_buffer_size(32)
        .with_splitting_strategy(SplittingStrategy::Sentence)
        .with_semantic_encoder(Arc::clone(&model));

    DocxProcessor::extract_text(&PathBuf::from("test_files/test.docx")).unwrap();

    let _out = embed_file(
        "test_files/test.pdf",
        &model,
        Some(&config),
        None::<fn(Vec<EmbedData>)>,
    )
    .await
    .unwrap()
    .unwrap();

    let now = Instant::now();

    let _out = embed_directory_stream(
        PathBuf::from("test_files"),
        &model,
        None,
        // Some(vec!["txt".to_string()]),
        Some(&config),
        None::<fn(Vec<EmbedData>)>,
    )
    .await
    .unwrap()
    .unwrap();

    let embedded_files = _out.iter().map(|e| e.metadata.as_ref().unwrap().get("file_name").unwrap().clone()).collect::<Vec<_>>();
    let mut embedded_files_set = HashSet::new();
    embedded_files_set.extend(embedded_files);
    println!("Embedded files: {:?}", embedded_files_set);

    println!("Number of chunks: {:?}", _out.len());
    let elapsed_time = now.elapsed();
    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());
}
