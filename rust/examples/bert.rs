use embed_anything::config::{SplittingStrategy, TextEmbedConfig};
use embed_anything::embeddings::embed::EmbedderBuilder;
use embed_anything::{embed_directory_stream, embed_file, embed_files_batch};
use std::collections::HashSet;
use std::sync::Arc;
use std::{path::PathBuf, time::Instant};

#[tokio::main]
async fn main() {
    let model = Arc::new(
        EmbedderBuilder::new()
            .model_id(Some("jinaai/jina-embeddings-v2-small-en"))
            .revision(None)
            .token(None)
            .from_pretrained_hf()
            .unwrap(),
    );

    let config = TextEmbedConfig::default()
        .with_chunk_size(1000, Some(0.3))
        .with_batch_size(32)
        .with_buffer_size(32)
        .with_splitting_strategy(SplittingStrategy::Sentence);

    let now = Instant::now();

    let _out = embed_file("test_files/test.pdf", &model, Some(&config), None)
        .await
        .unwrap()
        .unwrap();

    let _out_2 = embed_files_batch(
        vec![
            PathBuf::from("test_files/test.pdf"),
            PathBuf::from("test_files/test.txt"),
        ],
        &model,
        Some(&config),
        None,
    )
    .await
    .unwrap()
    .unwrap();

    let elapsed_time: std::time::Duration = now.elapsed();

    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());

    let now = Instant::now();

    let _out = embed_directory_stream(
        PathBuf::from("test_files"),
        &model,
        None,
        // Some(vec!["txt".to_string()]),
        Some(&config),
        None,
    )
    .await
    .unwrap()
    .unwrap();

    let embedded_files = _out
        .iter()
        .map(|e| {
            e.metadata
                .as_ref()
                .unwrap()
                .get("file_name")
                .unwrap()
                .clone()
        })
        .collect::<Vec<_>>();
    let mut embedded_files_set = HashSet::new();
    embedded_files_set.extend(embedded_files);
    println!("Embedded files: {:?}", embedded_files_set);

    println!("Number of chunks: {:?}", _out.len());
    let elapsed_time: std::time::Duration = now.elapsed();
    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());
}
