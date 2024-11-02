use embed_anything::config::TextEmbedConfig;
use embed_anything::embeddings::embed::{EmbedData, Embedder, TextEmbedder};
use embed_anything::text_loader::SplittingStrategy;
use embed_anything::{embed_directory_stream, embed_file};
use std::sync::Arc;
use std::{path::PathBuf, time::Instant};

#[tokio::main]
async fn main() {
    let model = Arc::new(Embedder::Text(
        TextEmbedder::from_pretrained_hf("jina", "jinaai/jina-embeddings-v2-small-en", None)
            .unwrap(),
    ));
    let config = TextEmbedConfig::default()
        .with_chunk_size(256)
        .with_batch_size(32)
        .with_buffer_size(32)
        .with_splitting_strategy(SplittingStrategy::Semantic)
        .with_semantic_encoder(Arc::clone(&model));

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
        PathBuf::from("bench"),
        &model,
        None,
        // Some(vec!["txt".to_string()]),
        Some(&config),
        None::<fn(Vec<EmbedData>)>,
    )
    .await
    .unwrap()
    .unwrap();

    println!("Number of chunks: {:?}", _out.len());
    let elapsed_time = now.elapsed();
    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());
}
