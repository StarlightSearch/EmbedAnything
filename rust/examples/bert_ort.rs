use embed_anything::config::TextEmbedConfig;
use embed_anything::embeddings::embed::{EmbedData, Embeder};
use embed_anything::text_loader::SplittingStrategy;
use embed_anything::{embed_directory_stream, embed_file};
use std::sync::Arc;
use std::{path::PathBuf, time::Instant};

#[tokio::main]
async fn main() {
    let model = Arc::new(
        Embeder::from_pretrained_ort("bert", "BAAI/bge-small-en-v1.5", None).unwrap(),
    );
    let config = TextEmbedConfig::new(
        Some(256),
        Some(4),
        Some(4),
        Some(SplittingStrategy::Sentence),
        Some(model.clone()),
    );

    let now = Instant::now();

    let _out = embed_file(
        "bench/attention.pdf",
        &model,
        Some(&config),
        None::<fn(Vec<EmbedData>)>,
    )
    .await
    .unwrap()
    .unwrap();

    let elapsed_time = now.elapsed();
    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());

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