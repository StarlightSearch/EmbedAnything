use embed_anything::config::TextEmbedConfig;
use embed_anything::embed_directory;
use embed_anything::embeddings::embed::{EmbedData, Embeder};
use std::{path::PathBuf, time::Instant};

fn main() {
    let now = Instant::now();

    let model = Embeder::from_pretrained_hf("bert", "sentence-transformers/all-MiniLM-L6-v2", None).unwrap();
    let config = TextEmbedConfig::new(Some(256), Some(32));
    let _out = embed_directory(
        PathBuf::from("test_files"),
        &model,
        Some(vec!["pdf".to_string()]),
        Some(&config),
        None::<fn(Vec<EmbedData>)>,
    )
    .unwrap();

    let elapsed_time = now.elapsed();
    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());
}