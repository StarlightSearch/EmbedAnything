use embed_anything::config::{BertConfig, EmbedConfig};
use embed_anything::embed_directory;
use std::{path::PathBuf, time::Instant};

fn main() {
    let now = Instant::now();

    let bert_config = BertConfig {
        model_id: Some("sentence-transformers/all-MiniLM-L12-v2".to_string()),
        revision: None,
        chunk_size: Some(100),
    };

    let config = EmbedConfig {
        bert: Some(bert_config),
        ..Default::default()
    };

    let out = embed_directory(
        PathBuf::from("test_files"),
        "Bert",
        Some(vec!["pdf".to_string()]),
        Some(&config),
    )
    .unwrap();

    let elapsed_time = now.elapsed();
    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());
}
