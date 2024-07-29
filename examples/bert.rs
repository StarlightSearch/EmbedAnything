use embed_anything::config::{BertConfig, EmbedConfig, JinaConfig};
use embed_anything::embed_directory;
use std::{path::PathBuf, time::Instant};

fn main() {
    let now = Instant::now();

    let bert_config = BertConfig {
        model_id: Some("sentence-transformers/all-MiniLM-L12-v2".to_string()),
        revision: None,
        chunk_size: Some(100),
    };

    let jina_config = JinaConfig {
        model_id: Some("jinaai/jina-embeddings-v2-base-en".to_string()),
        revision: None,
        chunk_size: Some(100),
    };

    let config = EmbedConfig {
        bert: Some(bert_config),
        jina: Some(jina_config),
        ..Default::default()
    };

    let _out_bert = embed_directory(
        PathBuf::from("test_files"),
        "Bert",
        Some(vec!["pdf".to_string()]),
        Some(&config),
        None,
    )
    .unwrap()
    .unwrap();

    let _out_jina = embed_directory(
        PathBuf::from("test_files"),
        "Jina",
        Some(vec!["pdf".to_string()]),
        Some(&config),
        None,
    )
    .unwrap()
    .unwrap();

    let elapsed_time = now.elapsed();
    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());
}
