use std::sync::Arc;

use embed_anything::{config::TextEmbedConfig, embeddings::embed::EmbedderBuilder};

#[tokio::main]
async fn main() {
    let model = Arc::new(
        EmbedderBuilder::new()
            .model_architecture("qwen3")
            .model_id(Some("Qwen/Qwen3-Embedding-0.6B"))
            .revision(None)
            .token(None)
            .from_pretrained_hf()
            .unwrap(),
    );

    let config = TextEmbedConfig::default()
        .with_chunk_size(1000, Some(0.3))
        .with_batch_size(2)
        .with_buffer_size(32)
        .with_late_chunking(true);
    let out = model
        .embed_file("test_files/attention.pdf", Some(&config), None)
        .await
        .unwrap()
        .unwrap();

    for d in out {
        println!("{}", d.text.unwrap());
        println!("---");
    }
}
