use std::sync::Arc;

use embed_anything::{config::TextEmbedConfig, embeddings::embed::EmbedderBuilder};

#[tokio::main]
async fn main() {
    let model = Arc::new(
        EmbedderBuilder::new()
            .model_id(Some("jinaai/jina-embeddings-v2-small-en"))
            .revision(None)
            .path_in_repo(Some("model.onnx"))
            .from_pretrained_onnx()
            .unwrap(),
    );

    let config = TextEmbedConfig::default()
        .with_chunk_size(1000, Some(0.3))
        .with_batch_size(4)
        .with_buffer_size(32)
        .with_late_chunking(true);

    let out = model
        .embed_file("test_files/test.pdf", Some(&config), None)
        .await
        .unwrap()
        .unwrap();

    for d in out {
        println!("{}", d.text.unwrap());
        println!("---");
    }
}
