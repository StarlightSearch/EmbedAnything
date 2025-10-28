use std::sync::Arc;

use embed_anything::{config::TextEmbedConfig, embeddings::embed::Embedder};

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let text_embed_config = TextEmbedConfig::default()
        .with_chunk_size(1000, Some(0.3))
        .with_batch_size(8)
        .with_buffer_size(8);
    let cohere_model: Arc<Embedder> =
        Arc::new(Embedder::from_pretrained_cloud("cohere-vision", "embed-v4.0", None).unwrap());

    // let embeddings = cohere_model.embed_query(&["What are Positional Encodings"], Some(&text_embed_config)).await?;
    let embeddings = cohere_model
        .embed_file("test_files/colpali.pdf", Some(&text_embed_config), None)
        .await?;
    println!("{:?}", embeddings.unwrap().len());
    Ok(())
}
