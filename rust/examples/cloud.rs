use std::{path::PathBuf, sync::Arc};

use embed_anything::{
    config::TextEmbedConfig, embed_directory_stream, embed_file, embeddings::embed::Embedder,
};

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let text_embed_config = TextEmbedConfig::default()
        .with_chunk_size(1000, Some(0.3))
        .with_batch_size(512)
        .with_buffer_size(512);
    let cohere_model =
        Embedder::from_pretrained_cloud("cohere", "embed-english-v3.0", None).unwrap(); // You can add your api key here
    let openai_model =
        Embedder::from_pretrained_cloud("openai", "text-embedding-3-small", None).unwrap(); // You can add your api key here
    let openai_model: Arc<Embedder> = Arc::new(openai_model);
    let _openai_embeddings = embed_directory_stream(
        PathBuf::from("test_files"),
        &openai_model,
        Some(vec!["pdf".to_string()]),
        Some(&text_embed_config),
        None,
    )
    .await?
    .unwrap();

    let _file_embedding = embed_file(
        "test_files/attention.pdf",
        &openai_model,
        Some(&text_embed_config),
        None,
    )
    .await?
    .unwrap();

    let _cohere_embedding = embed_file(
        "test_files/attention.pdf",
        &cohere_model,
        Some(&text_embed_config),
        None,
    )
    .await?
    .unwrap();

    println!("Cohere embedding: {:?}", _cohere_embedding);

    Ok(())
}
