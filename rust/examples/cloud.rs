use std::{path::PathBuf, sync::Arc};

use embed_anything::{
    config::TextEmbedConfig,
    embed_directory_stream, embed_file,
    embeddings::embed::{EmbedData, Embedder},
    text_loader::SplittingStrategy,
};

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let semantic_encoder =
       Embedder::from_pretrained_cloud("openai", "text-embedding-3-small", None).unwrap();
    let text_embed_config = TextEmbedConfig::new(
        Some(1000),
        Some(256),
        Some(32),
        None,
        None,
    );
    let cohere_model =
        Embedder::from_pretrained_cloud("cohere", "embed-english-v3.0", None).unwrap();
    let openai_model =
        Embedder::from_pretrained_cloud("openai", "text-embedding-3-small", None).unwrap();
    let openai_model: Arc<Embedder> = Arc::new(openai_model);
    let _openai_embeddings = embed_directory_stream(
        PathBuf::from("test_files"),
        &openai_model,
        Some(vec!["pdf".to_string()]),
        Some(&text_embed_config),
        None::<fn(Vec<EmbedData>)>,
    )
    .await?
    .unwrap();

    let _file_embedding = embed_file(
        "test_files/attention.pdf",
        &openai_model,
        Some(&text_embed_config),
        None::<fn(Vec<EmbedData>)>,
    )
    .await?
    .unwrap();


    let _cohere_embedding = embed_file(
        "test_files/attention.pdf",
        &cohere_model,
        Some(&text_embed_config),
        None::<fn(Vec<EmbedData>)>,
    )
    .await?
    .unwrap();

    println!("Cohere embedding: {:?}", _cohere_embedding);

    Ok(())
}
