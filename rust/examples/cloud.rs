use std::{path::PathBuf, sync::Arc};

use embed_anything::{
    config::TextEmbedConfig,
    embed_directory_stream, embed_file, embed_query,
    embeddings::{
        cloud::cohere::CohereEmbeder,
        embed::{EmbedData, Embeder},
    },
};

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let text_embed_config = TextEmbedConfig::new(Some(256), Some(32), None);
    let cohere_model = Embeder::Cohere(CohereEmbeder::new("embed-english-v3.0".to_string(), None));
    let openai_model =
        Embeder::from_pretrained_cloud("openai", "text-embedding-3-small", Some("sk-XfPIIaC2ytOJ5ybBSqgALVCCu0SRhm40oszJYhmTrzT3BlbkFJUA6eB4mJAfBDJO_dKJTh54SAL9DHNyFtUTHVYbKnYA".to_string())).unwrap();

    let openai_model: Arc<Embeder> = Arc::new(openai_model);
    let _openai_embeddings = embed_directory_stream(
        PathBuf::from("test_files"),
        &openai_model,
        Some(vec!["pdf".to_string()]),
        Some(&text_embed_config),
        None::<fn(Vec<EmbedData>)>,
    )
    .await?
    .unwrap();

    let _cohere_embedding: Option<Vec<EmbedData>> = embed_file(
        "test_files/attention.pdf",
        &cohere_model,
        Some(&text_embed_config),
        None::<fn(Vec<EmbedData>)>,
    )
    .await?;

    Ok(())
}
