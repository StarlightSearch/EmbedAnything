use std::sync::Arc;

use embed_anything::{config::ImageEmbedConfig, embed_file, embed_query, embeddings::{embed::{EmbedData, Embedder}, local::colpali::ColPaliEmbedder}};
use embed_anything::embeddings::embed::EmbedImage;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let colpali_model = Arc::new(Embedder::from_pretrained_hf(
        "colpali",
        "vidore/colpali-v1.2-merged",
        None,
    )?);
    let config = ImageEmbedConfig::new(Some(32));
    let image_embeddings = embed_file("/home/akshay/EmbedAnything/test_files/clip/cat1.jpg", &colpali_model, None, None::<fn(Vec<EmbedData>)>).await?;
    let query_embeddings = embed_query(vec!["Hello".to_string()], &colpali_model, None).await?;

    println!("{:?}", image_embeddings);

    Ok(())
}
