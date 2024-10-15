use std::sync::Arc;

use embed_anything::{config::ImageEmbedConfig, embed_file, embed_query, embeddings::{embed::{EmbedData, Embedder}, local::colpali::ColPaliEmbedder}};
use embed_anything::embeddings::embed::EmbedImage;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {

    let colpali_model = ColPaliEmbedder::new( "vidore/colpali-v1.2-merged", None)?;
    let file_path = "/home/akshay/projects/EmbedAnything/test_files/attention.pdf";
    let batch_size = 1;
    let embed_data = colpali_model.embed_file(file_path, batch_size)?;
    println!("{:?}", embed_data);

    let prompt = "What is attention?";
    let query_embeddings = colpali_model.embed_query(prompt)?;
    println!("{:?}", query_embeddings);
    Ok(())
    
}
