use embed_anything::embeddings::local::colpali::{ColPaliEmbed, ColPaliEmbedder};

fn main() -> Result<(), anyhow::Error> {
    let colpali_model = ColPaliEmbedder::new("vidore/colpali-v1.2-merged", None)?;
    let file_path = "test_files/attention.pdf";
    let batch_size = 1;
    let embed_data = colpali_model.embed_file(file_path.into(), batch_size)?;
    println!("{:?}", embed_data.len());

    let prompt = "What is attention?";
    let query_embeddings = colpali_model.embed_query(prompt)?;
    println!("{:?}", query_embeddings.len());
    Ok(())
}
