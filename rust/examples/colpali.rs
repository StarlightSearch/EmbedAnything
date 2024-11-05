use embed_anything::embeddings::{
    embed::{EmbedData, EmbeddingResult},
    local::{
        colpali::{ColPaliEmbed, ColPaliEmbedder},
        colpali_ort::OrtColPaliEmbedder,
    },
};
use half::f16;
fn main() -> Result<(), anyhow::Error> {
    // use onnx model
    let colpali_model = OrtColPaliEmbedder::new("akshayballal/colpali-v1.2-merged-onnx", None)?;

    // let colpali_model = ColPaliEmbedder::new("vidore/colpali-v1.2-merged", None)?;
    let file_path = "test_files/attention.pdf";
    let batch_size = 1;
    let embed_data: Vec<EmbedData<f16>> = colpali_model.embed_file(file_path.into(), batch_size)?;
    println!("{:?}", embed_data.len());

    let prompt = "What is attention?";
    let query_embeddings: Vec<EmbedData<f16>> = colpali_model.embed_query(prompt)?;
    println!("{:?}", query_embeddings.len());
    Ok(())
}
