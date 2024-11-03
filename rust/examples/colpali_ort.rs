use embed_anything::embeddings::local::colpali_ort::OrtColPaliEmbedder;
use embed_anything::embeddings::local::colpali::ColPaliEmbed;

fn main() -> anyhow::Result<()> {
    let model = OrtColPaliEmbedder::new("akshayballal/colpali-v1.2-merged-onnx", None, 128)?;
    let embeddings = model.embed(&["Hello, world!".to_string()], None)?;
    println!("{:?}", embeddings);
    Ok(())
}
