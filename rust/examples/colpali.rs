use clap::{Parser, ValueEnum};
use embed_anything::embeddings::local::{
    colpali::{ColPaliEmbed, ColPaliEmbedder},
    colpali_ort::OrtColPaliEmbedder,
};

#[derive(Parser, Debug, Clone, ValueEnum)]
enum ModelType {
    Ort,
    Normal,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Choose model type: 'ort' or 'normal'
    #[arg(short, long, default_value = "normal")]
    model_type: ModelType,
}

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    let colpali_model = match args.model_type {
        ModelType::Ort => Box::new(OrtColPaliEmbedder::new(
            "akshayballal/colpali-v1.2-merged-onnx",
            None,
        )?) as Box<dyn ColPaliEmbed>,
        ModelType::Normal => Box::new(ColPaliEmbedder::new("vidore/colpali-v1.2-merged", None)?)
            as Box<dyn ColPaliEmbed>,
    };
    // ... rest of the code ...
    let file_path = "test_files/attention.pdf";
    let batch_size = 4;
    let embed_data = colpali_model.embed_file(file_path.into(), batch_size)?;
    println!("{:?}", embed_data.len());

    let prompt = "What is attention?";
    let query_embeddings = colpali_model.embed_query(prompt)?;
    println!("{:?}", query_embeddings.len());
    Ok(())
}
