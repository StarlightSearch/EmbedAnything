use clap::{Parser, ValueEnum};

use embed_anything::embeddings::local::{colpali::ColPaliEmbed, colsmol::ColSmolEmbedder};

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
        ModelType::Ort => {
            panic!("ORT is not supported without ORT");
        }
        ModelType::Normal => Box::new(ColSmolEmbedder::new(
            "akshayballal/colSmol-256M-merged",
            None,
        )?) as Box<dyn ColPaliEmbed>,
    };
    // ... rest of the code ...

    let file_path = "test_files/colpali.pdf";
    let batch_size = 2;
    let embed_data = colpali_model.embed_file(file_path.into(), batch_size)?;
    println!("{:?}", embed_data.len());

    let prompt = "What is attention?";
    let query_embeddings = colpali_model.embed_query(prompt)?;
    println!("{:?}", query_embeddings.len());
    Ok(())
}
