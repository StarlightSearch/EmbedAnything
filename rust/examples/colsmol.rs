use clap::{Parser, ValueEnum};

#[cfg(feature = "ort")]
use embed_anything::embeddings::local::colsmol_ort::{OrtColSmolEmbedder, ColSmolEmbed};



#[derive(Parser, Debug, Clone, ValueEnum)]
enum ModelType {
    Ort,
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
            #[cfg(feature = "ort")]
            {
                Box::new(OrtColSmolEmbedder::new(
                    "onnx-community/colSmol-256M-ONNX",
                    None,
                    Some("onnx"),
                )?) as Box<dyn ColSmolEmbed>
            }
            #[cfg(not(feature = "ort"))]
            {
                panic!("ORT is not supported without ORT");
            }
        }
     
    };
    // ... rest of the code ...

    let image_path = "test.jpg";
    let image = image::open(image_path)?;
    let embed_data = colpali_model.embed_image(image_path.into(), None)?;
    println!("{:?}", embed_data);
    // let file_path = "test_files/colpali.pdf";
    // let batch_size = 4;
    // let embed_data = colpali_model.embed_file(file_path.into(), batch_size)?;
    // println!("{:?}", embed_data.len());

    // let prompt = "What is attention?";
    // let query_embeddings = colpali_model.embed_query(prompt)?;
    // println!("{:?}", query_embeddings.len());
    Ok(())
}
