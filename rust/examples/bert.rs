use embed_anything::config::TextEmbedConfig;
use embed_anything::embeddings::embed::{EmbedData, Embeder};
use embed_anything::{embed_directory, embed_file};
use std::{path::PathBuf, time::Instant};

fn main() {
    let now = Instant::now();

    let model = Embeder::from_pretrained_hf("bert", "sentence-transformers/all-MiniLM-L6-v2", None)
        .unwrap();
    let config = TextEmbedConfig::new(Some(256), Some(32));
    let _out = embed_directory(
        PathBuf::from("test_files"),
        &model,
        None,
        // Some(vec!["pdf".to_string()]),
        Some(&config),
        None::<fn(Vec<EmbedData>)>,
    )
    .unwrap().unwrap();

    println!("{:?}", _out.len() );

    let out = embed_file(
        "test_files/test.pdf",
        &model,
        None,
        None::<fn(Vec<EmbedData>)>,
    )
    .unwrap().unwrap();

    println!("\nEmbedded Text from PDF:");
    for (index, embed_data) in out.iter().enumerate() {
        println!("Chunk {}:", index + 1);
        println!("Text: {}", embed_data.text.clone().unwrap());
        println!("Embedding length: {}", embed_data.embedding.len());
        println!("Metadata: {:?}", embed_data.metadata);
        println!("---");
    }

    let elapsed_time = now.elapsed();
    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());
}
