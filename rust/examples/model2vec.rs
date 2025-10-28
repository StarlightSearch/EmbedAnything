use embed_anything::config::{SplittingStrategy, TextEmbedConfig};
use embed_anything::embeddings::embed::EmbedderBuilder;
use embed_anything::Dtype;
use std::collections::HashSet;
use std::sync::Arc;
use std::{path::PathBuf, time::Instant};

#[tokio::main]
async fn main() {
    let model = Arc::new(
        EmbedderBuilder::new()
            .model_id(Some("minishlab/potion-base-8M"))
            .revision(None)
            .token(None)
            .dtype(Some(Dtype::F16))
            .from_pretrained_hf()
            .unwrap(),
    );

    let config = TextEmbedConfig::default()
        .with_chunk_size(1000, Some(0.3))
        .with_batch_size(32)
        .with_buffer_size(32)
        .with_splitting_strategy(SplittingStrategy::Semantic {
            semantic_encoder: model.clone(),
        })
        .with_pdf_backend("lopdf");

    let now = Instant::now();

    // Embed files batch
    let _out_2 = model
        .embed_files_batch(
            vec!["test_files/test.pdf", "test_files/test.txt"],
            Some(&config),
            None,
        )
        .await
        .unwrap()
        .unwrap();

    // Embed file
    let _out = model
        .embed_file("test_files/test.pdf", Some(&config), None)
        .await
        .unwrap()
        .unwrap();

    let elapsed_time: std::time::Duration = now.elapsed();

    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());

    let now = Instant::now();

    // Embed a directory
    let _out = model
        .embed_directory_stream(
            PathBuf::from("test_files"),
            Some(vec!["pdf".to_string(), "txt".to_string()]),
            Some(&config),
            None,
        )
        .await
        .unwrap()
        .unwrap();

    // Embed an html file
    let _out2 = model
        .embed_webpage("https://www.google.com".to_string(), Some(&config), None)
        .await
        .unwrap()
        .unwrap();

    let embedded_files = _out
        .iter()
        .map(|e| {
            e.metadata
                .as_ref()
                .unwrap()
                .get("file_name")
                .unwrap()
                .clone()
        })
        .collect::<Vec<_>>();
    let mut embedded_files_set = HashSet::new();
    embedded_files_set.extend(embedded_files);
    println!("Embedded files: {:?}", embedded_files_set);

    println!("Number of chunks: {:?}", _out.len());
    let elapsed_time: std::time::Duration = now.elapsed();
    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());
}
