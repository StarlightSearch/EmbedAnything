use embed_anything::config::TextEmbedConfig;
use embed_anything::embeddings::embed::{EmbedData, Embeder};
use embed_anything::embed_directory_stream;
use std::sync::Arc;
use std::{path::PathBuf, time::Instant};

#[tokio::main]
async fn main() {

    // let model = Embeder::from_pretrained_hf("jina", "jinaai/jina-embeddings-v2-small-en", None)
    let model = Embeder::from_pretrained_hf("jina","jinaai/jina-embeddings-v2-small-en", None)
        .unwrap();
    let config = TextEmbedConfig::new(Some(512), Some(12));

    // let out = embed_file(
    //     "test_files/test.pdf",
    //     &model,
    //     None,
    //     None::<fn(Vec<EmbedData>)>,
    // )
    // .unwrap()
    // .unwrap();

    let model: Arc<Embeder> = Arc::new(model);
    let now = Instant::now();

    let _out = embed_directory_stream(
        PathBuf::from("test_files"),
        &model,
        None,
        // Some(vec!["txt".to_string()]),
        Some(&config),
        None::<fn(Vec<EmbedData>)>,
    )
    .await
    .unwrap()
    .unwrap();

    
    println!("Number of chunks: {:?}", _out.len());
    let elapsed_time = now.elapsed();
    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());

    // println!("\nEmbedded Text from PDF:");


    // println!("\nEmbedded Text from PDF:");
    // for (index, embed_data) in _out.iter().enumerate() {
    //     println!("Chunk {}:", index + 1);
    //     println!("Text: {}", embed_data.text.clone().unwrap());
    //     println!("Embedding length: {}", embed_data.embedding.len());
    //     println!("Metadata: {:?}", embed_data.metadata);
    //     println!("---");
    // }


}
