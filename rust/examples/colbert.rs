use embed_anything::config::{SplittingStrategy, TextEmbedConfig};
use embed_anything::embeddings::embed::{Embedder, EmbeddingResult};
use embed_anything::{embed_file, embed_query};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = Arc::new(
        Embedder::from_pretrained_onnx(
            "colbert",
            // Some(ONNXModel::ModernBERTBase),
            None,
            None,
            Some("answerdotai/answerai-colbert-small-v1"),
            None,
            Some("onnx/model_fp16.onnx"),
        )
        .unwrap(),
    );

    let config = TextEmbedConfig::default()
        .with_chunk_size(1000, Some(0.3))
        .with_batch_size(32)
        .with_buffer_size(1000)
        .with_splitting_strategy(SplittingStrategy::Sentence);

    // get files in bench
    let files = std::fs::read_dir("bench")
        .unwrap()
        .map(|f| f.unwrap().path())
        .collect::<Vec<_>>();

    let now = Instant::now();

    let futures = files
        .par_iter()
        .map(|file| embed_file(file, &model, Some(&config), None))
        .collect::<Vec<_>>();

    let _data = futures.into_iter().next().unwrap().await?.unwrap();

    let elapsed_time = now.elapsed();
    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());

    let sentences = [
        "The quick brown fox jumps over the lazy dog",
        "The cat is sleeping on the mat",
        "The dog is barking at the moon",
        "I love pizza",
        "The dog is sitting in the park",
        "Der Hund sitzt im Park", // German for "The dog is sitting in the park"
        "pizza is the best",
        "मैं पिज्जा पसंद करता हूं", // Hindi for "I like pizza"
    ];

    let doc_embeddings = embed_query(&sentences, &model, Some(&config))
        .await
        .unwrap();

    // print out the embeddings for the first sentence
    let EmbeddingResult::MultiVector(vec) = doc_embeddings[0].embedding.clone() else {
        panic!("output should be a multi vector");
    };
    for (i, v) in vec.iter().enumerate() {
        println!("{}: {:?}", i, v);
    }

    Ok(())
}
