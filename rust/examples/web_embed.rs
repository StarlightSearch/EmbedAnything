use std::sync::Arc;

use candle_core::Tensor;
use embed_anything::{
    config::TextEmbedConfig,
    embed_query, embed_webpage,
    embeddings::embed::{EmbedData, Embedder},
    text_loader::SplittingStrategy,
};

#[tokio::main]
async fn main() {
    let start_time = std::time::Instant::now();
    let url = "https://www.scrapingbee.com/blog/web-scraping-rust/".to_string();

    let embedder = Arc::new(
        Embedder::from_pretrained_hf("bert", "sentence-transformers/all-MiniLM-L6-v2", None)
            .unwrap(),
    );

    let embed_config = TextEmbedConfig::default()
        .with_chunk_size(256, Some(0.3))
        .with_batch_size(32)
        .with_buffer_size(100)
        .with_splitting_strategy(SplittingStrategy::Sentence)
        .with_semantic_encoder(Arc::clone(&embedder));

    let embed_data = embed_webpage(
        url,
        &embedder,
        Some(&embed_config),
        None::<fn(Vec<EmbedData>)>,
    )
    .await
    .unwrap()
    .unwrap();
    let embeddings = embed_data
        .iter()
        .map(|data| data.embedding.to_dense().unwrap())
        .collect::<Vec<_>>();

    // Convert embeddings to a tensor
    let embeddings = Tensor::from_vec(
        embeddings.iter().flatten().cloned().collect::<Vec<f32>>(),
        (embeddings.len(), embeddings[0].len()),
        &candle_core::Device::Cpu,
    )
    .unwrap();

    let query = vec!["Rust for web scraping".to_string()];
    let query_embedding: Vec<f32> = embed_query(query, &embedder, Some(&embed_config))
        .await
        .unwrap()
        .iter()
        .map(|data| data.embedding.to_dense().unwrap())
        .flatten()
        .collect();

    let query_embedding_tensor = Tensor::from_vec(
        query_embedding.clone(),
        (1, query_embedding.len()),
        &candle_core::Device::Cpu,
    )
    .unwrap();

    let similarities = embeddings
        .matmul(&query_embedding_tensor.transpose(0, 1).unwrap())
        .unwrap()
        .detach()
        .squeeze(1)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let max_similarity_index = similarities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let data = &embed_data[max_similarity_index].metadata;

    println!("{:?}", data);
    println!("Time taken: {:?}", start_time.elapsed());
}
