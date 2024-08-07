use candle_core::Tensor;
use embed_anything::{
    config::{EmbedConfig, JinaConfig},
    embed_query, embed_webpage,
};

fn main() {
    let start_time = std::time::Instant::now();
    let url = "https://www.scrapingbee.com/blog/web-scraping-rust/".to_string();
    let embeder = "Jina".to_string();

    let jina_config = JinaConfig {
        model_id: Some("jinaai/jina-embeddings-v2-base-en".to_string()),
        revision: None,
        chunk_size: Some(1000),
        batch_size: Some(32),
    };

    let embed_config = EmbedConfig {
        jina: Some(jina_config),
        ..Default::default()
    };

    let embed_data = embed_webpage(url, &embeder, Some(&embed_config), None)
        .unwrap()
        .unwrap();
    let embeddings: Vec<Vec<f32>> = embed_data
        .iter()
        .map(|data| data.embedding.clone())
        .collect();

    // Convert embeddings to a tensor
    let embeddings = Tensor::from_vec(
        embeddings.iter().flatten().cloned().collect::<Vec<f32>>(),
        (embeddings.len(), embeddings[0].len()),
        &candle_core::Device::Cpu,
    )
    .unwrap();

    let query = vec!["Rust for web scraping".to_string()];
    let query_embedding: Vec<f32> = embed_query(query, &embeder, Some(&embed_config))
        .unwrap()
        .iter()
        .map(|data| data.embedding.clone())
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
