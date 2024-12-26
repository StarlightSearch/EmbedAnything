use embed_anything::reranker::jina::Dtype;
fn main() {
    let reranker = embed_anything::reranker::jina::JinaReranker::new(
        "jinaai/jina-reranker-v2-base-multilingual",
        None,
        Dtype::FP16,
    )
    .unwrap();

    
    let sentences = vec![
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
        "The new movie is awesome",
        "The cat plays in the garden",
        "A woman watches TV",
        "The new movie is so great",
        "Do you like pizza?",
    ];
   

    let query = vec!["There is a cat outside"];

    let reranker_results = reranker.rerank(query, sentences, 32).unwrap();
    let pretty_results = serde_json::to_string_pretty(&reranker_results).unwrap();
    println!("{}", pretty_results);
}
