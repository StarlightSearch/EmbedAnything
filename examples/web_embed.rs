use embed_anything::file_processor::website_processor;
use candle_core::Tensor;

#[tokio::main]
async fn main() {
    let url = "https://en.wikipedia.org/wiki/Long_short-term_memory";

    let website_processor = website_processor::WebsiteProcesor;
    let webpage = website_processor.process_website(url).await.unwrap();
    let embeder = embed_anything::embedding_model::bert::BertEmbeder::default();
    let embed_data = webpage.embed_webpage(&embeder).unwrap();
    let embeddings: Vec<Vec<f32>> = embed_data.iter().map(|data| data.embedding.clone()).collect();

    let embeddings = Tensor::from_vec(
        embeddings.iter().flatten().cloned().collect::<Vec<f32>>(),
        (embeddings.len(), embeddings[0].len()),
        &candle_core::Device::Cpu,
    ).unwrap();

    let query = vec!["how to use lstm for nlp".to_string()];
    let query_embedding: Vec<f32> = embeder.embed(&query, None).unwrap().iter().map(|data| data.embedding.clone()).flatten().collect();
    
    let query_embedding_tensor = Tensor::from_vec(
        query_embedding.clone(),
        (1, query_embedding.len()),
        &candle_core::Device::Cpu,
    ).unwrap();


    let similarities = embeddings
        .matmul(&query_embedding_tensor.transpose(0, 1).unwrap())
        .unwrap()
        .detach()
        .squeeze(1)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let max_similarity_index = similarities.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let data = &embed_data[max_similarity_index];

    println!("{:?}", data);
    
}