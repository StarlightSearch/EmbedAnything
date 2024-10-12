use candle_core::{Device, Tensor};
use embed_anything::{
    embed_image_directory, embed_query,
    embeddings::embed::{EmbedData, Embedder},
};
use std::{path::PathBuf, sync::Arc, time::Instant};

#[tokio::main]
async fn main() {
    let now = Instant::now();

    let model = Embedder::from_pretrained_hf("clip", "openai/clip-vit-base-patch32", None).unwrap();
    let model: Arc<Embedder> = Arc::new(model);
    let out = embed_image_directory(
        PathBuf::from("test_files"),
        &model,
        None,
        None::<fn(Vec<EmbedData>)>,
    )
    .await
    .unwrap()
    .unwrap();

    let query_emb_data = embed_query(vec!["Photo of a monkey".to_string()], &model, None)
        .await
        .unwrap();
    let n_vectors = out.len();

    let vector = out
        .iter()
        .map(|embed| embed.embedding.clone())
        .collect::<Vec<_>>()
        .into_iter()
        .map(|x| x.to_dense().unwrap())
        .flatten()
        .collect::<Vec<_>>();

    let out_embeddings = Tensor::from_vec(
        vector,
        (n_vectors, out[0].embedding.to_dense().unwrap().len()),
        &Device::Cpu,
    )
    .unwrap();

    let image_paths = out
        .iter()
        .map(|embed| embed.text.clone().unwrap())
        .collect::<Vec<_>>();

    let query_embeddings = Tensor::from_vec(
        query_emb_data
            .iter()
            .map(|embed| embed.embedding.clone())
            .collect::<Vec<_>>()
            .into_iter()
            .map(|x| x.to_dense().unwrap())
            .flatten()
            .collect::<Vec<_>>(),
        (1, query_emb_data[0].embedding.to_dense().unwrap().len()),
        &Device::Cpu,
    )
    .unwrap();

    let similarities = out_embeddings
        .matmul(&query_embeddings.transpose(0, 1).unwrap())
        .unwrap()
        .detach()
        .squeeze(1)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut indices: Vec<usize> = (0..similarities.len()).collect();
    indices.sort_by(|a, b| similarities[*b].partial_cmp(&similarities[*a]).unwrap());

    let top_3_indices = indices[0..3].to_vec();
    let top_3_image_paths = top_3_indices
        .iter()
        .map(|i| image_paths[*i].clone())
        .collect::<Vec<String>>();

    let similar_image = top_3_image_paths[0].clone();

    println!("{:?}", similar_image);

    let elapsed_time = now.elapsed();
    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());
}
