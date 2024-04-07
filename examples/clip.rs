use candle_core::{Device, Tensor};
use embed_anything::{embed_directory, embed_query};
use std::path::PathBuf;

fn main() {
    //    let out =  embed_file("test_files/TUe_SOP_AI_2.pdf", "Bert").unwrap();
    let out = embed_directory(PathBuf::from("test_files"), "Clip").unwrap();
    let query_emb_data = embed_query(vec!["Photo of a monkey".to_string()], "Clip").unwrap();
    let n_vectors = out.len();
    let out_embeddings = Tensor::from_vec(
        out.iter()
            .map(|embed| embed.embedding.clone())
            .collect::<Vec<_>>()
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<f32>>(),
        (n_vectors, out[0].embedding.len()),
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
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<f32>>(),
        (1, query_emb_data[0].embedding.len()),
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

    let similar_image =top_3_image_paths[0].clone();

    println!("{:?}", similar_image)
}
