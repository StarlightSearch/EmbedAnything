use candle_core::{Device, Tensor};
use embed_anything::config::TextEmbedConfig;
use embed_anything::embeddings::embed::{EmbedData, Embedder};
use embed_anything::embeddings::local::text_embedding::ONNXModel;
use embed_anything::text_loader::SplittingStrategy;
use embed_anything::{embed_file, embed_query};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model =

        Arc::new(Embedder::from_pretrained_onnx("bert", ONNXModel::AllMiniLML6V2, None).unwrap());
    let config = TextEmbedConfig::new(
        Some(1000),
        Some(256),
        Some(256),
        Some(SplittingStrategy::Sentence),
        Some(model.clone()),
    );

    // get files in bench
    let files = std::fs::read_dir("bench")
        .unwrap()
        .map(|f| f.unwrap().path())
        .collect::<Vec<_>>();

    let now = Instant::now();


    let futures = files
        .par_iter()
        .map(|file| {
            embed_file(file, &model, Some(&config), None::<fn(Vec<EmbedData>)>)
        })
        .collect::<Vec<_>>();

    let _data = futures.into_iter().next().unwrap().await;


    let elapsed_time = now.elapsed();
    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());

    let sentences = [
        "The quick brown fox jumps over the lazy dog",
        "The cat is sleeping on the mat",
        "The dog is barking at the moon",
        "I love pizza",
        "I like to have pasta",
        "The dog is sitting in the park",
        "The window is broken",
        "pizza is the best",

    ]
    .iter()
    .map(|s| s.to_string())
    .collect::<Vec<_>>();

    let doc_embeddings = embed_query(sentences.clone(), &model, Some(&config))
        .await
        .unwrap();
    let n_vectors = doc_embeddings.len();
    let out_embeddings = Tensor::from_vec(
        doc_embeddings
            .iter()
            .map(|embed| embed.embedding.clone())
            .collect::<Vec<_>>()
            .into_iter()
            .map(|x| x.to_dense().unwrap())
            .flatten()
            .collect::<Vec<_>>(),
        (n_vectors, doc_embeddings[0].embedding.to_dense().unwrap().len()),
        &Device::Cpu,
    )
    .unwrap();

    let mut similarities = vec![];
    for i in 0..n_vectors {
        let e_i = out_embeddings.get(i)?;
        for j in (i + 1)..n_vectors {
            let e_j = out_embeddings.get(j)?;
            let sum_ij = (&e_i * &e_j)?.sum_all()?.to_scalar::<f32>()?;
            let sum_i2 = (&e_i * &e_i)?.sum_all()?.to_scalar::<f32>()?;
            let sum_j2 = (&e_j * &e_j)?.sum_all()?.to_scalar::<f32>()?;
            let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
            similarities.push((cosine_similarity, i, j))
        }
    }
    similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    for &(score, i, j) in similarities[..5].iter() {
        println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
    }

    Ok(())

}