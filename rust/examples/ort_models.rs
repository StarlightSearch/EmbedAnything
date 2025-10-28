use candle_core::{Device, Tensor};
use embed_anything::config::{SplittingStrategy, TextEmbedConfig};
use embed_anything::embeddings::embed::EmbedderBuilder;
use embed_anything::embeddings::local::text_embedding::ONNXModel;
use embed_anything::{embed_file, embed_query, Dtype};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = Arc::new(
        EmbedderBuilder::new()
            .model_architecture("bert")
            .onnx_model_id(Some(ONNXModel::ModernBERTBase))
            .dtype(Some(Dtype::F16))
            .from_pretrained_onnx()
            .unwrap(),
    );

    let config = TextEmbedConfig::default()
        .with_chunk_size(1000, Some(0.3))
        .with_batch_size(32)
        .with_buffer_size(256)
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

    for chunk in _data {
        println!("--------------------------------");

        println!("{:?}", chunk.text.unwrap());
        println!("\n");
    }

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
    let n_vectors = doc_embeddings.len();
    let out_embeddings = Tensor::from_vec(
        doc_embeddings
            .iter()
            .map(|embed| embed.embedding.clone())
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|x| x.to_dense().unwrap())
            .collect::<Vec<_>>(),
        (
            n_vectors,
            doc_embeddings[0].embedding.to_dense().unwrap().len(),
        ),
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
