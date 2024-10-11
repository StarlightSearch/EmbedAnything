use candle_core::{Device, Tensor};
use embed_anything::config::TextEmbedConfig;
use embed_anything::embed_query;
use embed_anything::embeddings::embed::Embeder;
use embed_anything::embeddings::local::text_embedding::ONNXModel;
use embed_anything::text_loader::{SplittingStrategy, TextLoader};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model =
        Arc::new(Embeder::from_pretrained_ort("bert", ONNXModel::BGESmallENV15Q, None).unwrap());
    let config = TextEmbedConfig::new(
        Some(256),
        Some(2),
        Some(2),
        Some(SplittingStrategy::Sentence),
        Some(model.clone()),
    );

    // get files in bench
    let files = std::fs::read_dir("bench")
        .unwrap()
        .map(|f| f.unwrap().path())
        .collect::<Vec<_>>();
    let text_loader = TextLoader::new(1000);

    // for file in files {
    //     let text = TextLoader::extract_text(file.to_str().unwrap()).unwrap();
    //     let chunks = text_loader
    //         .split_into_chunks(&text, SplittingStrategy::Sentence, None)
    //         .unwrap();

    //     let data = embed_query(chunks, &model, Some(&config)).await.unwrap();
    // }
    let now = Instant::now();

    let futures = files
        .par_iter()
        .map(|file| {
            let text = TextLoader::extract_text(file).unwrap();
            let chunks = text_loader
                .split_into_chunks(&text, SplittingStrategy::Sentence, None)
                .unwrap();
            embed_query(chunks, &model, Some(&config))
        })
        .collect::<Vec<_>>();

    let data = futures::future::join_all(futures)
        .await
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let elapsed_time = now.elapsed();
    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());

    let sentences = [
        "The quick brown fox jumps over the lazy dog",
        "The cat is sleeping on the mat",
        "The dog is barking at the moon",
        "I love pizza",
        "I like to have pasta",
        "The dog is sitting in the park",
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
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<f32>>(),
        (n_vectors, doc_embeddings[0].embedding.len()),
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

    // let now = Instant::now();

    // let _out = embed_directory_stream(
    //     PathBuf::from("bench"),
    //     &model,
    //     None,
    //     // Some(vec!["txt".to_string()]),
    //     Some(&config),
    //     None::<fn(Vec<EmbedData>)>,
    // )
    // .await
    // .unwrap()
    // .unwrap();

    // println!("Number of chunks: {:?}", _out.len());
    // let elapsed_time = now.elapsed();
    // println!("Elapsed Time: {}", elapsed_time.as_secs_f32());

    // let now = Instant::now();

    // let _out = embed_file(
    //     "bench/attention.pdf",
    //     &model,
    //     Some(&config),
    //     None::<fn(Vec<EmbedData>)>,
    // )
    // .await
    // .unwrap()
    // .unwrap();

    // let elapsed_time = now.elapsed();
    // println!("Elapsed Time: {}", elapsed_time.as_secs_f32());
}
