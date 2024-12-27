use clap::{Parser, ValueEnum};

use candle_core::{Device, Tensor};
use embed_anything::{
    config::TextEmbedConfig,
    embed_query,
    embeddings::{embed::{Embedder, TextEmbedder}, local::text_embedding::ONNXModel},
    text_loader::SplittingStrategy,
};
use std::sync::Arc;

#[derive(Parser, Debug, Clone, ValueEnum)]
enum ModelType {
    Ort,
    Normal,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Choose model type: 'ort' or 'normal'
    #[arg(short, long, default_value = "normal")]
    model_type: ModelType,
}


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let model = match args.model_type {
        ModelType::Ort => Arc::new(Embedder::from_pretrained_onnx("sparse-bert", ONNXModel::SPLADEPPENV2, None).unwrap()),
        ModelType::Normal => Arc::new(Embedder::Text(
            TextEmbedder::from_pretrained_hf("sparse-bert", "prithivida/Splade_PP_en_v1", None)
                .unwrap(),
        )),
    };

    let config = TextEmbedConfig::default()
        .with_chunk_size(256, Some(0.3))
        .with_batch_size(32)
        .with_buffer_size(100)
        .with_splitting_strategy(SplittingStrategy::Sentence)
        .with_semantic_encoder(Arc::clone(&model));



    let sentences = [
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
        "The new movie is awesome",
        "The cat plays in the garden",
        "A woman watches TV",
        "The new movie is so great",
        "Do you like pizza?",
    ]
    .iter()
    .map(|x| x.to_string())
    .collect::<Vec<_>>();

    let n_sentences = sentences.len();

    let out = embed_query(sentences.clone(), &model, Some(&config))
        .await
        .unwrap();

    let embeddings = out
        .iter()
        .map(|embed| embed.embedding.to_dense().unwrap())
        .flatten()
        .collect::<Vec<_>>();


    let embeddings_tensor = Tensor::from_vec(
        embeddings.clone(),
        (n_sentences, out[0].embedding.to_dense().unwrap().len()),
        &Device::Cpu,
    )
    .unwrap();

    let mut similarities = vec![];
    for i in 0..n_sentences {
        let e_i = embeddings_tensor.get(i).unwrap();
        for j in (i + 1)..n_sentences {
            let e_j = embeddings_tensor.get(j).unwrap();
            let sum_ij = (&e_i * &e_j)?.sum_all()?.to_scalar::<f32>()?;
            let sum_i2 = (&e_i * &e_i)?.sum_all()?.to_scalar::<f32>()?;
            let sum_j2 = (&e_j * &e_j)?.sum_all()?.to_scalar::<f32>()?;
            let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
            similarities.push((cosine_similarity, i, j))
        }
    }
    println!("similarities: {:?}", similarities);
    similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    for &(score, i, j) in similarities[..5].iter() {
        println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
    }

    Ok(())
}
