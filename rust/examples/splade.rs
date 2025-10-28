use clap::{Parser, ValueEnum};

use candle_core::{Device, Tensor};
use embed_anything::{
    config::{SplittingStrategy, TextEmbedConfig},
    embed_query,
    embeddings::{
        embed::{Embedder, EmbedderBuilder},
        local::text_embedding::ONNXModel,
    },
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
        ModelType::Ort => Arc::new(
            Embedder::from_pretrained_onnx(
                "sparse-bert",
                Some(ONNXModel::SPLADEPPENV2),
                None,
                None,
                None,
                None,
            )
            .unwrap(),
        ),
        ModelType::Normal => Arc::new(
            EmbedderBuilder::new()
                .model_id(Some("prithivida/Splade_PP_en_v1"))
                .revision(None)
                .from_pretrained_hf()
                .unwrap(),
        ),
    };

    let config = TextEmbedConfig::default()
        .with_chunk_size(1000, Some(0.3))
        .with_batch_size(32)
        .with_buffer_size(100)
        .with_splitting_strategy(SplittingStrategy::Sentence);

    let sentences = [
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
        "The new movie is awesome",
        "The cat plays in the garden",
        "A woman watches TV",
        "The new movie is so great",
        "Do you like pizza?",
    ];

    let n_sentences = sentences.len();

    let out = embed_query(&sentences, &model, Some(&config))
        .await
        .unwrap();

    let embeddings = out
        .iter()
        .flat_map(|embed| embed.embedding.to_dense().unwrap())
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
    similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    for &(score, i, j) in similarities[..5].iter() {
        println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
    }

    Ok(())
}
