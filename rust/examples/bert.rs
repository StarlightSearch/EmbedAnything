use candle_core::{Device, Tensor};
use embed_anything::config::TextEmbedConfig;
use embed_anything::embeddings::embed::{EmbedData, EmbedderBuilder};
use embed_anything::file_processor::docx_processor::DocxProcessor;
use embed_anything::text_loader::SplittingStrategy;
use embed_anything::{embed_directory_stream, embed_file, embed_query, Dtype};
use std::collections::HashSet;
use std::sync::Arc;
use std::{path::PathBuf, time::Instant};

#[tokio::main]
async fn main() {
    let model = Arc::new(
        EmbedderBuilder::new()
            .model_architecture("bert")
            .model_id(Some("sentence-transformers/all-MiniLM-L6-v2"))
            .revision(None)
            .token(None)
            .dtype(Some(Dtype::F16))
            .from_pretrained_hf()
            .unwrap(),
    );

    // let config = TextEmbedConfig::default()
    //     .with_chunk_size(256, Some(0.3))
    //     .with_batch_size(32)
    //     .with_buffer_size(32)
    //     .with_splitting_strategy(SplittingStrategy::Sentence)
    //     .with_semantic_encoder(Some(Arc::clone(&model)));

    // DocxProcessor::extract_text(&PathBuf::from("test_files/test.docx")).unwrap();
    // let now = Instant::now();

    // let _out = embed_file(
    //     "test_files/test.pdf",
    //     &model,
    //     Some(&config),
    //     None::<fn(Vec<EmbedData>)>,
    // )
    // .await
    // .unwrap()
    // .unwrap();

    // let elapsed_time: std::time::Duration = now.elapsed();

    // println!("Elapsed Time: {}", elapsed_time.as_secs_f32());

    // let now = Instant::now();

    // let _out = embed_directory_stream(
    //     PathBuf::from("test_files"),
    //     &model,
    //     None,
    //     // Some(vec!["txt".to_string()]),
    //     Some(&config),
    //     None::<fn(Vec<EmbedData>)>,
    // )
    // .await
    // .unwrap()
    // .unwrap();

    // let embedded_files = _out
    //     .iter()
    //     .map(|e| {
    //         e.metadata
    //             .as_ref()
    //             .unwrap()
    //             .get("file_name")
    //             .unwrap()
    //             .clone()
    //     })
    //     .collect::<Vec<_>>();
    // let mut embedded_files_set = HashSet::new();
    // embedded_files_set.extend(embedded_files);
    // println!("Embedded files: {:?}", embedded_files_set);

    // println!("Number of chunks: {:?}", _out.len());
    // let elapsed_time: std::time::Duration = now.elapsed();
    // println!("Elapsed Time: {}", elapsed_time.as_secs_f32());

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

    let embeddings = embed_query(&sentences, &model, None).await.unwrap();
    let embeddings = embeddings
        .iter()
        .map(|e| e.embedding.clone().to_dense().unwrap())
        .collect::<Vec<_>>();
    let embeddings = Tensor::from_vec(
        embeddings.clone().into_iter().flatten().collect::<Vec<_>>(),
        vec![embeddings.len(), embeddings[0].len()],
        &Device::Cpu,
    )
    .unwrap();
    let mut similarities = vec![];
    let n_sentences = embeddings.dim(0).unwrap();
    for i in 0..n_sentences {
        let e_i = embeddings.get(i).unwrap();
        for j in (i + 1)..n_sentences {
            let e_j = embeddings.get(j).unwrap();
            let sum_ij = (&e_i * &e_j)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let sum_i2 = (&e_i * &e_i)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let sum_j2 = (&e_j * &e_j)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
            similarities.push((cosine_similarity, i, j))
        }
    }

    similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    for &(score, i, j) in similarities[..5].iter() {
        println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
    }
}
