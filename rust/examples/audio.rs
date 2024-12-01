use std::sync::Arc;

use embed_anything::{
    config::TextEmbedConfig, emb_audio, embeddings::embed::Embedder,
    file_processor::audio::audio_processor::AudioDecoderModel, text_loader::SplittingStrategy,
};

#[tokio::main]
async fn main() {
    let audio_path = std::path::PathBuf::from("test_files/audio/samples_hp0.wav");
    let mut audio_decoder = AudioDecoderModel::from_pretrained(
        Some("openai/whisper-tiny.en"),
        Some("main"),
        "tiny-en",
        false,
    )
    .unwrap();

    let bert_model = Arc::new(
        Embedder::from_pretrained_hf("bert", "sentence-transformers/all-MiniLM-L6-v2", None)
            .unwrap(),
    );

    let semantic_encoder = Arc::new(
        Embedder::from_pretrained_hf("jina", "jinaai/jina-embeddings-v2-small-en", None).unwrap(),
    );
    let text_embed_config = TextEmbedConfig::default()
        .with_chunk_size(256, Some(0.3))
        .with_batch_size(32)
        .with_splitting_strategy(SplittingStrategy::Sentence)
        .with_semantic_encoder(Arc::clone(&semantic_encoder));

    let embeddings = emb_audio(
        audio_path,
        &mut audio_decoder,
        &bert_model,
        Some(&text_embed_config),
    )
    .await
    .unwrap()
    .unwrap();

    println!("{:?}", embeddings);
}
