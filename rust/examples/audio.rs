use std::sync::Arc;

use embed_anything::{
    config::{SplittingStrategy, TextEmbedConfig},
    emb_audio,
    embeddings::embed::EmbedderBuilder,
    file_processor::audio::audio_processor::AudioDecoderModel,
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
        EmbedderBuilder::new()
            .model_id(Some("sentence-transformers/all-MiniLM-L6-v2"))
            .revision(None)
            .token(None)
            .from_pretrained_hf()
            .unwrap(),
    );

    let text_embed_config = TextEmbedConfig::default()
        .with_chunk_size(1000, Some(0.3))
        .with_batch_size(32)
        .with_splitting_strategy(SplittingStrategy::Sentence);

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
