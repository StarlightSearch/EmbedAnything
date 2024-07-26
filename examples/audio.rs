use embed_anything::{config, embed_file};

fn main() {
    let audio_path = std::path::PathBuf::from("test_files/audio/samples_hp0.wav");

    let decoder_config = config::AudioDecoderConfig::new(
        Some("openai/whisper-tiny.en".to_string()),
        Some("main".to_string()),
        Some("tiny-en".to_string()),
        Some(false),
    );

    let jina_config = config::JinaConfig::new(
        Some("jinaai/jina-embeddings-v2-small-en".to_string()),
        Some("main".to_string()),
        Some(100),
    );

    let config = config::EmbedConfig {
        audio_decoder: Some(decoder_config),
        jina: Some(jina_config),
        ..Default::default()
    };

    let embeddings = embed_file(audio_path.to_str().unwrap(), "Audio", Some(&config), None).unwrap();

    println!("{:?}", embeddings);
}
