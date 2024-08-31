use embed_anything::{config::{self}, emb_audio, embeddings::embed::Embeder, file_processor::audio::audio_processor::AudioDecoderModel};

fn main() {
    let audio_path = std::path::PathBuf::from("test_files/audio/samples_hp0.wav");
    let audio_decoder = AudioDecoderModel::from_pretrained("whisper", Some("openai/whisper-tiny.en"), Some("main"), "tiny-en").unwrap();

    let bert_model = Embeder::from_pretrained("bert", "sentence-transformers/all-MiniLM-L6-v2", None).unwrap();

    let text_embed_config = config::TextEmbedConfig::new(Some(256), Some(32));  
    let embeddings = emb_audio(audio_path, audio_decoder, &bert_model, Some(&text_embed_config)).unwrap();

    println!("{:?}", embeddings);
}
