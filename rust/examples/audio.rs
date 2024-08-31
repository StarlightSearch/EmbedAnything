use embed_anything::{config, emb_audio, embeddings::embed::Embeder, file_processor::audio::audio_processor::AudioDecoderModel};

fn main() {
    let audio_path = std::path::PathBuf::from("test_files/audio/samples_hp0.wav");
    let mut audio_decoder = AudioDecoderModel::from_pretrained(Some("openai/whisper-tiny.en"), Some("main"), "tiny-en", false).unwrap();

    let bert_model = Embeder::from_pretrained_hf("bert", "sentence-transformers/all-MiniLM-L6-v2", None).unwrap();

    let text_embed_config = config::TextEmbedConfig::new(Some(256), Some(32));  
    let embeddings = emb_audio(audio_path, &mut audio_decoder, &bert_model, Some(&text_embed_config)).unwrap();

    println!("{:?}", embeddings);
}
