use embed_anything::{emb_audio, embed_file, embedding_model};

fn main() {
    let audio_path = std::path::PathBuf::from("test_files/audio/samples_hp0.wav");

    let embeddings = embed_file(audio_path.to_str().unwrap(), "Whisper-Bert").unwrap();

    println!("{:?}", embeddings);
}
