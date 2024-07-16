use embed_anything::embed_file;

fn main() {
    let audio_path = std::path::PathBuf::from("test_files/audio/samples_hp0.wav");

    let embeddings = embed_file(audio_path.to_str().unwrap(), "Whisper-Bert").unwrap();

    // println!("{:?}", embeddings);
}
