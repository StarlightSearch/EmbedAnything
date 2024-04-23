use embed_anything::embed_directory;
use std::{path::PathBuf, time::Instant};

fn main() {
    let now = Instant::now();
    let out = embed_directory(
        PathBuf::from("test_files"),
        "Bert",
        Some(vec!["pdf".to_string()]),
    )
    .unwrap();
    println!("{:?}", out);
    let elapsed_time = now.elapsed();
    println!("Elapsed Time: {}", elapsed_time.as_secs_f32());
}
