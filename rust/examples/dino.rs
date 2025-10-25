use candle_core::{Device, Tensor};
use embed_anything::{
    embed_image_directory, embed_query,
    embeddings::embed::{Embedder, EmbedderBuilder},
};
use std::{path::PathBuf, sync::Arc, time::Instant};

#[tokio::main]
async fn main() {
    let now = Instant::now();

    let model = EmbedderBuilder::new()
        .model_id(Some("facebook/dinov2-small"))
        .revision(None)
        .token(None)
        .from_pretrained_hf()
        .unwrap();
    let model: Arc<Embedder> = Arc::new(model);
    let out = embed_image_directory(PathBuf::from("test_files"), &model, None, None)
        .await
        .unwrap()
        .unwrap();
}