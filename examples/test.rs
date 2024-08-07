use embed_anything::{embedding_model::jina::JinaEmbeder, text_loader::TextLoader};

fn main() {
    let file = "test_files/attention.pdf";
    let text = TextLoader::extract_text(file).unwrap();

    let chunks = TextLoader::split_into_chunks(&text, 200).unwrap();

    let embeder = JinaEmbeder::new("jinaai/jina-embeddings-v2-base-en".to_string(), None).unwrap();

    let embeddings = embeder.embed(&chunks).unwrap();
    println!("{}", embeddings.len());
}