use std::path::PathBuf;

use embed_anything::{
    config::{CloudConfig, EmbedConfig},
    embed_directory,
};

fn main() {
    // Embedding with Cohere
    let cloud_config = CloudConfig {
        provider: Some("Cohere".to_string()),
        model: Some("embed-english-v3.0".to_string()),
        api_key: None,
        chunk_size: Some(1000),
    };

    let embed_config = EmbedConfig {
        cloud: Some(cloud_config),
        ..Default::default()
    };

    let _out = embed_directory(
        PathBuf::from("test_files"),
        "Cloud",
        Some(vec!["pdf".to_string()]),
        Some(&embed_config),
        None,
    )
    .unwrap()
    .unwrap();

    // Embedding with OpenAI
    let cloud_config = CloudConfig {
        provider: Some("OpenAI".to_string()),
        model: Some("text-embedding-3-small".to_string()),
        api_key: None,
        chunk_size: Some(1000),
    };

    let embed_config = EmbedConfig {
        cloud: Some(cloud_config),
        ..Default::default()
    };

    let _out = embed_directory(
        PathBuf::from("test_files"),
        "Cloud",
        Some(vec!["pdf".to_string()]),
        Some(&embed_config),
        None,
    )
    .unwrap()
    .unwrap();

    println!("{:#?}", _out[0].text.as_ref().unwrap());
}
