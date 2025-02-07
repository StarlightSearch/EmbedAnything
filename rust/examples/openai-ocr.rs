use std::{os, path::PathBuf};
use std::io::Write;

use base64::Engine;
use embed_anything::file_processor::pdf_processor::get_images_from_pdf;
use image::{GenericImageView, ImageFormat};
use itertools::Itertools;
use ollama_rs::{
    generation::{
        completion::{request::GenerationRequest, GenerationResponse},
        images::Image,
        options::GenerationOptions,
    },
    Ollama,
};
use reqwest::get;
use tokio::runtime::Runtime;
use reqwest::Client;
use serde_json::{json, Value};
use lazy_static::lazy_static;

const IMAGE_URL: &str = "https://images.pexels.com/photos/1054655/pexels-photo-1054655.jpeg";
const PDF_FILE: &str = "/home/akshay/colpali/court.pdf";
const PROMPT: &str = "Convert the provided image into Markdown format. Ensure that all content from the page is included, such as headers, footers, subtexts, images (with alt text if possible), tables, and any other elements.
Requirements:
  - Output Only Markdown: Return solely the Markdown content without any additional explanations or comments.
  - No Delimiters: Do not use code fences or delimiters like ```markdown.
  - Complete Content: Do not omit any part of the page, including headers, footers, and subtext.";

lazy_static! {
    static ref OPENAI_API_KEY: String = std::env::var("OPENAI_API_KEY").unwrap();
}
const MODEL: &str = "gpt-4o-mini";

const BATCH_SIZE: usize = 4;

fn main() {
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        // Download the image and encode it to base64
        let bytes = match download_image(IMAGE_URL).await {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Failed to download image: {}", e);
                return;
            }
        };
        let images = get_images_from_pdf(&PathBuf::from(PDF_FILE)).unwrap();

        for (i, images_chunk) in images.chunks(BATCH_SIZE).enumerate() {
            let images_converted = images_chunk.iter().map(|image| {
                let mut buf = Vec::new();
                let mut cursor = std::io::Cursor::new(&mut buf);
                image.write_to(&mut cursor, ImageFormat::Png).unwrap();
                let engine = base64::engine::general_purpose::STANDARD;
                let base64_image = engine.encode(&buf);
                json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:image/png;base64,{}", base64_image)
                    }
                })
            }).collect_vec();

            let mut request = vec![json!({
                "type": "text",
                "text": PROMPT
            })];
            request.extend(images_converted);

            let response = send_request(request).await.unwrap();
            println!("{}", response);
            
            // Open file in append mode instead of create mode
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("output.md")
                .unwrap();
                
            // Add a separator between pages if it's not the first page
            if i > 0 {
                file.write_all(b"\n\n---\n\n").unwrap();
            }
            file.write_all(response.as_bytes()).unwrap();
        }
    });
}

// Function to download the image
async fn download_image(url: &str) -> Result<Vec<u8>, reqwest::Error> {
    let response = get(url).await?;
    let bytes = response.bytes().await?;
    Ok(bytes.to_vec())
}

// Function to send the request to the model
async fn send_request(
    request: Vec<Value>,
) -> Result<String, Box<dyn std::error::Error>> {
    let client = Client::new();
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", OPENAI_API_KEY.to_string()))
        .json(&json!({
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": request
            }],
            "max_tokens": 2048,
            "temperature": 0.1
        }))
        .send()
        .await?;

    let response: Value = response.json().await?;
    Ok(response["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string())
}
