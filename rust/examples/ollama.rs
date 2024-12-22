use std::path::PathBuf;
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

const IMAGE_URL: &str = "https://images.pexels.com/photos/1054655/pexels-photo-1054655.jpeg";
const PDF_FILE: &str = "test_files/attention.pdf";
const PROMPT: &str = "OCR all text from this document word by word exactly as it is in markdown format.";

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

        for (i, images_chunk) in images.chunks(1).enumerate() {
            let images_converted = images_chunk.iter().map(|image| {
                let mut buf = Vec::new();
                let mut cursor = std::io::Cursor::new(&mut buf);
                image.write_to(&mut cursor, ImageFormat::Png).unwrap();
                let engine = base64::engine::general_purpose::STANDARD;
                let base64_image = engine.encode(&buf);
                Image::from_base64(&base64_image)
            }).collect_vec();

            let request = GenerationRequest::new("minicpm-v:latest".to_string(), PROMPT.to_string())
                .add_image(images_converted[0].clone())
                // .add_image(images_converted[1].clone())
                .options(GenerationOptions::default().num_predict(-1));

            let response = send_request(request).await.unwrap();
            println!("{}", response.response);
            let mut file = std::fs::File::create(format!("output_{}.md", i)).unwrap();
            file.write_all(response.response.as_bytes()).unwrap();
        }

        // let image = images[5].clone();
        // let image = images[5].clone();

        // let mut buf = Vec::new();
        // let mut cursor = std::io::Cursor::new(&mut buf);
        // image.write_to(&mut cursor, ImageFormat::Png).unwrap();
        // let engine = base64::engine::general_purpose::STANDARD;
        // let base64_image = engine.encode(&buf);

        // let image = Image::from_base64(&base64_image);

        // // Create a GenerationRequest with the model and prompt, adding the image
        // let request = GenerationRequest::new("minicpm-v:latest".to_string(), PROMPT.to_string())
        //     .add_image(image)
        //     .options(GenerationOptions::default().num_predict(-1));
        // // Send the request to the model and get the response
        // let response = match send_request(request).await {
        //     Ok(r) => r,
        //     Err(e) => {
        //         eprintln!("Failed to get response: {}", e);
        //         return;
        //     }
        // };
        // // println!("{:?}", response);

        // // Print the response
        // println!("{}", response.response);

        // save as markdown file
        // let mut file = std::fs::File::create("output.md").unwrap();
        // file.write_all(response.response.as_bytes()).unwrap();
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
    request: GenerationRequest,
) -> Result<GenerationResponse, Box<dyn std::error::Error>> {
    let ollama = Ollama::default();
    let response = ollama.generate(request).await?;
    Ok(response)
}
