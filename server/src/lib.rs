use std::collections::HashMap;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use actix_multipart::Multipart;
use actix_web::dev::Server;
use actix_web::{get, post, web, App, HttpResponse, HttpServer};
use embed_anything::config::TextEmbedConfig;
use embed_anything::{
    embed_files_batch, embed_query,
    embeddings::embed::{EmbedData, EmbedderBuilder, EmbeddingResult},
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

// OpenAI-compatible request structure
#[derive(Deserialize)]
struct OpenAIEmbedRequest {
    model: String,
    input: Vec<String>,
}

// OpenAI-compatible response structures
#[derive(Serialize)]
struct OpenAIEmbedResponse {
    object: String,
    data: Vec<EmbeddingData>,
    model: String,
    usage: Usage,
}

#[derive(Serialize)]
struct EmbeddingData {
    object: String,
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: Option<String>,
}

#[derive(Deserialize)]
struct PdfBatchRequest {
    model: String,
    files: Vec<String>,
}

#[derive(Serialize)]
struct PdfEmbeddingResponse {
    object: String,
    data: Vec<PdfEmbeddingItem>,
    model: String,
}

#[derive(Serialize)]
struct PdfEmbeddingItem {
    object: String,
    index: usize,
    embedding: PdfEmbeddingVector,
    metadata: Option<HashMap<String, String>>,
    text: Option<String>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum PdfEmbeddingVector {
    Dense(Vec<f32>),
    Multi(Vec<Vec<f32>>),
}

fn pdf_embedding_response(model: String, embeddings: Vec<EmbedData>) -> PdfEmbeddingResponse {
    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embed_data)| {
            let embedding_vector = match embed_data.embedding {
                EmbeddingResult::DenseVector(vec) => PdfEmbeddingVector::Dense(vec),
                EmbeddingResult::MultiVector(vecs) => PdfEmbeddingVector::Multi(vecs),
            };

            PdfEmbeddingItem {
                object: "embedding".to_string(),
                index,
                embedding: embedding_vector,
                metadata: embed_data.metadata,
                text: embed_data.text,
            }
        })
        .collect();

    PdfEmbeddingResponse {
        object: "list".to_string(),
        data,
        model,
    }
}

#[get("/health_check")]
async fn health_check() -> HttpResponse {
    HttpResponse::Ok().finish()
}

#[post("/v1/embeddings")]
async fn create_embeddings(req: web::Json<OpenAIEmbedRequest>) -> HttpResponse {
    // Validate input
    if req.input.is_empty() {
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: ErrorDetail {
                message: "Input cannot be empty".to_string(),
                error_type: "invalid_request_error".to_string(),
                code: Some("empty_input".to_string()),
            },
        });
    }

    // Create embedder
    let embedder = match EmbedderBuilder::new()
        .model_id(Some(req.model.as_str()))
        .from_pretrained_hf()
    {
        Ok(embedder) => embedder,
        Err(e) => {
            return HttpResponse::InternalServerError().json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("Failed to initialize embedder: {}", e),
                    error_type: "server_error".to_string(),
                    code: Some("embedder_init_failed".to_string()),
                },
            });
        }
    };

    // Convert input to string slices
    let input_slices: Vec<&str> = req.input.iter().map(|s| s.as_str()).collect();

    // Generate embeddings
    let config = TextEmbedConfig::default();
    let embeddings = match embed_query(&input_slices, &embedder, Some(&config)).await {
        Ok(embeddings) => embeddings,
        Err(e) => {
            return HttpResponse::InternalServerError().json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("Failed to generate embeddings: {}", e),
                    error_type: "server_error".to_string(),
                    code: Some("embedding_generation_failed".to_string()),
                },
            });
        }
    };

    // Convert to OpenAI format
    let embedding_data: Vec<EmbeddingData> = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embed_data)| {
            let embedding_vector = match embed_data.embedding {
                embed_anything::embeddings::embed::EmbeddingResult::DenseVector(vec) => vec,
                embed_anything::embeddings::embed::EmbeddingResult::MultiVector(_) => {
                    // For multi-vector embeddings, we'll flatten them (this might need adjustment based on requirements)
                    vec![0.0] // Placeholder - you might want to handle this differently
                }
            };

            EmbeddingData {
                object: "embedding".to_string(),
                index,
                embedding: embedding_vector,
            }
        })
        .collect();

    // Calculate usage (simplified - you might want to implement proper token counting)
    let total_tokens = req.input.iter().map(|s| s.split_whitespace().count()).sum();

    let response = OpenAIEmbedResponse {
        object: "list".to_string(),
        data: embedding_data,
        model: req.model.clone(),
        usage: Usage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };

    HttpResponse::Ok().json(response)
}

#[post("/v1/pdf_embeddings")]
async fn create_pdf_embeddings(req: web::Json<PdfBatchRequest>) -> HttpResponse {
    if req.files.is_empty() {
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: ErrorDetail {
                message: "File list cannot be empty".to_string(),
                error_type: "invalid_request_error".to_string(),
                code: Some("empty_files".to_string()),
            },
        });
    }

    let mut file_paths = Vec::with_capacity(req.files.len());
    for file in &req.files {
        let path = PathBuf::from(file);
        let extension_is_pdf = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("pdf"))
            .unwrap_or(false);

        if !extension_is_pdf {
            return HttpResponse::BadRequest().json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("Unsupported file type for '{}'. Expected a PDF.", file),
                    error_type: "invalid_request_error".to_string(),
                    code: Some("invalid_file_type".to_string()),
                },
            });
        }

        if !path.exists() {
            return HttpResponse::BadRequest().json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("File '{}' does not exist", file),
                    error_type: "invalid_request_error".to_string(),
                    code: Some("file_not_found".to_string()),
                },
            });
        }

        if !path.is_file() {
            return HttpResponse::BadRequest().json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("'{}' is not a file", file),
                    error_type: "invalid_request_error".to_string(),
                    code: Some("invalid_path".to_string()),
                },
            });
        }

        file_paths.push(path);
    }

    let embedder = match EmbedderBuilder::new()
        .model_id(Some(req.model.as_str()))
        .from_pretrained_hf()
    {
        Ok(embedder) => Arc::new(embedder),
        Err(e) => {
            return HttpResponse::InternalServerError().json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("Failed to initialize embedder: {}", e),
                    error_type: "server_error".to_string(),
                    code: Some("embedder_init_failed".to_string()),
                },
            })
        }
    };

    let config = TextEmbedConfig::default();

    let embeddings = match embed_files_batch(file_paths, &embedder, Some(&config), None).await {
        Ok(Some(embeddings)) => embeddings,
        Ok(None) => Vec::new(),
        Err(e) => {
            return HttpResponse::InternalServerError().json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("Failed to generate embeddings: {}", e),
                    error_type: "server_error".to_string(),
                    code: Some("embedding_generation_failed".to_string()),
                },
            })
        }
    };

    let response = pdf_embedding_response(req.model.clone(), embeddings);

    HttpResponse::Ok().json(response)
}

#[post("/v1/pdf_embeddings/upload")]
async fn create_pdf_embeddings_upload(mut payload: Multipart) -> HttpResponse {
    let temp_dir = match tempfile::tempdir() {
        Ok(dir) => dir,
        Err(e) => {
            return HttpResponse::InternalServerError().json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("Failed to create temp directory: {}", e),
                    error_type: "server_error".to_string(),
                    code: Some("temp_dir_creation_failed".to_string()),
                },
            })
        }
    };
    

    let mut model: Option<String> = None;
    let mut file_paths: Vec<PathBuf> = Vec::new();

    while let Some(item) = payload.next().await {
        let mut field = match item {
            Ok(field) => field,
            Err(e) => {
                return HttpResponse::BadRequest().json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Invalid multipart payload: {}", e),
                        error_type: "invalid_request_error".to_string(),
                        code: Some("invalid_multipart".to_string()),
                    },
                })
            }
        };

        let field_name = field
            .content_disposition()
            .and_then(|cd| cd.get_name())
            .unwrap_or("");

        if field_name == "model" {
            let mut bytes = Vec::new();
            while let Some(chunk) = field.next().await {
                let chunk = match chunk {
                    Ok(data) => data,
                    Err(e) => {
                        return HttpResponse::InternalServerError().json(ErrorResponse {
                            error: ErrorDetail {
                                message: format!("Failed to read model field: {}", e),
                                error_type: "server_error".to_string(),
                                code: Some("model_read_failed".to_string()),
                            },
                        })
                    }
                };
                bytes.extend_from_slice(&chunk);
            }

            match String::from_utf8(bytes) {
                Ok(value) => {
                    let trimmed = value.trim();
                    if !trimmed.is_empty() {
                        model = Some(trimmed.to_string());
                    }
                }
                Err(_) => {
                    return HttpResponse::BadRequest().json(ErrorResponse {
                        error: ErrorDetail {
                            message: "Model field must be valid UTF-8".to_string(),
                            error_type: "invalid_request_error".to_string(),
                            code: Some("invalid_model_field".to_string()),
                        },
                    })
                }
            }
        } else if field_name == "files" {
            let filename_is_pdf = field
                .content_disposition()
                .and_then(|cd| cd.get_filename())
                .map(|name| name.to_lowercase().ends_with(".pdf"))
                .unwrap_or(false);

            let is_pdf_mime = field
                .content_type()
                .map(|mime| mime.essence_str() == "application/pdf")
                .unwrap_or(false);

            if !filename_is_pdf && !is_pdf_mime {
                return HttpResponse::BadRequest().json(ErrorResponse {
                    error: ErrorDetail {
                        message: "Uploaded file must be a PDF".to_string(),
                        error_type: "invalid_request_error".to_string(),
                        code: Some("invalid_file_type".to_string()),
                    },
                });
            }

            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let filename = format!("upload_{}_{}.pdf", timestamp, file_paths.len());
            let file_path = temp_dir.path().join(filename);

            let mut file = match File::create(&file_path).await {
                Ok(file) => file,
                Err(e) => {
                    return HttpResponse::InternalServerError().json(ErrorResponse {
                        error: ErrorDetail {
                            message: format!("Failed to create temp file: {}", e),
                            error_type: "server_error".to_string(),
                            code: Some("temp_file_creation_failed".to_string()),
                        },
                    })
                }
            };

            while let Some(chunk) = field.next().await {
                let data = match chunk {
                    Ok(data) => data,
                    Err(e) => {
                        return HttpResponse::InternalServerError().json(ErrorResponse {
                            error: ErrorDetail {
                                message: format!("Failed to read uploaded file: {}", e),
                                error_type: "server_error".to_string(),
                                code: Some("file_read_failed".to_string()),
                            },
                        })
                    }
                };

                if let Err(e) = file.write_all(&data).await {
                    return HttpResponse::InternalServerError().json(ErrorResponse {
                        error: ErrorDetail {
                            message: format!("Failed to write temp file: {}", e),
                            error_type: "server_error".to_string(),
                            code: Some("temp_file_write_failed".to_string()),
                        },
                    });
                }
            }

            file_paths.push(file_path);
        } else {
            while let Some(chunk) = field.next().await {
                if chunk.is_err() {
                    break;
                }
            }
        }
    }

    let model = match model {
        Some(model) => model,
        None => {
            return HttpResponse::BadRequest().json(ErrorResponse {
                error: ErrorDetail {
                    message: "Missing model field in multipart payload".to_string(),
                    error_type: "invalid_request_error".to_string(),
                    code: Some("missing_model".to_string()),
                },
            })
        }
    };

    if file_paths.is_empty() {
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: ErrorDetail {
                message: "No PDF files were uploaded".to_string(),
                error_type: "invalid_request_error".to_string(),
                code: Some("empty_files".to_string()),
            },
        });
    }

    let embedder = match EmbedderBuilder::new()
        .model_id(Some(model.as_str()))
        .from_pretrained_hf()
    {
        Ok(embedder) => Arc::new(embedder),
        Err(e) => {
            return HttpResponse::InternalServerError().json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("Failed to initialize embedder: {}", e),
                    error_type: "server_error".to_string(),
                    code: Some("embedder_init_failed".to_string()),
                },
            })
        }
    };

    let config = TextEmbedConfig::default();

    let embeddings =
        match embed_files_batch(file_paths.clone(), &embedder, Some(&config), None).await {
            Ok(Some(embeddings)) => embeddings,
            Ok(None) => Vec::new(),
            Err(e) => {
                return HttpResponse::InternalServerError().json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Failed to generate embeddings: {}", e),
                        error_type: "server_error".to_string(),
                        code: Some("embedding_generation_failed".to_string()),
                    },
                })
            }
        };

    let response = pdf_embedding_response(model.clone(), embeddings);

    HttpResponse::Ok().json(response)
}

pub fn run(listener: TcpListener) -> std::io::Result<Server> {
    let server = HttpServer::new(|| {
        App::new()
            .service(health_check)
            .service(create_embeddings)
            .service(create_pdf_embeddings)
            .service(create_pdf_embeddings_upload)
    })
    .listen(listener)?
    .run();
    Ok(server)
}
