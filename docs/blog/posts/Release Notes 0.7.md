---
draft: false 
date: 2026-01-11
authors: 
 - sonam
slug: release-notes-7
title: Release Notes 0.7
---
# Release Notes 0.7

0.7 is all about making deployment easy and integrations to data sources. Including Prebuilt Docker Image,SearchR1 example to improve context for Agents and AWS S3 bucket Integration.

<!-- more -->

## 📋 Summary of Changes (0.6.8 → 0.7.0)

This release represents a significant milestone in making EmbedAnything easier to deploy and more powerful for production use. Here's a summary of the key improvements and additions since version 0.6.8:

### 🚀 Major Features

- **Prebuilt Docker Image**: Production-ready Docker image available for immediate deployment
- **AWS S3 Integration**: Direct support for fetching and embedding files from S3 buckets
- **SearchR1 Agent**: Advanced agent framework for interweaving retrieved results to improve context

### 🐛 Bug Fixes & Stability

- **Fixed Parameter Mismatching in Rerank**: Resolved issues with reranking function parameters
- **Fixed Qwen3Embed Concurrency Panic**: Addressed panic issues when using Qwen3 embeddings concurrently
- **Fixed File Extension Handling**: Improved error handling for files without extensions
- **Enhanced Error Handling**: Better error messages and recovery mechanisms


---

## 🐳 Prebuilt Docker Image

We're excited to announce that a prebuilt Docker image is now available! You can now pull the image and spin up the server without building from source. This makes deployment significantly easier and faster.

### Quick Start with Docker

#### Pull the Prebuilt Image

```bash
docker pull starlightsearch/embedanything-server:latest
```

#### Run the Container

```bash
docker run -p 8080:8080 starlightsearch/embedanything-server:latest
```

The server will start on `http://0.0.0.0:8080`.

### Building from Source

If you prefer to build the Docker image yourself, you can use the provided Dockerfile:

```bash
docker build -f server.Dockerfile -t embedanything-server .
docker run -p 8080:8080 embedanything-server
```

### Server Features

The Actix server provides an OpenAI-compatible API for generating embeddings. We chose Actix Server for:

1. **Blazing fast**: Consistently ranks among the fastest web frameworks in benchmarks like TechEmpower
2. **Asynchronous by default**: Built on Rust's async/await, enabling efficient I/O-bound workloads
3. **Lightweight & modular**: Minimal core with extensible middleware, plugins, and integrations
4. **Type-safe**: Strong type guarantees ensure fewer runtime surprises
5. **Production-ready**: Stable, mature, and already used in industries like fintech, IoT, and SaaS platforms

For benchmarks between Python and Rust servers, check out this blog: https://www.jonvet.com/blog/benchmarking-python-rust-web-servers

### API Usage

#### Create Embeddings

**Endpoint:** `POST /v1/embeddings`

**Request:**
```json
{
  "model": "sentence-transformers/all-MiniLM-L12-v2",
  "input": ["The quick brown fox jumps over the lazy dog"]
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0023064255, -0.009327292, ...]
    }
  ],
  "model": "sentence-transformers/all-MiniLM-L12-v2",
  "usage": {
    "prompt_tokens": 9,
    "total_tokens": 9
  }
}
```

#### Health Check

**Endpoint:** `GET /health_check`

Returns a 200 OK status if the server is running.

#### Example Usage with curl

```bash
# Create embeddings
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L12-v2",
    "input": ["Hello world", "How are you?"]
  }'

# Health check
curl http://localhost:8080/health_check
```

#### Example Usage with Python

```python
import requests

# Create embeddings
response = requests.post(
    "http://localhost:8080/v1/embeddings",
    json={
        "model": "sentence-transformers/all-MiniLM-L12-v2",
        "input": ["The quick brown fox jumps over the lazy dog"]
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"Generated {len(data['data'])} embeddings")
    print(f"First embedding dimension: {len(data['data'][0]['embedding'])}")
else:
    print(f"Error: {response.json()}")
```

### Error Handling

The API returns OpenAI-compatible error responses:

```json
{
  "error": {
    "message": "Error description",
    "type": "error_type",
    "code": "error_code"
  }
}
```

For more details, see the [Actix Server Guide](/docs/guides/actix_server.md).

## 🔍 SearchR1 Agent Integration

We've included the SearchR1 agent, which is a powerful method of interweaving retrieved results to improve context. This agent enables more sophisticated reasoning by dynamically integrating search results into the generation process.

### How SearchR1 Works

SearchR1 uses a unique approach where:
- The model conducts reasoning inside `<think>` tags
- When knowledge gaps are identified, it calls a search engine via `<search> query </search>` tags
- Search results are returned between `<information>` and `</information>` tags
- The agent can search multiple times, iteratively refining its understanding
- Once sufficient information is gathered, it provides the answer inside `<answer>` tags

This interweaving of retrieved results with reasoning creates a more contextually aware and accurate response generation process. The agent actively identifies knowledge gaps and explores different perspectives of a topic before providing a final answer.

### Example Usage

The SearchR1 agent is available in our examples. Check out `examples/SearchAgent/` for complete implementation examples showing how to integrate SearchR1 with EmbedAnything's retrieval capabilities using LanceDB.

## ☁️ Direct AWS S3 Bucket Integration

We've added direct integration with AWS S3 buckets, allowing you to fetch and embed files directly from your S3 storage without manual downloads.

### Features

- Fetch files directly from S3 buckets
- Support for explicit credentials or environment variables
- Seamless integration with EmbedAnything's embedding pipeline
- Save files locally or work with them in memory

### Usage

#### Using Explicit Credentials

```python
from embed_anything import S3Client, EmbeddingModel, WhichModel, TextEmbedConfig

# Create S3Client with credentials
s3_client = S3Client(
    access_key_id="your-access-key-id",
    secret_access_key="your-secret-access-key",
    region="us-east-1"
)

# Fetch a file from S3
file = s3_client.get_file_from_s3(
    bucket_name="your-bucket-name", 
    key="path/to/your/file.txt"
).save_file()

# Embed the file
embedder = EmbeddingModel.from_pretrained_hf(
    model_id="jinaai/jina-embeddings-v2-small-en"
)
embeddings = embedder.embed_file(
    file, 
    config=TextEmbedConfig(
        chunk_size=1000, 
        batch_size=32, 
        splitting_strategy="sentence"
    )
)
```

#### Using Environment Variables

```python
from embed_anything import S3Client

# Create S3Client from environment variables
# Reads from: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
s3_client = S3Client.from_env()

# Fetch and use files as above
file = s3_client.get_file_from_s3(
    bucket_name="your-bucket-name", 
    key="path/to/your/file.pdf"
).save_file()
```

### S3Client Methods

- `get_file_from_s3(bucket_name, key)`: Fetches a file from S3 and returns an `S3File` object
- `S3File.save_file(file_path)`: Saves the file to the local filesystem (optional path parameter)
- `S3File.bytes`: Access file contents as bytes
- `S3File.key`: Get the S3 key/path

For a complete example, see `examples/s3_example.py`.

---

We're excited about these improvements and look forward to seeing how you use them in your projects! For questions or feedback, please open an issue on our [GitHub repository](https://github.com/StarlightSearch/EmbedAnything).
