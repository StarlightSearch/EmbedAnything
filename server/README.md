# EmbedAnything OpenAI-Compatible Server

This server provides an OpenAI-compatible API for generating embeddings using the EmbedAnything library.

## Features

- OpenAI-compatible `/v1/embeddings` endpoint
- PDF embeddings via `/v1/pdf_embeddings/upload` (file upload)
- Support for multiple embedding models (Jina, BERT, etc.)
- Health check endpoint

## Running the Server

```bash
cd server
cargo run
```

The server will start on `http://0.0.0.0:8080`.

## API Usage

### Create Embeddings

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

### PDF Embeddings (Upload Files)

**Endpoint:** `POST /v1/pdf_embeddings/upload`

Upload PDF files as multipart form data to generate embeddings. Accepts one or more PDF files.

**Request:** `multipart/form-data`
- `model` (required): The embedding model to use (e.g., `sentence-transformers/all-MiniLM-L12-v2`)
- `files` (required): One or more PDF file uploads

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0023064255, -0.009327292, ...],
      "metadata": null,
      "text": "Extracted text from PDF page..."
    }
  ],
  "model": "sentence-transformers/all-MiniLM-L12-v2"
}
```

### Health Check

**Endpoint:** `GET /health_check`

Returns a 200 OK status if the server is running.

## Error Handling

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

## Example Usage with curl

```bash
# Create embeddings
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L12-v2",
    "input": ["Hello world", "How are you?"]
  }'

# Upload PDF files for embeddings
curl -X POST http://localhost:8080/v1/pdf_embeddings/upload \
  -F "model=sentence-transformers/all-MiniLM-L12-v2" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"

# Health check
curl http://localhost:8080/health_check
```

## Example Usage with Python

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

# Upload PDF files for embeddings
with open("document.pdf", "rb") as f:
    upload_response = requests.post(
        "http://localhost:8080/v1/pdf_embeddings/upload",
        data={"model": "sentence-transformers/all-MiniLM-L12-v2"},
        files={"files": ("document.pdf", f, "application/pdf")}
    )
if upload_response.status_code == 200:
    data = upload_response.json()
    print(f"Generated {len(data['data'])} PDF embeddings")
```
