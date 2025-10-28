# EmbedAnything OpenAI-Compatible Server

This server provides an OpenAI-compatible API for generating embeddings using the EmbedAnything library.

## Features

- OpenAI-compatible `/v1/embeddings` endpoint
- Support for multiple embedding models (Jina, BERT, etc.)
- Proper error handling with OpenAI-style error responses
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
  "model": "text-embedding-ada-002",
  "input": ["The quick brown fox jumps over the lazy dog"],
  "encoding_format": "float"
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
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 9,
    "total_tokens": 9
  }
}
```

### Health Check

**Endpoint:** `GET /health_check`

Returns a 200 OK status if the server is running.

## Supported Models

The server maps OpenAI model names to EmbedAnything model architectures:

- `text-embedding-ada-002` → Jina embeddings
- `text-embedding-3-small` → Jina embeddings  
- `text-embedding-3-large` → Jina embeddings
- `text-embedding-ada-001` → BERT embeddings
- Unknown models → Default to Jina embeddings

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
    "model": "text-embedding-ada-002",
    "input": ["Hello world", "How are you?"]
  }'

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
        "model": "text-embedding-ada-002",
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
