# Using Qwen3-Embedding

Qwen3-Embedding is a powerful multilingual embedding model designed for text embedding, retrieval, and reranking tasks. It supports over 100 languages, including various programming languages, making it ideal for international applications and code search.

## Key Features

- **Multilingual support**: Over 100 languages including programming languages
- **High quality**: State-of-the-art performance on retrieval benchmarks
- **Efficient**: 0.6B parameter model balances quality and speed
- **Versatile**: Works for both embedding and reranking tasks

## Basic Usage

```python
from embed_anything import EmbeddingModel, WhichModel, TextEmbedConfig, Dtype
import numpy as np

# Initialize Qwen3 embedding model
model = EmbeddingModel.from_pretrained_hf(
    model_id="Qwen/Qwen3-Embedding-0.6B",
    dtype=Dtype.F32  # Use float32 for best quality, F16 for faster inference
)

# Configure embedding parameters
config = TextEmbedConfig(
    chunk_size=1000,
    batch_size=2,
    splitting_strategy="sentence"
)

# Embed a document
data = model.embed_file("test_files/document.pdf", config=config)

# Access embeddings
for item in data:
    print(f"Text: {item.text[:200]}...")
    print(f"Embedding dimension: {len(item.embedding)}")
```

## Query and Retrieval

```python
from embed_anything import EmbeddingModel, WhichModel, TextEmbedConfig, Dtype
import numpy as np

# Load model
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Qwen3, 
    model_id="Qwen/Qwen3-Embedding-0.6B",
    dtype=Dtype.F32
)

# Embed documents
config = TextEmbedConfig(chunk_size=1000, batch_size=2, splitting_strategy="sentence")
data = model.embed_file("test_files/document.pdf", config=config)

# Embed a query
query = "Which GPU is used for training"
query_embedding = np.array(model.embed_query([query])[0].embedding)

# Calculate similarities
embedding_array = np.array([e.embedding for e in data])
similarities = np.matmul(query_embedding, embedding_array.T)

# Get top 5 most relevant chunks
top_5_indices = np.argsort(similarities)[-5:][::-1]

print("Top 5 most relevant results:")
for i, idx in enumerate(top_5_indices, 1):
    print(f"\n{i}. Score: {similarities[idx]:.4f}")
    print(f"   Text: {data[idx].text[:200]}...")
```

## Multilingual Support

Qwen3 excels at multilingual tasks:

```python
from embed_anything import EmbeddingModel, WhichModel, Dtype
import numpy as np

# Load model
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Qwen3,
    model_id="Qwen/Qwen3-Embedding-0.6B",
    dtype=Dtype.F32
)

# Embed text in different languages
texts = [
    "Hello, how are you?",           # English
    "Bonjour, comment allez-vous?",  # French
    "Hola, ¿cómo estás?",            # Spanish
    "你好，你好吗？",                  # Chinese
    "こんにちは、元気ですか？"          # Japanese
]

embeddings = model.embed_query(texts)
embeddings_array = np.array([e.embedding for e in embeddings])

# Calculate cross-lingual similarities
similarities = np.matmul(embeddings_array, embeddings_array.T)

# Find similar meanings across languages
for i, text1 in enumerate(texts):
    for j, text2 in enumerate(texts[i+1:], i+1):
        print(f"Similarity: {similarities[i][j]:.4f}")
        print(f"  {text1}")
        print(f"  {text2}\n")
```

## Code Search

Qwen3 is excellent for code search and retrieval:

```python
from embed_anything import EmbeddingModel, WhichModel, TextEmbedConfig, Dtype
import numpy as np

# Load model
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Qwen3,
    model_id="Qwen/Qwen3-Embedding-0.6B",
    dtype=Dtype.F32
)

# Embed code files
config = TextEmbedConfig(chunk_size=1000, batch_size=4)
code_embeddings = model.embed_file("src/main.py", config=config)

# Search for code functionality
query = "function that processes user input"
query_emb = np.array(model.embed_query([query])[0].embedding)

# Find relevant code snippets
code_array = np.array([e.embedding for e in code_embeddings])
similarities = np.matmul(query_emb, code_array.T)

top_results = np.argsort(similarities)[-3:][::-1]
for idx in top_results:
    print(f"Score: {similarities[idx]:.4f}")
    print(f"Code: {code_embeddings[idx].text}\n")
```

## Performance Optimization

```python
from embed_anything import EmbeddingModel, WhichModel, Dtype

# Use F16 for faster inference with minimal quality loss
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Qwen3,
    model_id="Qwen/Qwen3-Embedding-0.6B",
    dtype=Dtype.F16  # Half precision for 2x speedup
)

# Increase batch size for better throughput
config = TextEmbedConfig(
    chunk_size=1000,
    batch_size=8,  # Larger batch for better GPU utilization
    splitting_strategy="sentence"
)
```

## Complete Example

``` python
--8<-- "examples/qwen.py"
```

## When to Use Qwen3

- **Multilingual applications**: When you need to support multiple languages
- **Code search**: Searching through codebases and documentation
- **International RAG**: Building RAG systems for global users
- **Mixed content**: Documents with text and code
- **Quality over speed**: When embedding quality is more important than speed

## Comparison with Other Models

| Model | Languages | Code Support | Speed | Quality |
|-------|-----------|--------------|-------|---------|
| Qwen3 | 100+ | Excellent | Medium | High |
| Jina | English-focused | Good | Fast | High |
| BERT | English | Limited | Fast | Medium |

## Best Practices

1. **Use F32 for production**: Better quality, acceptable speed
2. **Use F16 for development**: Faster iteration, minimal quality loss
3. **Batch processing**: Process multiple documents together
4. **Appropriate chunk size**: 500-1500 characters work well
5. **Sentence splitting**: Use sentence splitting for better chunk quality