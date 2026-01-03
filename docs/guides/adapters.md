# Using Vector Database Adapters

Vector database adapters allow you to stream embeddings directly to your vector database without keeping them in memory. This enables efficient, memory-safe indexing of large document collections.

## How Adapters Work

Adapters implement a common interface that:
1. **Create indexes**: Set up vector indexes with the correct dimensions
2. **Stream embeddings**: Receive embeddings as they're generated
3. **Store metadata**: Preserve text, file paths, and other metadata
4. **Enable search**: Make embeddings searchable via the vector database

## Using Elasticsearch

Elasticsearch is a distributed search and analytics engine. Perfect for production deployments requiring scalability.

**Installation:**
```bash
pip install elasticsearch
```

**Basic Usage:**
```python
import embed_anything
import os
from embed_anything import EmbeddingModel, WhichModel, TextEmbedConfig
from embed_anything.vectordb import ElasticAdapter

# Initialize Elasticsearch adapter
es_adapter = ElasticAdapter(
    url="http://localhost:9200",  # Elasticsearch URL
    api_key=None  # Optional: API key for cloud Elasticsearch
)

# Create an index
es_adapter.create_index(
    dimension=384,           # Embedding dimension
    metric="cosine",         # Similarity metric
    index_name="documents"
)

# Load embedding model
model = EmbeddingModel.from_pretrained_hf(
    model_id="sentence-transformers/all-MiniLM-L12-v2"
)

# Embed files and stream to Elasticsearch
config = TextEmbedConfig(chunk_size=1000, batch_size=32)
data = embed_anything.embed_file(
    "test_files/document.pdf",
    embedder=model,
    config=config,
    adapter=es_adapter  # Streams directly to Elasticsearch
)
```

**Complete Example:**
``` python
--8<-- "examples/adapters/elastic.py"
```

## Using Weaviate

Weaviate is an open-source vector database with built-in ML capabilities.

**Installation:**
```bash
pip install weaviate-client
```

**Basic Usage:**
```python
import embed_anything
from embed_anything import EmbeddingModel, WhichModel
from embed_anything.vectordb import WeaviateAdapter

# Initialize Weaviate adapter
weaviate_adapter = WeaviateAdapter(
    url="http://localhost:8080",  # Weaviate instance URL
    api_key=None  # Optional: API key for cloud Weaviate
)

# Create a collection (index)
weaviate_adapter.create_index(
    dimension=384,
    metric="cosine",
    index_name="Documents"
)

# Load model and embed
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Bert,
    model_id="sentence-transformers/all-MiniLM-L12-v2"
)

# Stream embeddings to Weaviate
data = embed_anything.embed_directory(
    "test_files/",
    embedder=model,
    adapter=weaviate_adapter
)
```

**Complete Example:**
``` python
--8<-- "examples/adapters/weaviate_db.py"
```

## Using Pinecone

Pinecone is a managed vector database service, ideal for production applications.

**Installation:**
```bash
pip install pinecone
```

**Basic Usage:**
```python
import embed_anything
import os
from embed_anything import EmbeddingModel, WhichModel
from embed_anything.vectordb import PineconeAdapter

# Initialize Pinecone adapter
api_key = os.environ.get("PINECONE_API_KEY")
pinecone_adapter = PineconeAdapter(api_key)

# Create or use existing index
try:
    pinecone_adapter.delete_index("documents")
except:
    pass

pinecone_adapter.create_index(
    dimension=512,      # Must match your model's embedding dimension
    metric="cosine",
    index_name="documents"
)

# Load model
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Clip,
    model_id="openai/clip-vit-base-patch16"
)

# Embed images and stream to Pinecone
data = embed_anything.embed_image_directory(
    "test_files",
    embedder=model,
    adapter=pinecone_adapter
)
```

**Complete Example:**
``` python
--8<-- "examples/adapters/pinecone_db.py"
```

## Using Qdrant

[Qdrant](https://qdrant.tech/) is a high-performance vector database written in Rust.

**Installation:**
```bash
pip install qdrant-client
```

**Basic Usage:**
```python
import embed_anything
from embed_anything import EmbeddingModel, WhichModel
from embed_anything.vectordb import QdrantAdapter

# Initialize Qdrant adapter
qdrant_adapter = QdrantAdapter(
    url="http://localhost:6333",  # Qdrant server URL
    api_key=None  # Optional: API key for cloud Qdrant
)

# Create a collection
qdrant_adapter.create_index(
    dimension=384,
    metric="cosine",
    index_name="documents"
)

# Load model and embed
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Bert,
    model_id="sentence-transformers/all-MiniLM-L12-v2"
)

# Stream to Qdrant
data = embed_anything.embed_file(
    "test_files/document.pdf",
    embedder=model,
    adapter=qdrant_adapter
)
```

**Complete Example:**
``` python
--8<-- "examples/adapters/qdrant.py"
```

## Using Milvus

[Milvus](https://milvus.io/) is an open-source vector database built for scalable similarity search.

**Installation:**
```bash
pip install pymilvus
```

**Basic Usage:**
```python
import embed_anything
from embed_anything import EmbeddingModel, WhichModel
from embed_anything.vectordb import MilvusAdapter

# Initialize Milvus adapter
milvus_adapter = MilvusAdapter(
    host="localhost",  # Milvus host
    port=19530,        # Milvus port
    user=None,         # Optional: username
    password=None      # Optional: password
)

# Create a collection
milvus_adapter.create_index(
    dimension=384,
    metric="cosine",
    index_name="documents"
)

# Load model and embed
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Bert,
    model_id="sentence-transformers/all-MiniLM-L12-v2"
)

# Stream to Milvus
data = embed_anything.embed_directory(
    "test_files/",
    embedder=model,
    adapter=milvus_adapter
)
```

**Complete Example:**
``` python
--8<-- "examples/adapters/milvus_db.py"
```

## Benefits of Using Adapters

1. **Memory efficiency**: Embeddings are streamed directly to the database
2. **Scalability**: Handle large document collections without memory issues
3. **Persistence**: Embeddings are stored permanently in the database
4. **Search ready**: Immediately searchable after embedding
5. **Production ready**: Integrates with your existing vector database infrastructure

## Choosing the Right Adapter

- **Elasticsearch**: Best for existing Elasticsearch deployments, full-text + vector search
- **Weaviate**: Great for ML-powered applications, built-in classification
- **Pinecone**: Managed service, zero infrastructure management
- **Qdrant**: High performance, Rust-based, self-hosted
- **Milvus**: Scalable, cloud-native, production-grade

## Best Practices

1. **Match dimensions**: Ensure the adapter's dimension matches your model's output
2. **Choose metric**: Use "cosine" for most cases, "euclidean" for distance-based
3. **Batch processing**: Process multiple files for better throughput
4. **Error handling**: Wrap adapter calls in try-except blocks
5. **Index management**: Delete and recreate indexes when changing models
