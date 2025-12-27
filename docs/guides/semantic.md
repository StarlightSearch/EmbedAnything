# Using Semantic Chunking

Semantic chunking is a technique that splits text at semantically meaningful boundaries rather than fixed character or token limits. This approach preserves the logical flow and meaning of text, making it essential for applications like document retrieval, question answering, and summarization.

## Why Semantic Chunking?

Traditional chunking methods split text at fixed sizes (e.g., every 1000 characters), which can:
- Break sentences mid-thought
- Split related concepts
- Lose context between chunks

Semantic chunking addresses these issues by:
- Preserving complete thoughts and sentences
- Maintaining context within chunks
- Improving retrieval quality
- Better alignment with embedding model understanding

## Basic Usage

```python
from embed_anything import EmbeddingModel, WhichModel, TextEmbedConfig
import embed_anything

# Main embedding model - generates final embeddings
model = EmbeddingModel.from_pretrained_hf(
    model_id="sentence-transformers/all-MiniLM-L12-v2"
)

# Semantic encoder - determines where to split text
# This model analyzes text to find natural semantic breaks
semantic_encoder = EmbeddingModel.from_pretrained_hf(
    model_id="jinaai/jina-embeddings-v2-small-en"
)

# Configure semantic chunking
config = TextEmbedConfig(
    chunk_size=1000,                    # Target chunk size (approximate)
    batch_size=32,                      # Process 32 chunks at once
    splitting_strategy="semantic",      # Enable semantic splitting
    semantic_encoder=semantic_encoder    # Model for semantic analysis
)

# Embed a file with semantic chunking
data = embed_anything.embed_file("test_files/document.pdf", embedder=model, config=config)

# Chunks will be split at semantically meaningful boundaries
for i, item in enumerate(data):
    print(f"Chunk {i+1}:")
    print(f"Text: {item.text[:200]}...")
    print(f"Length: {len(item.text)} characters")
    print("---" * 20)
```

## How It Works

1. **Text Analysis**: The semantic encoder analyzes the text to identify semantic boundaries
2. **Boundary Detection**: Finds natural break points (sentence endings, paragraph breaks, topic shifts)
3. **Chunk Creation**: Creates chunks that respect these boundaries while staying close to the target size
4. **Embedding**: The main model embeds each semantically coherent chunk

## Advanced Configuration

```python
from embed_anything import EmbeddingModel, WhichModel, TextEmbedConfig
import embed_anything

# Use a more powerful semantic encoder for better boundary detection
semantic_encoder = EmbeddingModel.from_pretrained_hf(
    model_id="jinaai/jina-embeddings-v2-base-en"  # Larger model for better analysis
)

# Fine-tune chunking parameters
config = TextEmbedConfig(
    chunk_size=1500,                    # Larger chunks for more context
    batch_size=16,                      # Smaller batches for memory efficiency
    splitting_strategy="semantic",
    semantic_encoder=semantic_encoder,
    buffer_size=128                     # Buffer for streaming
)

# Process multiple files
data = embed_anything.embed_directory(
    "test_files/",
    embedder=model,
    config=config
)
```

## Comparison: Semantic vs. Regular Chunking

### Regular Chunking (Sentence-based)
```python
config = TextEmbedConfig(
    chunk_size=1000,
    splitting_strategy="sentence"  # Simple sentence splitting
)
```
**Result**: Chunks may break at awkward points, losing context.

### Semantic Chunking
```python
config = TextEmbedConfig(
    chunk_size=1000,
    splitting_strategy="semantic",
    semantic_encoder=semantic_encoder
)
```
**Result**: Chunks preserve meaning and context, improving retrieval quality.

## Best Practices

1. **Choose appropriate semantic encoder**: Use a model that matches your domain
2. **Balance chunk size**: Too small loses context, too large dilutes focus
3. **Match encoder and embedder**: Use similar model families for consistency
4. **Test chunk quality**: Inspect chunks to ensure they're semantically coherent

## Complete Example

``` python
--8<-- "examples/semantic_chunking.py"
```

## Use Cases

- **RAG Systems**: Better context preservation for retrieval-augmented generation
- **Document Q&A**: More accurate answers by maintaining context
- **Long-form content**: Books, research papers, technical documentation
- **Conversational AI**: Maintaining dialogue context