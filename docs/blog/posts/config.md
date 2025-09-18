---
draft: false 
date: 2025-09-15
authors: 
 - sonam
slug: semantic-late-chunking
title: How to write textembedconfig for chunking
---
# How to Configure TextEmbedConfig in EmbedAnything

After presenting at Google, PyCon DE, Berlin Buzzwords, and GDG Berlin, I was surprised by how many people approached me with questions about writing configurations, chunk sizes, and batch sizes for EmbedAnything. Since I had never specifically covered this topic in my talks or blog posts, I decided to create this comprehensive guide to clarify these concepts and explain how we handle your chunking strategy with vector streaming.

<!-- more -->


## Understanding TextEmbedConfig

TextEmbedConfig consists of three essential components that work together to optimize your text embedding process:

1. **The embedding model** - defines how text is converted to vectors
2. **Splitting strategy with chunk size** - determines how documents are divided
3. **Batch size** - controls vector streaming performance

Let's explore each component in detail.

## Setting Up the Embedding Model

The foundation of any embedding configuration is the model itself. Here's how to initialize an embedding model:

```python
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Bert, 
    model_id="sentence-transformers/all-MiniLM-L12-v2"
)
```

This example uses a BERT-based model from Hugging Face, but you can choose from various architectures depending on your specific needs.

## Splitting Strategies and Chunk Size

EmbedAnything offers multiple splitting strategies, each designed for different use cases. The two primary approaches are semantic chunking and sentence-based splitting.

### Semantic Chunking

Semantic chunking groups similar content together based on meaning rather than arbitrary boundaries. This approach requires a semantic encoder to determine content similarity.

First, set up your semantic encoder:

```python
semantic_encoder = EmbeddingModel.from_pretrained_hf(
    WhichModel.Jina, 
    model_id="jinaai/jina-embeddings-v2-small-en"
)
```

Then configure TextEmbedConfig with semantic chunking:

```python
config = TextEmbedConfig(
    chunk_size=1000,
    batch_size=32,
    splitting_strategy="semantic",
    semantic_encoder=semantic_encoder
)
```

### Late Chunking Strategy

Late chunking is an advanced technique that preserves contextual relationships throughout the entire document during the embedding process. The document is embedded as a whole first, then divided into chunks while maintaining rich contextual information.

```python
config = TextEmbedConfig(
    chunk_size=1000,
    batch_size=8,
    splitting_strategy="sentence",
    late_chunking=True
)
```

Key benefits of late chunking:
- Maintains contextual relationships across the entire document
- Produces more meaningful embeddings for each chunk
- Particularly effective for longer documents with complex relationships

## Understanding Key Configuration Parameters

### Chunk Size
The `chunk_size` parameter defines the maximum number of characters (or tokens, depending on the model) allowed in each chunk. Consider these factors when setting chunk size:

- **Smaller chunks**: Better for precise retrieval, more granular search results
- **Larger chunks**: Better for maintaining context, fewer total chunks to process
- **Model limitations**: Ensure chunk size doesn't exceed your embedding model's maximum input length

### Batch Size for Vector Streaming
Batch size controls how many chunks the embedding model processes simultaneously. This directly impacts performance and memory usage:

```python
# Conservative approach for limited resources
config = TextEmbedConfig(chunk_size=1000, batch_size=8, splitting_strategy="sentence")

# Aggressive approach for high-performance systems
config = TextEmbedConfig(chunk_size=1000, batch_size=32, splitting_strategy="semantic")
```

**Choosing the right batch size:**
- **Smaller batches (4-8)**: Lower memory usage, more stable processing
- **Larger batches (16-32)**: Faster processing, higher memory requirements
- **Experimentation is key**: Test different batch sizes with your specific documents and hardware

### Splitting Strategy Options

EmbedAnything supports several splitting strategies:

- **"semantic"**: Groups content by meaning (requires semantic encoder)
- **"sentence"**: Splits at sentence boundaries
- **Custom strategies**: Can be implemented for specialized use cases

## Putting It All Together

Here's a complete example showing different configuration approaches:

```python
# Basic semantic chunking configuration
semantic_encoder = EmbeddingModel.from_pretrained_hf(
    WhichModel.Jina, 
    model_id="jinaai/jina-embeddings-v2-small-en"
)

config_semantic = TextEmbedConfig(
    chunk_size=1000,
    batch_size=16,
    splitting_strategy="semantic",
    semantic_encoder=semantic_encoder
)

# Late chunking configuration for complex documents
config_late_chunking = TextEmbedConfig(
    chunk_size=1500,
    batch_size=8,
    splitting_strategy="sentence",
    late_chunking=True
)

# Simple sentence-based chunking
config_simple = TextEmbedConfig(
    chunk_size=800,
    batch_size=24,
    splitting_strategy="sentence"
)
```

## Performance Optimization Tips

1. **Start with default values** (chunk_size=1000, batch_size=8) and adjust based on your specific use case
2. **Monitor memory usage** when increasing batch size
3. **Consider your documents' structure** when choosing splitting strategy
4. **Test retrieval quality** with different chunk sizes
5. **Profile your pipeline** to find the optimal batch size for your hardware

## Best Practices

- Use semantic chunking for documents where meaning preservation is crucial
- Implement late chunking for complex documents with intricate relationships
- Adjust batch size based on your available memory and processing requirements
- Regularly evaluate your configuration's impact on both performance and retrieval quality

By understanding these configuration options and their trade-offs, you can optimize EmbedAnything for your specific use case, whether you're processing technical documentation, literary texts, or any other type of content that requires intelligent chunking and embedding.