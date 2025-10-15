# Qwen3 Reranker in EmbedAnything

The Qwen3 Reranker is a powerful document relevance scoring and ranking model that has been integrated into EmbedAnything. It provides high-quality relevance scores for documents based on user queries, making it ideal for search and retrieval applications.

## Overview

The Qwen3 Reranker is based on the Qwen3 architecture and has been optimized with ONNX for efficient inference. It's specifically designed for:

- **Document Relevance Scoring**: Assigning relevance scores to documents based on queries
- **Search Result Reranking**: Improving search result quality by reranking candidates
- **Information Retrieval**: Enhancing retrieval systems with semantic understanding
- **Question-Answering**: Ranking candidate answers by relevance to questions

## Key Features

- **High Quality**: State-of-the-art relevance scoring capabilities
- **ONNX Optimized**: Fast inference using ONNX Runtime
- **Batch Processing**: Efficient handling of multiple queries and documents
- **Flexible Scoring**: Both ranked results and raw scores available
- **Easy Integration**: Simple Python API for quick implementation

## Installation

The Qwen3 Reranker is included with EmbedAnything. Install the package:

```bash
pip install embed-anything
```

Additional dependencies:
```bash
pip install onnxruntime  # For ONNX inference
```

## Quick Start

### Basic Usage

```python
from embed_anything import Reranker, Dtype

# Qwen3 Reranker requires additional formatting
def format_query(query: str, instruction=None):
    """You may add instruction to get better results in specific fields."""
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    return f"<Instruct>: {instruction}\n<Query>: {query}\n"

def format_document(doc: str):
    return f"<Document>: {doc}"

# Initialize the Qwen3 reranker
reranker = Reranker.from_pretrained(
    "zhiqing/Qwen3-Reranker-0.6B-ONNX", 
    dtype=Dtype.F32
)

# Rerank documents for a query
query = ["What is machine learning?"]
documents = [
    "Machine learning is a subset of AI.",
    "The weather is sunny today.",
    "ML algorithms can learn from data.",
    "Pizza is a popular food."
]

# Format query and documents
query = [format_query(x) for x in query]
documents = [format_document(x) for x in documents]

results = reranker.rerank(query, documents, top_k=2)

# Display results
for result in results:
    print(f"Query: {result.query}")
    for doc in result.documents:
        print(f"  Rank {doc.rank}: {doc.document}")
        print(f"    Score: {doc.relevance_score:.4f}")
```

### Multiple Queries

```python
# Rerank for multiple queries simultaneously
queries = [
    "How to make coffee?",
    "What is Python programming?"
]

documents = [
    "Coffee is brewed from beans.",
    "Python is a programming language.",
    "The weather is nice.",
    "Coffee can be made with a French press.",
    "Python is great for beginners."
]

# Format queries and documents
queries = [format_query(x) for x in queries]
documents = [format_document(x) for x in documents]

results = reranker.rerank(queries, documents, top_k=3)

for result in results:
    print(f"Query: {result.query}")
    for doc in result.documents:
        print(f"  {doc.document} (Score: {doc.relevance_score:.4f})")
```

### Custom Scoring

```python
# Get raw scores for custom processing
scores = reranker.compute_scores(query, documents, batch_size=4)

# Custom ranking logic
doc_scores = list(zip(documents, scores[0]))
doc_scores.sort(key=lambda x: x[1], reverse=True)

for doc, score in doc_scores:
    print(f"Score: {score:.4f} | {doc}")
```

## API Reference

### Reranker Class

#### `Reranker.from_pretrained()`

Loads a pre-trained reranker model.

**Parameters:**
- `model_id` (str): Hugging Face model ID (e.g., "zhiqing/Qwen3-Reranker-0.6B-ONNX")
- `revision` (str, optional): Model revision/branch
- `dtype` (Dtype, optional): Model data type (F32, F16, etc.)
- `path_in_repo` (str, optional): Path to model files in repository

**Returns:**
- `Reranker`: Initialized reranker instance

#### `rerank()`

Reranks documents for given queries.

**Parameters:**
- `query` (List[str]): List of query strings
- `documents` (List[str]): List of document strings to rank
- `top_k` (int): Number of top documents to return

**Returns:**
- `List[RerankerResult]`: List of reranking results

#### `compute_scores()`

Computes raw relevance scores.

**Parameters:**
- `query` (List[str]): List of query strings
- `documents` (List[str]): List of document strings
- `batch_size` (int): Batch size for processing

**Returns:**
- `List[List[float]]`: Raw relevance scores for each query-document pair

### Data Structures

#### `RerankerResult`

```python
class RerankerResult:
    query: str                    # The query string
    documents: List[DocumentRank] # Ranked documents
```

#### `DocumentRank`

```python
class DocumentRank:
    document: str        # Document text
    relevance_score: float # Relevance score (0.0 to 1.0)
    rank: int           # Rank position (1-based)
```

## Model Variants

### Available Models

- **zhiqing/Qwen3-Reranker-0.6B-ONNX**: 0.6B parameter model, ONNX optimized
- **zhiqing/Qwen3-Reranker-0.6B**: Original PyTorch model

### Data Types

- **F32**: Full precision (default, best quality)
- **F16**: Half precision (faster, slightly lower quality)
- **INT8**: 8-bit quantization (fastest, lower quality)

## Performance Considerations

### Batch Processing

For optimal performance when processing multiple documents:

```python
# Use appropriate batch sizes
scores = reranker.compute_scores(queries, documents, batch_size=8)
```

### Memory Usage

- **F32**: ~2.4GB memory usage
- **F16**: ~1.2GB memory usage
- **INT8**: ~600MB memory usage

### Speed vs Quality Trade-offs

- **F32**: Best quality, slower inference
- **F16**: Good quality, balanced performance
- **INT8**: Lower quality, fastest inference

## Use Cases

### 1. Search Engine Reranking

```python
# After vector search, rerank results
vector_results = vector_search(query, top_k=100)
reranked_results = reranker.rerank([query], vector_results, top_k=10)
```

### 2. Question Answering

```python
# Rank candidate answers
question = "What is the capital of France?"
candidate_answers = [
    "Paris is the capital of France.",
    "France is a country in Europe.",
    "The Eiffel Tower is in Paris."
]

ranked_answers = reranker.rerank([question], candidate_answers, top_k=1)
best_answer = ranked_answers[0].documents[0].document
```

### 3. Document Filtering

```python
# Filter documents by relevance threshold
scores = reranker.compute_scores([query], documents, batch_size=4)
relevant_docs = [
    doc for doc, score in zip(documents, scores[0]) 
    if score > 0.5  # Threshold
]
```

### 4. Content Recommendation

```python
# Rank content by user query relevance
user_interest = "machine learning"
content_items = [
    "Introduction to ML",
    "Python programming basics",
    "Advanced ML algorithms",
    "Web development tutorial"
]

recommendations = reranker.rerank([user_interest], content_items, top_k=3)
```

## Best Practices

### 1. Model Selection

- Use ONNX models for production (faster inference)
- Choose data type based on quality vs. speed requirements
- Consider model size for memory constraints

### 2. Query Formulation

- Keep queries clear and specific
- Use natural language (the model understands context)
- Avoid overly long or complex queries

### 3. Document Preparation

- Ensure documents are well-formatted
- Remove irrelevant content before reranking
- Consider document length (very long documents may affect performance)

### 4. Performance Optimization

- Use batch processing for multiple queries
- Implement caching for repeated queries
- Consider async processing for web applications

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure ONNX Runtime is installed
   - Check internet connection for model download
   - Verify model ID is correct

2. **Memory Issues**
   - Use smaller data types (F16, INT8)
   - Reduce batch size
   - Process documents in smaller chunks

3. **Performance Issues**
   - Use ONNX models
   - Optimize batch sizes
   - Consider hardware acceleration (GPU)

### Error Messages

- **"Model not found"**: Check model ID and internet connection
- **"ONNX Runtime error"**: Ensure onnxruntime is properly installed
- **"Memory allocation failed"**: Reduce batch size or use smaller data type

## Examples

See the following example files for complete working examples:

- `examples/qwen3_reranker.py` - Comprehensive examples
- `examples/reranker.py` - Basic usage examples
- `rust/examples/reranker.rs` - Rust implementation examples

## Contributing

The Qwen3 reranker implementation is part of the EmbedAnything project. Contributions are welcome! See the main project documentation for contribution guidelines.

## License

The Qwen3 reranker is subject to the same license as the EmbedAnything project. Please refer to the project LICENSE file for details.
