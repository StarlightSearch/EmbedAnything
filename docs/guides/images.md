# Searching Images

EmbedAnything enables semantic image search using vision-language models like CLIP and SigLip. These models can understand both images and text, allowing you to search images using natural language queries.

## Overview

Image search with EmbedAnything works by:
1. **Embedding images**: Converting images into high-dimensional vectors
2. **Embedding queries**: Converting text queries into the same vector space
3. **Similarity search**: Finding images with similar embeddings to your query

## Basic Image Search

```python
import embed_anything
import numpy as np
from embed_anything import EmbeddingModel, WhichModel

# Load CLIP model for image-text embeddings
model = EmbeddingModel.from_pretrained_hf(
    model_id="openai/clip-vit-base-patch16"
)

# Embed all images in a directory
data = embed_anything.embed_image_directory("test_files", embedder=model)

# Convert embeddings to numpy array
embeddings = np.array([item.embedding for item in data])

# Embed a text query
query = ["Photo of a monkey"]
query_embedding = np.array(
    embed_anything.embed_query(query, embedder=model)[0].embedding
)

# Calculate cosine similarities
similarities = np.dot(embeddings, query_embedding)

# Find most similar images
most_similar_idx = np.argmax(similarities)
print(f"Most similar image: {data[most_similar_idx].text}")
print(f"Similarity score: {similarities[most_similar_idx]:.4f}")
```

## Ranking Multiple Results

```python
import embed_anything
import numpy as np
from embed_anything import EmbeddingModel, WhichModel

# Load model
model = EmbeddingModel.from_pretrained_hf(
    model_id="openai/clip-vit-base-patch16"
)

# Embed images
data = embed_anything.embed_image_directory("test_files", embedder=model)
embeddings = np.array([item.embedding for item in data])

# Query
query = ["Photo of a monkey"]
query_embedding = np.array(
    embed_anything.embed_query(query, embedder=model)[0].embedding
)

# Calculate similarities
similarities = np.dot(embeddings, query_embedding)

# Get top 5 results
top_5_indices = np.argsort(similarities)[::-1][:5]

print("Top 5 most similar images:")
for i, idx in enumerate(top_5_indices, 1):
    print(f"{i}. {data[idx].text} (score: {similarities[idx]:.4f})")
```

## Using SigLip (Alternative to CLIP)

SigLip is a newer model that often performs better than CLIP:

```python
from embed_anything import EmbeddingModel

# Load SigLip model
model = EmbeddingModel.from_pretrained_hf(
    model_id="google/siglip-base-patch16-224"
)

# Use it the same way as CLIP
data = embed_anything.embed_image_directory("test_files", embedder=model)
```

## Image-to-Image Search

You can also search for images similar to a reference image:

```python
import embed_anything
import numpy as np
from embed_anything import EmbeddingModel, WhichModel

# Load model
model = EmbeddingModel.from_pretrained_hf(
    model_id="openai/clip-vit-base-patch16"
)

# Embed all images
data = embed_anything.embed_image_directory("test_files", embedder=model)
embeddings = np.array([item.embedding for item in data])

# Embed a reference image (treat it as a query)
reference_image_path = "test_files/reference.jpg"
reference_data = embed_anything.embed_file(reference_image_path, embedder=model)
reference_embedding = np.array(reference_data[0].embedding)

# Find similar images
similarities = np.dot(embeddings, reference_embedding)
most_similar_idx = np.argmax(similarities)

print(f"Most similar to reference: {data[most_similar_idx].text}")
```

## Complete Example

``` python
--8<-- "examples/clip.py"
```

## Supported Models

EmbedAnything supports the following models for image search:

### CLIP Models
- `openai/clip-vit-base-patch32` - Fastest, smaller embeddings
- `openai/clip-vit-base-patch16` - Balanced performance (recommended)
- `openai/clip-vit-large-patch14-336` - Best quality, slower
- `openai/clip-vit-large-patch14` - High quality

### SigLip Models
- `google/siglip-base-patch16-224` - Modern alternative to CLIP
- `google/siglip-large-patch16-384` - Higher quality

## Performance Tips

1. **Model selection**: Use `clip-vit-base-patch16` for a good balance of speed and quality
2. **Batch processing**: Process multiple images in parallel
3. **GPU acceleration**: Install `embed-anything-gpu` for faster inference
4. **Normalize embeddings**: For cosine similarity, embeddings are automatically normalized

## Use Cases

- **Content moderation**: Find inappropriate images
- **Product search**: Search e-commerce catalogs
- **Photo organization**: Organize personal photo libraries
- **Visual search**: Find similar products or designs
- **Accessibility**: Describe images for visually impaired users