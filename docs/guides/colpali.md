# Using ColPali

ColPali is a specialized model designed for high-performance document embedding and semantic search. It processes PDFs by treating each page as an image, making it ideal for documents with complex layouts, tables, and figures. ColPali supports both native Candle and ONNX formats for flexible deployment.

## Key Features

- **Document-focused**: Optimized for PDF documents with visual elements
- **Page-level embeddings**: Each page is embedded as a whole, preserving layout context
- **Fast inference**: ONNX support for optimized performance
- **Query matching**: Efficient similarity search across document pages

## Basic Usage

```python
from embed_anything import ColpaliModel
import numpy as np
from pathlib import Path

# Load ColPali model (Candle backend)
model = ColpaliModel.from_pretrained("vidore/colpali-v1.2-merged", None)

# Or load ONNX model for faster inference
# model = ColpaliModel.from_pretrained_onnx(
#     "starlight-ai/colpali-v1.2-merged-onnx", 
#     None
# )

# Get all PDF files in a directory
directory = Path("test_files")
files = list(directory.glob("*.pdf"))

# Embed all PDF files
file_embed_data = []
for file in files:
    try:
        # Embed each file (returns page-level embeddings)
        embedding = model.embed_file(str(file), batch_size=1)
        file_embed_data.extend(embedding)
    except Exception as e:
        print(f"Error embedding file {file}: {e}")

print(f"Total pages embedded: {len(file_embed_data)}")
```

## Querying Documents

```python
from embed_anything import ColpaliModel
import numpy as np
from tabulate import tabulate

# Load model
model = ColpaliModel.from_pretrained_onnx(
    "starlight-ai/colpali-v1.2-merged-onnx", 
    None
)

# Embed documents (assuming file_embed_data from previous example)
file_embeddings = np.array([e.embedding for e in file_embed_data])

# Define your query
query = "What are Positional Encodings"

# Embed the query
query_embedding = model.embed_query(query)
query_embeddings = np.array([e.embedding for e in query_embedding])

# Calculate similarity scores using einsum for efficient computation
# This computes the maximum similarity across tokens and sums them
scores = (
    np.einsum("bnd,csd->bcns", query_embeddings, file_embeddings)
    .max(axis=3)      # Max across token dimension
    .sum(axis=2)      # Sum across query tokens
    .squeeze()        # Remove singleton dimensions
)

# Get top 5 most relevant pages
top_pages = np.argsort(scores)[::-1][:5]

# Display results
table = [
    [
        file_embed_data[page].metadata["file_path"],
        file_embed_data[page].metadata["page_number"],
        f"{scores[page]:.4f}"
    ]
    for page in top_pages
]

print(tabulate(table, headers=["File Name", "Page Number", "Score"], tablefmt="grid"))

# Access page images if needed
images = [file_embed_data[page].metadata.get("image") for page in top_pages]
```

## Complete Example

``` python
--8<-- "examples/colpali.py"
```

## When to Use ColPali

- **Document search**: When you need to search through PDFs with complex layouts
- **Visual content**: Documents with tables, figures, and diagrams
- **Page-level retrieval**: When you want to retrieve entire pages rather than text chunks
- **Multimodal documents**: Documents where visual layout is important

## Performance Tips

1. **Use ONNX for production**: ONNX models are faster and use less memory
2. **Batch processing**: Process multiple files in parallel when possible
3. **Batch size**: Use `batch_size=1` for ColPali as it processes pages individually
4. **GPU acceleration**: Install `embed-anything-gpu` for GPU support
