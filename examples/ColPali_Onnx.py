### ColPali (Document Embedding)

#ColPali is optimized for document and image-text embedding tasks.

from embed_anything import ColpaliModel
import numpy as np

# Load ColPali ONNX model
model = ColpaliModel.from_pretrained_onnx(
    "starlight-ai/colpali-v1.2-merged-onnx", 
    None
)

# Embed a PDF file (ColPali processes pages as images)
data = model.embed_file("test_files/document.pdf", batch_size=1)

# Query the embedded document
query = "What is the main topic?"
query_embedding = model.embed_query(query)

# Calculate similarity scores
file_embeddings = np.array([e.embedding for e in data])
query_emb = np.array([e.embedding for e in query_embedding])

# Find most relevant pages
scores = np.einsum("bnd,csd->bcns", query_emb, file_embeddings).max(axis=3).sum(axis=2).squeeze()
top_pages = np.argsort(scores)[::-1][:5]

for page_idx in top_pages:
    print(f"Page {data[page_idx].metadata['page_number']}: {data[page_idx].text[:200]}")
