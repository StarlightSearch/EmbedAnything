
### ColBERT (Late-Interaction Embeddings)

#ColBERT provides token-level embeddings for fine-grained semantic matching.


from embed_anything import ColbertModel
import numpy as np

# Load ColBERT ONNX model
model = ColbertModel.from_pretrained_onnx(
    "jinaai/jina-colbert-v2", 
    path_in_repo="onnx/model.onnx"
)

# Embed sentences
sentences = [
    "The quick brown fox jumps over the lazy dog", 
    "The cat is sleeping on the mat", 
    "The dog is barking at the moon", 
    "I love pizza", 
    "The dog is sitting in the park"
]

# ColBERT returns token-level embeddings
embeddings = model.embed(sentences, batch_size=2)

# Each embedding is a matrix: [num_tokens, embedding_dim]
for i, emb in enumerate(embeddings):
    print(f"Sentence {i+1}: {sentences[i]}")
    print(f"Embedding shape: {emb.shape}")  # Shape: (num_tokens, embedding_dim)
