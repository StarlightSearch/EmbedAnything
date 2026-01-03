import embed_anything
from embed_anything import EmbedData, EmbeddingModel, ONNXModel, WhichModel, embed_query
from embed_anything.vectordb import Adapter
import os
from time import time
import numpy as np
import heapq


model = EmbeddingModel.from_pretrained_hf(
    "prithivida/Splade_PP_en_v1"
)

## ONNX model
# model = EmbeddingModel.from_pretrained_onnx(
#     WhichModel.SparseBert,
#     ONNXModel.SPLADEPPENV2,
# )
sentences = [
    "The cat sits outside",
    "A man is playing guitar",
    "I love pasta",
    "The new movie is awesome",
    "The cat plays in the garden",
    "A woman watches TV",
    "The new movie is so great",
    "Do you like pizza?",
]

embedddings = embed_query(sentences, embedder=model)

embed_vector = np.array([e.embedding for e in embedddings])

similarities = np.matmul(embed_vector, embed_vector.T)

# get top 5 similarities and show the two sentences and their similarity scores
# Flatten the upper triangle of the similarity matrix, excluding the diagonal
similarity_scores = [
    (similarities[i, j], i, j)
    for i in range(len(sentences))
    for j in range(i + 1, len(sentences))
]

# Get the top 5 similarity scores
top_5_similarities = heapq.nlargest(5, similarity_scores, key=lambda x: x[0])

# Print the top 5 similarities with sentences
for score, i, j in top_5_similarities:
    print(f"Score: {score:.2} | {sentences[i]} | {sentences[j]}")
