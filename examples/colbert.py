import heapq
from embed_anything import (
    EmbeddingModel,
    TextEmbedConfig,
    WhichModel,
    embed_query,
    ONNXModel,
    Dtype,
)
import os
from time import time
import numpy as np

model = EmbeddingModel.from_pretrained_onnx(
    model = WhichModel.ColBert, hf_model_id = "answerdotai/answerai-colbert-small-v1", path_in_repo="onnx/model_fp16.onnx"
)

sentences = [
    "The quick brown fox jumps over the lazy dog",
    "The cat is sleeping on the mat",
    "The dog is barking at the moon",
    "I love pizza",
    "I like to have pasta",
    "The dog is sitting in the park",
]

query = "There is a dog walking in the park"


doc_embeddings = np.array([e.embedding for e in embed_query(sentences, embedder=model)])

query_embeddings = np.array([e.embedding for e in embed_query([query], embedder=model)])


print("shape of doc_embedddings", doc_embeddings.shape)

scores = (
    np.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)
    .max(axis=3)
    .sum(axis=2)
    .squeeze()
)

for i, score in enumerate(scores):
    print(f"{sentences[i]}: {score}")

