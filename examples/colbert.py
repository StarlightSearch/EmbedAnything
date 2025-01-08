
from embed_anything import (
    embed_query,
    ColbertModel
)
import os
from time import time
import numpy as np

model:ColbertModel = ColbertModel.from_pretrained_onnx(
    hf_model_id="jinaai/jina-colbert-v2",
    path_in_repo="onnx/model.onnx",
)

# model:ColbertModel = ColbertModel.from_pretrained_onnx(
#     hf_model_id="answerdotai/answerai-colbert-small-v1",
#     path_in_repo="onnx/model_fp16.onnx",
# )


sentences = [
    "The quick brown fox jumps over the lazy dog",
    "The cat is sleeping on the mat",
    "The dog is barking at the moon",
    "I love pizza",
    "I like to have pasta",
    "The dog is sitting in the park",
]

query = "I like italian food"

doc_embeddings = np.array([e.embedding for e in model.embed(sentences, is_doc=True)])

query_embeddings = np.array([e.embedding for e in model.embed([query], is_doc=False)])

print("shape of doc_embedddings", doc_embeddings.shape)

scores = (
    np.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)
    .max(axis=3)
    .sum(axis=2)
    .squeeze()
)

for i, score in enumerate(scores):
    print(f"{sentences[i]}: {score}")

