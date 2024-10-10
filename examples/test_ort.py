import embed_anything
from embed_anything import EmbedData, EmbeddingModel, TextEmbedConfig, WhichModel
from embed_anything.vectordb import Adapter
import os
from time import time
from fastembed import TextEmbedding
import numpy as np

# model = EmbeddingModel.from_pretrained_hf(
#     WhichModel.Bert, model_id="sentence-transformers/all-MiniLM-L12-v2"
# )

model = EmbeddingModel.from_pretrained_onnx(
    WhichModel.Bert, model_id="BAAI/bge-small-en-v1.5", revision="main"
)

sentences = [
    "The quick brown fox jumps over the lazy dog",
    "The cat is sleeping on the mat",
    "The dog is barking at the moon",
    "I love pizza",
    "I like to have pasta",
    "The dog is sitting in the park",
]

embedddings = embed_anything.embed_query(sentences, embeder=model)

embed_vector = np.array([e.embedding for e in embedddings])

print("shape of embed_vector", embed_vector.shape)
similarities = np.matmul(embed_vector, embed_vector.T)

print(similarities)

model = TextEmbedding(model_name = "BAAI/bge-small-en-v1.5",
                      providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

embeddings = np.array(list(model.embed(sentences)))

print(embeddings.shape)

similarities = np.matmul(embeddings, embeddings.T)

print(similarities)
