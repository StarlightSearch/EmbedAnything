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
    model = WhichModel.Bert, model_name=ONNXModel.ModernBERTBase, dtype = Dtype.Q4F16
)

# model = EmbeddingModel.from_pretrained_hf(
#     model = WhichModel.Bert, model_name="BAAI/bge-small-en-v1.5"
# )

sentences = [
    "The quick brown fox jumps over the lazy dog",
    "The cat is sleeping on the mat",
    "The dog is barking at the moon",
    "I love pizza",
    "I like to have pasta",
    "The dog is sitting in the park",
]

embedddings = embed_query(sentences, embedder=model)

embed_vector = np.array([e.embedding for e in embedddings])

print("shape of embed_vector", embed_vector.shape)
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


from embed_anything import EmbeddingModel, WhichModel, embed_query, TextEmbedConfig
import os
import pymupdf
from semantic_text_splitter import TextSplitter
import os

model = EmbeddingModel.from_pretrained_onnx(WhichModel.Bert, ONNXModel.BGESmallENV15Q)
splitter = TextSplitter(1000)
config = TextEmbedConfig(batch_size=128)


def embed_anything():
    # get all pdfs from test_files

    for file in os.listdir("bench"):
        text = []
        doc = pymupdf.open("bench/" + file)

        for page in doc:
            text.append(page.get_text())

        text = " ".join(text)
        chunks = splitter.chunks(text)
        embeddings = embed_query(chunks, model, config)


start = time()
embed_anything()

print(time() - start)
