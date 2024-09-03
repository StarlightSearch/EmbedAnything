import re
from typing import Dict, List
import uuid
import time
import embed_anything
from embed_anything import EmbedData, EmbeddingModel, TextEmbedConfig, WhichModel
from embed_anything.vectordb import Adapter
from pinecone import Pinecone, ServerlessSpec
import os
from time import time


model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Bert, model_id="sentence-transformers/all-MiniLM-L12-v2"
)
config = TextEmbedConfig(chunk_size=512, batch_size=32)

start = time()
# data = embed_anything.embed_file("test_files/test.pdf", embeder=model, config=config)

data: list[EmbedData] = embed_anything.embed_directory(
    "test_files", embeder=model, config=config
)

end = time()

# for i, d in enumerate(data):
#     print(f"Chunk {i+1}\n")
#     print("-" * 100)

#     print(d.text)
#     print("Metadata\n")
#     print(d.metadata)
#     print("-" * 100)
#     print("\n")

print(f"Time taken: {end - start} seconds")
