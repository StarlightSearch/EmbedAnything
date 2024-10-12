import time
import embed_anything
from embed_anything import EmbedData, EmbeddingModel, TextEmbedConfig, WhichModel
from embed_anything.vectordb import Adapter
import os
from time import time


# model = EmbeddingModel.from_pretrained_hf(
#     WhichModel.Bert, model_id="sentence-transformers/all-MiniLM-L12-v2"
# )

model = EmbeddingModel.from_pretrained_onnx(
    WhichModel.Bert, model_id="BAAI/bge-small-en-v1.5", revision="main"
)

# with semantic encoder
# semantic_encoder = EmbeddingModel.from_pretrained_hf(WhichModel.Jina, model_id = "jinaai/jina-embeddings-v2-small-en")
# config = TextEmbedConfig(chunk_size=256, batch_size=32, splitting_strategy = "semantic", semantic_encoder=semantic_encoder)

# without semantic encoder
config = TextEmbedConfig(chunk_size=256, batch_size=32, buffer_size  = 64,splitting_strategy = "sentence")

# data = embed_anything.embed_file("test_files/bank.txt", embeder=model, config=config)

# for d in data:
#     print(d.text)
#     print("---"*20)
start = time()

data: list[EmbedData] = embed_anything.embed_directory(
    "bench", embeder=model, config=config
)

end = time()

print(embed_anything.embed_query(["What is the capital of India?"], embeder=model, config=config))
print(f"Time taken: {end - start} seconds")
