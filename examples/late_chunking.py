import heapq
from embed_anything import EmbedData, EmbeddingModel, TextEmbedConfig, WhichModel

from embed_anything import Dtype, ONNXModel
import numpy as np


model:EmbeddingModel = EmbeddingModel.from_pretrained_hf(
    WhichModel.Jina, model_id="jinaai/jina-embeddings-v2-small-en"
)
config = TextEmbedConfig(
    chunk_size=1000,
    batch_size=8,
    splitting_strategy="sentence",
    late_chunking=False,
)

# Embed a single file
data: list[EmbedData] = model.embed_file("test_files/attention.pdf", config=config)

# Print the embedded data
# for d in data:
#     print(d.text)
#     print("---" * 20)

query = "What are positional encodings?"

query_embedding = np.array(model.embed_query([query])[0].embedding)

embedding_array = np.array([e.embedding for e in data])

similarities = np.matmul(query_embedding, embedding_array.T)

# get top 5 similarities and its index
top_5_similarities = np.argsort(similarities)[-10:][::-1]

# Print the top 5 similarities with sentences
for i in top_5_similarities:
    print(f"Score: {similarities[i]:.2} | {data[i].text}")
    print("---" * 20)
