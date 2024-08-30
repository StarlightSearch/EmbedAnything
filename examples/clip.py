import os
import numpy as np
import embed_anything
from embed_anything import EmbedData
from PIL import Image
import time

start = time.time()

model = embed_anything.EmbeddingModel.from_pretrained_local(
    embed_anything.WhichModel.Clip,
    model_id="openai/clip-vit-base-patch32",
    revision="refs/pr/15",
)
data: list[EmbedData] = embed_anything.embed_directory("test_files", embeder=model)

embeddings = np.array([data.embedding for data in data])

print(data[0])

query = ["Photo of a monkey?"]
query_embedding = np.array(
    embed_anything.embed_query(query, embeder=model)[0].embedding
)

similarities = np.dot(embeddings, query_embedding)

max_index = np.argmax(similarities)

# Image.open(data[max_index].text).show()
print(data[max_index].text)
end = time.time()
print("Time taken: ", end - start)
