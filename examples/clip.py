import os
import numpy as np
import embed_anything as embed_anything
from embed_anything import EmbedData
from PIL import Image
import time

start = time.time()

clip_config = embed_anything.ClipConfig(
    model_id="openai/clip-vit-base-patch32", revision="refs/pr/15"
)

config = embed_anything.EmbedConfig(clip=clip_config)

data: list[EmbedData] = embed_anything.embed_directory(
    "test_files", embeder="Clip", config=config
)

embeddings = np.array([data.embedding for data in data])

print(data[0])

query = ["Photo of a monkey?"]
query_embedding = np.array(
    embed_anything.embed_query(query, embeder="Clip")[0].embedding
)

similarities = np.dot(embeddings, query_embedding)

max_index = np.argmax(similarities)

# Image.open(data[max_index].text).show()
print(data[max_index].text)
end = time.time()
print("Time taken: ", end - start)
