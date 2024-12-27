import numpy as np
import embed_anything
from embed_anything import EmbedData
import time

start = time.time()

# Load the model.
model = embed_anything.EmbeddingModel.from_pretrained_hf(
    embed_anything.WhichModel.Clip,
    model_id="openai/clip-vit-base-patch16",
)
data: list[EmbedData] = embed_anything.embed_image_directory(
    "test_files", embedder=model
)

# Convert the embeddings to a numpy array
embeddings = np.array([data.embedding for data in data])

print(data[0])

# Embed a query
query = ["Photo of a monkey?"]
query_embedding = np.array(
    embed_anything.embed_query(query, embedder=model)[0].embedding
)

# Calculate the similarities between the query embedding and all the embeddings
similarities = np.dot(embeddings, query_embedding)

# Find the index of the most similar embedding
max_index = np.argmax(similarities)

# Print the most similar image
print(data[max_index].text)
end = time.time()
print("Time taken: ", end - start)
