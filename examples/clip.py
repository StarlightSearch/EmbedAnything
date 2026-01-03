import numpy as np
import embed_anything
from embed_anything import EmbedData
import time

start = time.time()

# Load the model.
model = embed_anything.EmbeddingModel.from_pretrained_hf(
    model_id="google/siglip-base-patch16-224",
)
data: list[EmbedData] = embed_anything.embed_image_directory(
    "test_files", embedder=model
)

# Convert the embeddings to a numpy array
embeddings = np.array([data.embedding for data in data])

# Embed a query
query = ["Photo of a monkey"]
query_embedding = np.array(
    embed_anything.embed_query(query, embedder=model)[0].embedding
)

# Calculate the similarities between the query embedding and all the embeddings
similarities = np.dot(embeddings, query_embedding)

# Find the index of the most similar embedding
max_index = np.argmax(similarities)

print("Descending order of similarity: ")
indices = np.argsort(similarities)[::-1]
for idx in indices:
    print(data[idx].text)

print("----------- ")

# Print the most similar image
print("Most similar image: ", data[max_index].text)
end = time.time()
print("Time taken: ", end - start)
