from embed_anything import EmbeddingModel, TextEmbedConfig, WhichModel
import numpy as np
from pathlib import Path
from tabulate import tabulate
from embed_anything import EmbedData
from pdf2image import convert_from_path


# Initialize the model once
model: EmbeddingModel = EmbeddingModel.from_pretrained_cloud(
    WhichModel.CohereVision, model_id="embed-v4.0"
)


# Get all PDF files in the directory
directory = Path("test_files")
files = directory.glob("*.pdf")
# files = [Path("test_files/attention.pdf")]

file_embed_data: list[EmbedData] = []
for file in files:
    try:
        embedding: list[EmbedData] = model.embed_file(
            str(file), TextEmbedConfig(batch_size=8)
        )
        file_embed_data.extend(embedding)
    except Exception as e:
        print(f"Error embedding file {file}: {e}")

# Define the query
query = "What are the Bleu score results for the attention paper?"

# Scoring
file_embeddings = np.array([e.embedding for e in file_embed_data])
query_embedding = model.embed_query([query])
query_embeddings = np.array([e.embedding for e in query_embedding])
print(file_embeddings.shape)
print(query_embeddings.shape)


scores = np.dot(query_embeddings, file_embeddings.T).squeeze()

# Get top pages
top_pages = np.argsort(scores)[-5:][::-1].tolist()  # Convert to list

print(top_pages)
# Extract file names and page numbers
table = [
    [
        file_embed_data[int(page)].metadata["file_path"],
        file_embed_data[int(page)].metadata["page_number"],
    ]
    for page in top_pages
]

# Print the results in a table
print(tabulate(table, headers=["File Name", "Page Number"], tablefmt="grid"))

images = [file_embed_data[int(page)].metadata["image"] for page in top_pages]
