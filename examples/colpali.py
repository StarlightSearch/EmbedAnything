from embed_anything import EmbedData, ColpaliModel
import numpy as np
from tabulate import tabulate
from pathlib import Path


# Load the model
model: ColpaliModel = ColpaliModel.from_pretrained("vidore/colpali-v1.2-merged", None)

# Load ONNX Model
# model: ColpaliModel = ColpaliModel.from_pretrained_onnx(
#     "starlight-ai/colpali-v1.2-merged-onnx", None
# )

# Get all PDF files in the directory
directory = Path("test_files")
files = list(directory.glob("*.pdf"))
# files = [Path("test_files/attention.pdf")]

file_embed_data: list[EmbedData] = []
for file in files:
    try:
        embedding: list[EmbedData] = model.embed_file(str(file), batch_size=1)
        file_embed_data.extend(embedding)
    except Exception as e:
        print(f"Error embedding file {file}: {e}")

# Define the query
query = "What are Positional Encodings"

# Scoring
file_embeddings = np.array([e.embedding for e in file_embed_data])
query_embedding = model.embed_query(query)
query_embeddings = np.array([e.embedding for e in query_embedding])
print(file_embeddings.shape)
print(query_embeddings.shape)

scores = (
    np.einsum("bnd,csd->bcns", query_embeddings, file_embeddings)
    .max(axis=3)
    .sum(axis=2)
    .squeeze()
)

# Get top pages
top_pages = np.argsort(scores)[::-1][:5]

# Extract file names and page numbers
table = [
    [
        file_embed_data[page].metadata["file_path"],
        file_embed_data[page].metadata["page_number"],
    ]
    for page in top_pages
]

# Print the results in a table
print(tabulate(table, headers=["File Name", "Page Number"], tablefmt="grid"))

images = [file_embed_data[page].metadata["image"] for page in top_pages]
