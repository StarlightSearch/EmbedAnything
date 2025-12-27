"""
Example: Fetching files from AWS S3 and embedding them

This example shows how to:
1. Create an S3Client with AWS credentials
2. Fetch files from S3 buckets
3. Embed the downloaded files
"""

import embed_anything
from embed_anything import S3Client, EmbeddingModel, WhichModel, TextEmbedConfig
import os

# Step 1: Create S3Client with explicit credentials
print("Example 1: Create S3Client with credentials")
s3_client = S3Client(
    access_key_id="",
    secret_access_key="",
    region=""
)
print(f"S3Client created: {s3_client}")

file = s3_client.get_file_from_s3(bucket_name="embed-anything", key="test.txt").save_file()
print(f"File: {file}")

# Step 2: Embed the file
embedder = EmbeddingModel.from_pretrained_hf(
    model_id="jinaai/jina-embeddings-v2-small-en"
)
embeddings = embedder.embed_file(file, config=TextEmbedConfig(chunk_size=1000, batch_size=32, buffer_size=64, splitting_strategy="sentence"))
print(f"Embeddings: {embeddings}")

# Step 3: Embed a directory
directory = "bench"
embeddings = embedder.embed_directory(directory, config=TextEmbedConfig(chunk_size=1000, batch_size=32, buffer_size=64, splitting_strategy="sentence"))
print(f"Embeddings: {embeddings}")