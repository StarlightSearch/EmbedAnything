import time
import numpy as np
import embed_anything
from embed_anything import EmbedData, EmbeddingModel, TextEmbedConfig, WhichModel

# Initialize the model once
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Jina, model_id="jinaai/jina-embeddings-v2-small-en"
)

# Example 1: Embedding a Directory
def embed_directory_example():
    # Configure the embedding process
    config = TextEmbedConfig(
        chunk_size=256, batch_size=32, buffer_size=64, splitting_strategy="sentence"
    )

    # Start timing
    start = time.time()

    # Embed all files in a directory
    data: list[EmbedData] = embed_anything.embed_directory(
        "../../bench", embeder=model, config=config
    )

    # End timing
    end = time.time()

    print(f"Time taken to embed directory: {end - start} seconds")

# Example 2: Embedding a Query
def embed_query_example():
    # Configure the embedding process
    config = TextEmbedConfig(chunk_size=256, batch_size=32, splitting_strategy="sentence")

    start = time.time()

    # Embed a query
    embeddings: EmbedData = embed_anything.embed_query(
        ["Hello world my"], embeder=model, config=config
    )[0]

    end = time.time()

    print(f"Time taken to embed query: {end - start} seconds")

    # Print the shape of the embedding
    print(np.array(embeddings.embedding).shape)

# Example 3: Embedding a File
def embed_file_example():
    # Configure the embedding process
    config = TextEmbedConfig(
        chunk_size=256, batch_size=32, buffer_size=64, splitting_strategy="sentence"
    )

    start = time.time()

    # Embed a single file
    data: list[EmbedData] = embed_anything.embed_file(
        "../../test_files/bank.txt", embeder=model, config=config
    )

    end = time.time()

    print(f"Time taken to embed file: {end - start} seconds")

# Call the examples
embed_directory_example()
embed_query_example()
embed_file_example()
