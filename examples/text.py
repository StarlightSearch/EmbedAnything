import time
import numpy as np
import embed_anything
from embed_anything import EmbedData, EmbeddingModel, TextEmbedConfig, WhichModel

# Initialize the model once
model:EmbeddingModel = EmbeddingModel.from_pretrained_hf(
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
    data: list[EmbedData] = model.embed_directory(
        "bench", config=config
    )

    # End timing
    end = time.time()

    print(f"Time taken to embed directory: {end - start} seconds")


# Example 2: Embedding a Query
def embed_query_example():
    # Configure the embedding process
    config = TextEmbedConfig(
        chunk_size=256, batch_size=32, splitting_strategy="sentence"
    )

    # Embed a query
    embeddings: EmbedData = model.embed_query(
        ["Hello world my"], config=config
    )[0]

    # Print the shape of the embedding
    print(np.array(embeddings.embedding).shape)

    # Embed another query and print the result
    print(
        embed_anything.embed_query(
            ["What is the capital of India?"], embedder=model, config=config
        )
    )


# Example 3: Embedding a File
def embed_file_example():
    # Configure the embedding process
    config = TextEmbedConfig(
        chunk_size=256, batch_size=32, buffer_size=64, splitting_strategy="sentence"
    )

    # Embed a single file
    data: list[EmbedData] = model.embed_file(
        "test_files/bank.txt", config=config
    )

    # Print the embedded data
    for d in data:
        print(d.text)
        print("---" * 20)


# Call the examples
embed_directory_example()
embed_query_example()
embed_file_example()
