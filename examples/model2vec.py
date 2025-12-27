from embed_anything import EmbedData, EmbeddingModel, TextEmbedConfig

# Initialize the model once
model:EmbeddingModel = EmbeddingModel.from_pretrained_hf(
    model_id="minishlab/potion-base-8M"
)


# Example 3: Embedding a File
def embed_file_example():
    # Configure the embedding process
    config = TextEmbedConfig(
        chunk_size=1000, batch_size=32, buffer_size=64, splitting_strategy="sentence"
    )

    # Embed a single file
    data: list[EmbedData] = model.embed_file(
        "test_files/bank.txt", config=config
    )

    # Print the embedded data
    for d in data:
        print(d.text)
        print("---" * 20)

embed_file_example()