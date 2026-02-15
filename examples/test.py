from embed_anything import EmbeddingModel, WhichModel, TextEmbedConfig
import embed_anything

# Load a custom BERT model from Hugging Face
model = EmbeddingModel.from_pretrained_hf(
    model_id="sentence-transformers/all-MiniLM-L12-v2"
)

# Configure embedding parameters
config = TextEmbedConfig(
    chunk_size=1000,      # Maximum characters per chunk
    batch_size=32,        # Number of chunks to process in parallel
    splitting_strategy="sentence"  # How to split text: "sentence", "word", or "semantic"
)

# Embed a file (supports PDF, TXT, MD, etc.)
data = embed_anything.embed_file("/home/sonamAI/projects/EmbedAnything/bench/attention.pdf", embedder=model, config=config)

# Access the embeddings and text
for item in data:
    print(f"Text: {item.text[:100]}...")  # First 100 characters
    print(f"Embedding shape: {len(item.embedding)}")
    print(f"Metadata: {item.metadata}")
    print("---" * 20)