import embed_anything

model = embed_anything.EmbeddingModel.from_pretrained_hf(
    model_id="sentence-transformers/all-MiniLM-L6-v2"
)
data = embed_anything.embed_webpage("https://www.akshaymakes.com/", embedder=model)

print(data)