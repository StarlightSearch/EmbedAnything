import embed_anything
from embed_anything import EmbeddingModel, TextEmbedConfig, WhichModel

model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Jina, model_id="jinaai/jina-embeddings-v2-small-en"
)

# with semantic encoder
semantic_encoder = EmbeddingModel.from_pretrained_hf(
    WhichModel.Jina, model_id="jinaai/jina-embeddings-v2-small-en"
)
config = TextEmbedConfig(
    chunk_size=1000,
    batch_size=32,
    splitting_strategy="semantic",
    semantic_encoder=semantic_encoder,
)

data = embed_anything.embed_file("test_files/bank.txt", embedder=model, config=config)

for d in data:
    print(d.text)
    print("---" * 20)
