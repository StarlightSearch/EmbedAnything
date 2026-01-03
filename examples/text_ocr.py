# OCR Requires `tesseract` and `poppler` to be installed.

import time
import embed_anything
from embed_anything import EmbedData, EmbeddingModel, TextEmbedConfig, WhichModel
from time import time


model = EmbeddingModel.from_pretrained_hf(
    model_id="jinaai/jina-embeddings-v2-small-en"
)

config = TextEmbedConfig(
    chunk_size=1000,
    batch_size=32,
    buffer_size=64,
    splitting_strategy="sentence",
    use_ocr=True,
)

start = time()

data: list[EmbedData] = embed_anything.embed_file(
    "/home/akshay/projects/starlaw/src-server/test_files/court.pdf",  # Replace with your file path
    embedder=model,
    config=config,
)
end = time()

for d in data:
    print(d.text)
    print("---" * 20)

print(f"Time taken: {end - start} seconds")
