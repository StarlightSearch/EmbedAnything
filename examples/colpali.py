from embed_anything import embed_file, EmbeddingModel, WhichModel
import numpy as np

model = EmbeddingModel.from_pretrained_hf(WhichModel.Colpali, "vidore/colpali-v1.2-merged", None)

embedding = embed_file("/home/akshay/EmbedAnything/test_files/clip/cat2.jpeg", model, None, None)

embeddings = np.array([e.embedding for e in embedding])

print(embeddings.shape)

