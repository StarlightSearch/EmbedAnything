from embed_anything import embed_file, ColpaliModel
import numpy as np
model:ColpaliModel = ColpaliModel.from_pretrained("vidore/colpali-v1.2-merged", None)
embedding = model.embed_file("/home/akshay/projects/EmbedAnything/test_files/attention.pdf")
embeddings = np.array([e.embedding for e in embedding])
print(embeddings.shape)