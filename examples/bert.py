import embed_anything
from embed_anything import BertConfig, EmbedConfig
import time

from embed_anything.embed_anything import JinaConfig

start_time = time.time()

bert_config = BertConfig(
    model_id="sentence-transformers/all-MiniLM-L12-v2", chunk_size=100
)

# available model_ids:
# jinaai/jina-embeddings-v2-base-en,
# jinaai/jina-embeddings-v2-small-en,
# jinaai/jina-embeddings-v2-base-zh,
# jinaai/jina-embeddings-v2-small-de
jina_config = JinaConfig(
    model_id="jinaai/jina-embeddings-v2-small-en", revision="main", chunk_size=100
)
embed_config = EmbedConfig(bert=bert_config)

data = embed_anything.embed_directory(
    "test_files", embeder="Bert", extensions=["pdf"], config=embed_config
)

data_file = embed_anything.embed_file(
    "./test_files/test.pdf", embeder="Bert", config=embed_config
)

print(data_file[0])
end_time = time.time()
print("Time taken: ", end_time - start_time)
