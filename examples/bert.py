import embed_anything
from embed_anything import BertConfig, EmbedConfig
import time

start_time = time.time()

bert_config = BertConfig(
    model_id="sentence-transformers/all-MiniLM-L12-v2", chunk_size=100
)
embed_config = EmbedConfig(bert=bert_config)

data = embed_anything.embed_directory("test_files", embeder="Bert", extensions=["pdf"])

data_file = embed_anything.embed_file(
    "./test_files/test.pdf", embeder="Bert", config=embed_config
)

print(data_file[0])
end_time = time.time()
print("Time taken: ", end_time - start_time)
