import time
import embed_anything
from embed_anything.embed_anything import BertConfig, EmbedConfig, JinaConfig


jina_models_dict = [
    {"model_id": "jinaai/jina-embeddings-v2-base-en", "revision":"main", "chunk_size":100},
    {"model_id": "jinaai/jina-embeddings-v2-small-en", "revision":"main", "chunk_size":100}
]

Bert_models_dict = [
    {"model_id": "sentence-transformers/all-MiniLM-L6-v2", "revision":"main", "chunk_size":100},
    {"model_id": "sentence-transformers/all-MiniLM-L12-v2", "revision":"main", "chunk_size":100},
    {"model_id": "sentence-transformers/paraphrase-MiniLM-L6-v2", "revision":"main", "chunk_size":100},
]



# for model in jina_models_dict:
#     start_time = time.time()
#     jina_config = JinaConfig(
#         model_id=model["model_id"], revision=model["revision"], chunk_size=model["chunk_size"]
#     )
#     embed_config = EmbedConfig(jina=jina_config)

#     data_file = embed_anything.embed_file(
#         "./test_files/test.pdf", embeder="Jina", config=embed_config
#     )
    
#     end_time = time.time()
#     print("Time taken: ", end_time - start_time)

for model in Bert_models_dict:
    start_time = time.time()
    print(model)
    bert_config = BertConfig(
        model_id=model["model_id"], revision=model["revision"], chunk_size=model["chunk_size"]
    )
    embed_config = EmbedConfig(bert=bert_config)

    data_file = embed_anything.embed_file(
        "./test_files/test.pdf", embeder="Bert", config=embed_config
    )
    
    end_time = time.time()
    print("Time taken: ", end_time - start_time)

