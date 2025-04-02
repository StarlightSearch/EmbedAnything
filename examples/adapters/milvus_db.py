
from pymilvus import connections, utility, FieldSchema, CollectionSchema, Collection, DataType
import os
from typing import Dict, List

import embed_anything
from embed_anything.vectordb import Adapter
from embed_anything import EmbedData


print("Milvus Vector DB - Adapter")

# default milvus docker image address & port
MILVUS_DB_HOST_ADDRESS = os.environ.get("MILVUS_DB_HOST_ADDRESS", "127.0.0.1")
MILVUS_DB_PORT = os.environ.get("MILVUS_DB_PORT", "19530")
MILVUS_COLLECTION_NAME = "MILVUS_EA_ADAPTER_COLLECTION"

EMBEDDINGS_DIM = 384

# https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
SENT_TF_MODEL_ID = "sentence-transformers/all-MiniLM-L12-v2"
EXAMPLE_FILE = "test_files/attention.pdf"


# typecheck
VectorEmbeddings = List[List[EmbedData]]


class MilvusVectorAdapter(Adapter):
    def __init__(self, host_address: str, host_port: int):
        connections.connect(host=host_address, port=host_port)
        self.milvus_collection_name = MILVUS_COLLECTION_NAME
        print("Ok - Milvus DB connection.")
        if utility.has_collection(self.milvus_collection_name):
            utility.drop_collection(self.milvus_collection_name)

        self.schema = self.define_schema()
        self.db = None # Milvus DB table/collect
    
    def define_schema(self):
        pk_field = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )
        embedding_field = FieldSchema(
            name="embeddings",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDINGS_DIM,
            description="vector embeddings"
        )
        text_field = FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=1098,
            description="text content"
        )
        file_name_field = FieldSchema(
            name="file_name",
            dtype=DataType.VARCHAR,
            max_length=255,
            description="Name of the file"
        )
        modified_field = FieldSchema(
            name="modified",
            dtype=DataType.VARCHAR,
            max_length=50,
            description="Last modified timestamp"
        )
        created_field = FieldSchema(
            name="created",
            dtype=DataType.VARCHAR,
            max_length=50,
            description="Created timestamp"
        )
        fields = [pk_field, embedding_field, text_field, file_name_field, modified_field, created_field]
        return CollectionSchema(fields=fields)

    
    def create_index(self):
        index_params = {
            "metric_type": "L2",
            # https://milvus.io/docs/ivf-flat.md
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.db = Collection(name=self.milvus_collection_name, schema=self.schema)
        # create index on the field to do the search query on.
        self.db.create_index(field_name="embeddings", index_params=index_params)

    
    def convert(self, embeddings: VectorEmbeddings) -> List[Dict]:
        ret_data = []
        for i, embedding in enumerate(embeddings):
            print(f"Ok - #{i} converting")
            dict = {
                "embeddings": embedding.embedding,
                "text": embedding.text,
                "file_name": embedding.metadata["file_name"],
                "modified": embedding.metadata["modified"],
                "created": embedding.metadata["created"],
            }
            # row wise append
            ret_data.append(dict)
        return ret_data
    
    def delete_index(self):
        pass

    def upsert(self, data: EmbedData):
        # rust lib outputs the embeddings here during embed_anything.embed_file()
        data = self.convert(data)
        self.db.insert(data)
        self.db.flush()
        print("Ok - vector embeddings inserted.")



if __name__ == "__main__":
    milvusAdapter = MilvusVectorAdapter(MILVUS_DB_HOST_ADDRESS, MILVUS_DB_PORT)
    milvusAdapter.create_index()

    # steps
    # 1. setup model and adapter
    # 2. create index
    # 3. embed and store operation using embed_anything.embed_file
    # 4. show example search

    transfomer_model = embed_anything.EmbeddingModel.from_pretrained_hf(
        embed_anything.WhichModel.Bert, 
        model_id=SENT_TF_MODEL_ID
    )

    data = embed_anything.embed_file(
        EXAMPLE_FILE,
        embedder=transfomer_model,
        adapter=milvusAdapter,
    )


