import embed_anything
import os

from typing import Dict, List
from embed_anything import EmbedData
from embed_anything.vectordb import Adapter
from embed_anything import EmbedData, EmbeddingModel, TextEmbedConfig, WhichModel

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class ElasticsearchAdapter(Adapter):

    def __init__(self, api_key: str, cloud_id: str, index_name: str = "anything"):
        self.es = Elasticsearch(cloud_id=cloud_id, api_key=api_key)
        self.index_name = index_name

    def create_index(
        self, dimension: int, metric: str, mappings={}, settings={}, **kwargs
    ):

        if "index_name" in kwargs:
            self.index_name = kwargs["index_name"]

        self.es.indices.create(
            index=self.index_name, mappings=mappings, settings=settings
        )

    def convert(self, embeddings: List[List[EmbedData]]) -> List[Dict]:
        data = []
        for embedding in embeddings:
            data.append(
                {
                    "text": embedding.text,
                    "embeddings": embedding.embedding,
                    "metadata": {
                        "file_name": embedding.metadata["file_name"],
                        "modified": embedding.metadata["modified"],
                        "created": embedding.metadata["created"],
                    },
                }
            )
        return data

    def delete_index(self, index_name: str):
        self.es.indices.delete(index=index_name)

    def gendata(self, data):
        for doc in data:
            yield doc

    def upsert(self, data: List[Dict]):
        data = self.convert(data)
        bulk(client=self.es, index="anything", actions=self.gendata(data))


index_name = "anything"
elastic_api_key = os.environ.get("ELASTIC_API_KEY")
elastic_cloud_id = os.environ.get("ELASTIC_CLOUD_ID")

# Initialize the ElasticsearchAdapter Class
elasticsearch_adapter = ElasticsearchAdapter(
    api_key=elastic_api_key,
    cloud_id=elastic_cloud_id,
    index_name=index_name,
)

# Prase PDF and insert documents into Elasticsearch.
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Bert, model_id="sentence-transformers/all-MiniLM-L12-v2"
)


data = embed_anything.embed_file(
    "/home/sonamAI/projects/EmbedAnything/test_files/attention.pdf",
    embedder=model,
    adapter=elasticsearch_adapter
)

# Create an Index with explicit mappings.
mappings = {
    "properties": {
        "embeddings": {"type": "dense_vector", "dims": 384},
        "text": {"type": "text"},
    }
}
settings = {}

elasticsearch_adapter.create_index(
    dimension=384,
    metric="cosine",
    mappings=mappings,
    settings=settings,
)

# Delete an Index
elasticsearch_adapter.delete_index(index_name=index_name)
