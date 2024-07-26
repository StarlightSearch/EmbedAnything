import os
import re
import uuid
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
from .embed_anything import EmbedData


class PineconeAdapter:
    def __init__(self, api_key: str, index_name: str):

        self.api_key = api_key
        self.index_name = index_name
        self.pc = Pinecone(api_key=self.api_key)

    def create_index(
        self,
        dimension: int,
        metric: str = "cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    ):
        self.pc.create_index(
            name=self.index_name, dimension=dimension, metric=metric, spec=spec
        )

    def delete_index(self, index_name: str):
        self.pc.delete_index(name=index_name)

    def convert(self, embeddings: List[List[EmbedData]]) -> List[Dict]:
        data_emb = []
        for i, embedding_list in enumerate(embeddings):
            for emb in embedding_list:
                data_emb.append(
                    {
                        "id": str(uuid.uuid4()),
                        "values": emb.embedding,
                        "metadata": {
                            "text": emb.text,
                            "file": re.split(r"/|\\", emb.metadata["file_name"])[-1],
                        },
                    }
                )
        return data_emb

    def upsert(self, data: List[Dict]):

        self.pc.Index(name=self.index_name).upsert(data)
