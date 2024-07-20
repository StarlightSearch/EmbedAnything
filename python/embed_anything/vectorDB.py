import os
import re
import uuid
from typing import List, Dict
from pinecone import Pinecone
from .embed_anything import EmbedData
import numpy as np



class PineconeAdapter:
    def __init__(self, api_key: str, index_name: str, dimension: int, metric: str):
        
        self.api_key = api_key
        self.index_name = index_name

        

  

    def convert_to_pinecone_format(self, embeddings: List[List[EmbedData]]) -> List[Dict]:
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

    def upsert_embeddings(self, data: List[Dict]):

        self.pc = Pinecone(api_key=self.api_key)

        self.pc.Index(name=self.index_name).upsert(data)
        