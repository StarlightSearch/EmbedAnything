import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
import embed_anything
from embed_anything import EmbedData, EmbeddingModel, WhichModel
from embed_anything.vectordb import Adapter


class QdrantAdapter(Adapter):
    """
    Adapter class for interacting with [Qdrant](https://qdrant.tech/).
    """

    def __init__(self, client: QdrantClient):
        """
        Initializes a new instance of the QdrantAdapter class.

        Args:
            client : An instance of qdrant_client.QdrantClient
        """
        self.client = client

    def create_index(
        self,
        dimension: int,
        metric: Distance = Distance.COSINE,
        index_name: str = "embed-anything",
        **kwargs,
    ):
        self.collection_name = index_name

        if not self.client.collection_exists(index_name):
            self.client.create_collection(
                collection_name=index_name,
                vectors_config=VectorParams(size=dimension, distance=metric),
            )

    def delete_index(self, index_name: str):
        self.client.delete_collection(collection_name=index_name)

    def convert(self, embeddings: List[EmbedData]) -> List[PointStruct]:
        points = []
        for embedding in embeddings:
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.embedding,
                    payload={
                        "text": embedding.text,
                        "file_name": embedding.metadata["file_name"],
                        "modified": embedding.metadata["modified"],
                        "created": embedding.metadata["created"],
                    },
                )
            )
        return points

    def upsert(self, data: List[Dict]):
        points = self.convert(data)
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )


def main():
    adapter = QdrantAdapter(QdrantClient(location=":memory:"))
    adapter.create_index(dimension=384)

    model = EmbeddingModel.from_pretrained_hf(
        WhichModel.Bert, model_id="sentence-transformers/all-MiniLM-L12-v2"
    )

    embed_anything.embed_file(
        "test_files/attention.pdf",
        embedder=model,
        adapter=adapter,
    )


if __name__ == "__main__":
    main()
