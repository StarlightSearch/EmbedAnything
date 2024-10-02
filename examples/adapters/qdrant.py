from qdrant_client import QdrantClient, models
from tqdm.auto import tqdm
import embed_anything
from embed_anything import EmbedData
from embed_anything.vectordb import Adapter
import uuid

from typing import List
from qdrant_client.models import PointStruct


class QdrantAdapter(Adapter):
    def __init__(self, api_key, url):
        super().__init__(api_key)
        self.qdrant_client = QdrantClient(
            url,
            api_key=api_key,
        )

    def create_index(self, index_name: str, vectors_config):
        self.index_name = index_name
        self.qdrant_client.create_collection(
            index_name,
            vectors_config
        )
        return self.index_name

    def convert(self, embeddings: List[EmbedData]) -> List[PointStruct]:

        points = []
        for embedding in embeddings:
            payload = embedding.metadata
            payload["text"] = embedding.text
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()), vector=embedding.embedding, payload=payload
                )
            )

        return points

    def upsert(self, data_):
        data_ = self.convert(data_)
        self.qdrant_client.upsert(
            collection_name=self.index_name,
            points=data_
        )

    def delete_index(self, index_name: str):
        self.qdrant_client.delete_collection(index_name)



qdrant_client = QdrantClient(
    "cloud.qdrant.io",
    api_key="api",
)

# create index

index_name = "Text10_image"
if index_name in qdrant_client.get_collections():
    qdrant_client.delete_collection(collection_name=index_name)



qdrant_client.create_collection(index_name, vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE))

model = embed_anything.EmbeddingModel.from_pretrained_hf(
    embed_anything.WhichModel.Clip,
    model_id="openai/clip-vit-base-patch16",
    # revision="refs/pr/15",
)


data= embed_anything.embed_image_directory(
    "../../test_files/clip", embeder=model, adapter=QdrantAdapter("api", "cloud.qdrant.io")

)


