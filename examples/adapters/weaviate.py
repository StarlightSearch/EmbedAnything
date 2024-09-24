import weaviate, os
import weaviate.classes as wvc
from tqdm.auto import tqdm
import embed_anything
from embed_anything import EmbedData
from embed_anything.vectordb import Adapter

from typing import List


class WeaviateAdapter(Adapter):
    def __init__(self, api_key, url):
        super().__init__(api_key)
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url, auth_credentials=wvc.init.Auth.api_key(api_key)
        )
        if self.client.is_ready():
            print("Weaviate is ready")

    def create_index(self, index_name: str):
        self.index_name = index_name
        self.collection = self.client.collections.create(
            index_name, vectorizer_config=wvc.config.Configure.Vectorizer.none()
        )
        return self.collection

    def convert(self, embeddings: List[EmbedData]):
        data = []
        for embedding in embeddings:
            property = embedding.metadata
            property["text"] = embedding.text
            data.append(
                wvc.data.DataObject(properties=property, vector=embedding.embedding)
            )
        return data

    def upsert(self, data_):
        data_=self.convert(data_)
        self.client.collections.get(self.index_name).data.insert_many(data_)

    def delete_index(self, index_name: str):
        self.client.collections.delete(index_name)


URL = "https://jp5vpgytqsm6pp4xyavkqg.c0.europe-west3.gcp.weaviate.cloud"
API_KEY = "FKTRwiaayVV1kdJhWjR7NRk8PrK98XMY3xlU"
weaviate_adapter = WeaviateAdapter(API_KEY, URL)

index_name = "Test_index"
if index_name in weaviate_adapter.client.collections.list_all():
    weaviate_adapter.delete_index(index_name)
weaviate_adapter.create_index("Test_index")


model = embed_anything.EmbeddingModel.from_pretrained_hf(
    embed_anything.WhichModel.Clip,
    model_id="openai/clip-vit-base-patch16",
    # revision="refs/pr/15",
)


data= embed_anything.embed_image_directory(
    "../../test_files/clip", embeder=model, adapter=weaviate_adapter
)


query_vector = embed_anything.embed_query(["image of a cat"], embeder=model)[
    0
].embedding

response = weaviate_adapter.collection.query.near_vector(
    near_vector=query_vector,
    limit=2,
    return_metadata=wvc.query.MetadataQuery(certainty=True),
)

for i in range(len(response.objects)):
    print(response.objects[i].properties["text"])
