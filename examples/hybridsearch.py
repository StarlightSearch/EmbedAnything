from qdrant_client import QdrantClient, models
from tqdm.auto import tqdm
import embed_anything
from embed_anything import EmbedData
from embed_anything.vectordb import Adapter
import uuid
from embed_anything import EmbedData, EmbeddingModel, TextEmbedConfig, WhichModel

from typing import List
from qdrant_client.models import PointStruct


client = QdrantClient(
    "cloud.qdrant.io",
    api_key="api",
)

sentences = [
    "The cat sits outside",
    "A man is playing guitar",
    "I love pasta",
    "The new movie is awesome",
    "The cat plays in the garden",
    "A woman watches TV",
    "The new movie is so great",
    "Do you like pizza?",
]


client.create_collection(
    collection_name="my-hybrid-collection",
    vectors_config={
        "jina": models.VectorParams(
            size=768,
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "bm42": models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        )
    },
)


query_text = ["best programming language for beginners?"]


jina_model = EmbeddingModel.from_pretrained_hf(
     model_id="jinaai/jina-embeddings-v2-small-en"
)

jina_embedddings = embed_anything.embed_query(sentences, embedder=jina_model)
jina_query = embed_anything.embed_query(query_text, embedder=jina_model)[0]


splade_model = EmbeddingModel.from_pretrained_hf(
    model_id="prithivida/Splade_PP_en_v1"
)
jina_embedddings = embed_anything.embed_query(sentences, embedder=jina_model)

splade_query = embed_anything.embed_query(query_text, embedder=splade_model)

client.query_points(
    collection_name="my-hybrid-collection",
    prefetch=[
        models.Prefetch(
            query=jina_query,  # <-- dense vector
            limit=10,
        ),
        models.Prefetch(
            query=splade_query,  # <-- dense vector
            limit=10,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
)
