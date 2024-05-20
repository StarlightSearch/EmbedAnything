import embed_anything
from openai import OpenAI

import os
import time
from pinecone import Pinecone
import numpy as np



data = embed_anything.embed_directory('Vector_database_files\test_paper.pdf', embeder= "OpenAI")
embeddings = np.array([data.embedding for data in data])

print(len(data))
query= embed_anything.embed_query(["what is AI?"], embeder="OpenAI")

pc = Pinecone(api_key="")
index = pc.Index("anything")

# for i in range(len(data)):
#     index.upsert(
#         vectors=[{"id": str(i), "values": data[i].embedding, "metadata": {"text": data[i].text}}]
#     )



def retrieval(query):
    query_embedding = embed_anything.embed_query(query, embeder="OpenAI")
    return index.query(vector=query_embedding[0].embedding, top_k=2)
index.fetch(["82", "81"])

# print(retrieval(["what is AI?"]))