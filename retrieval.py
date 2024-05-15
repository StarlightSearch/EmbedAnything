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

for i in range(len(data)):
    index.upsert(
        vectors=[{"id": str(i), "values": data[i].embedding, "metadata": {"text": data[i].text}}]
    )



def retrieval(query):
    query_embedding = embed_anything.embed_query(query, embeder="OpenAI")
    context =  index.query(vector=query_embedding[0].embedding, top_k=2)
    indices = [int(context.matches[i]['id']) for i in range(len(context.matches))]
    return indices

indices = retrieval(["what is AI?"])

def get_text(indices):
    return [index.fetch([str(e)])['vectors'][str(e)]['metadata']['text'] for e in indices]



content = query + " "
for i in range(min(len(indices), 3)):
    content += get_text(indices)[i] + " "



client = OpenAI()



response = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": content}
  ]
)
print(response.choices[0].message.content)