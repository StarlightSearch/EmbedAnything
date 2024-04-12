import os
import time
import numpy as np
os.add_dll_directory(r'D:\libtorch\lib')
from embed_anything import EmbedData
import embed_anything
from PIL import Image
# start = time.time()
# data:list[EmbedData] = embed_anything.embed_file("test_files/TUe_SOP_AI_2.pdf", embeder= "Bert")

# embeddings = np.array([data.embedding for data in data])

# end = time.time()

# print(embeddings)
# print("Time taken: ", end-start)


data:list[EmbedData] = embed_anything.embed_directory("test_files", embeder= "Clip")

embeddings = np.array([data.embedding for data in data])

print(data[0])

query = ["Photo of a dog?"]
query_embedding = np.array(embed_anything.embed_query(query, embeder= "Clip")[0].embedding)

similarities = np.dot(embeddings, query_embedding)

max_index = np.argmax(similarities)

Image.open(data[max_index].text).show()
print(data[max_index].text)
