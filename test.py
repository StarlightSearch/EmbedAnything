import os
import numpy as np
from python.embed_anything import embed_anything
from PIL import Image
import time

# start = time.time()
# data= embed_anything.embed_file("test_files/clip/cat1.jpg", embeder= "Clip")


# embeddings = np.array([data.embedding for data in data])

# print(data[0])

# query = ["Photo of a dog?"]
# query_embedding = np.array(embed_anything.embed_query(query, embeder= "Clip")[0].embedding)

# similarities = np.dot(embeddings, query_embedding)

# max_index = np.argmax(similarities)

# # Image.open(data[max_index].text).show()
# print(data[max_index].text)
# end = time.time()
# print("Time taken: ", end-start)


url = "https://www.akshaymakes.com/blogs/3d_convolution"

data = embed_anything.emb_webpage(url, embeder= "Bert")

print(data[0])