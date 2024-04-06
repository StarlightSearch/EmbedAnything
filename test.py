import os
import time
import numpy as np
os.add_dll_directory(r'D:\libtorch\lib')
from embed_anything import EmbedData
import embed_anything
# start = time.time()
# data:list[EmbedData] = embed_anything.embed_file("test_files/TUe_SOP_AI_2.pdf", embeder= "Bert")

# embeddings = np.array([data.embedding for data in data])

# end = time.time()

# print(embeddings)
# print("Time taken: ", end-start)


data:list[EmbedData] = embed_anything.embed_directory("test_files", embeder= "Bert")

embeddings = np.array([data.embedding for data in data])

print(data[0])