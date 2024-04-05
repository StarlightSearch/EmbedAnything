import os
os.add_dll_directory(r'D:\libtorch\lib')

import embed_anything
data = embed_anything.embed_file("test_files/TUe_SOP_AI_2.pdf", embeder= "AllMiniLmL12V2")

print(data[0])