import embed_anything
import time 
start_time = time.time()
data = embed_anything.embed_directory("test_files", embeder= "Bert", extensions=["pdf"])
print(data[0])
end_time = time.time() 
print("Time taken: ", end_time-start_time)