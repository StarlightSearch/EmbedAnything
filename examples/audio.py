import embed_anything
import time

start_time = time.time()
data = embed_anything.embed_file(
    "test_files/audio/samples_hp0.wav", embeder="Whisper-Jina"
)
print(data[0].metadata)
end_time = time.time()
print("Time taken: ", end_time - start_time)
