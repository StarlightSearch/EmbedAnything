import embed_anything
import os

from embed_anything.vectordb import PineconeAdapter


# Initialize the PineconeEmbedder class
api_key = os.environ.get("PINECONE_API_KEY")
index_name = "anything"
pinecone_adapter = PineconeAdapter(api_key)

try:
    pinecone_adapter.delete_index("anything")
except:
    pass

# Initialize the PineconeEmbedder class

pinecone_adapter.create_index(dimension=1536, metric="cosine")

# Embed the audio files
# Replace the line with a valid code snippet or remove it if not needed
data = embed_anything.embed_file(
    "test_files/test.pdf", embeder="OpenAI", adapter=pinecone_adapter
)
