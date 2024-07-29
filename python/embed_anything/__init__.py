"""
This module provides functions and classes for embedding queries, files, and directories using different embedding models.

The module includes the following functions:

 - `embed_query`: Embeds the given query and returns an EmbedData object.
 - `embed_file`: Embeds the file at the given path and returns a list of EmbedData objects.
 - `embed_directory`: Embeds all the files in the given directory and returns a list of EmbedData objects.

The module also includes the `EmbedData` class, which represents the data of an embedded file.

Usage:
------

```python
import embed_anything

# Create a config 
config = embed_anything.EmbedConfig(
    jina=embed_anything.JinaConfig(
        model_id="jinaai/jina-embeddings-v2-small-en", 
        revision="main", 
        chunk_size=100
    )
)

# Embed a file
data = embed_anything.embed_file("test_files/test.pdf", 
                embeder="Jina", 
                config=config)

# Embed a directory
data = embed_anything.embed_directory("test_files", 
                embeder="Jina", 
                config=config)

# Embed Audio
audio_decoder_config = embed_anything.AudioDecoderConfig(
    decoder_model_id="openai/whisper-tiny.en",
    decoder_revision="main",
    model_type="tiny-en",
    quantized=False,
)
jina_config = embed_anything.JinaConfig(
    model_id="jinaai/jina-embeddings-v2-small-en", 
    revision="main", 
    chunk_size=100
)

config = embed_anything.EmbedConfig(jina=jina_config, 
            audio_decoder=audio_decoder_config)
data = embed_anything.embed_file(
    "test_files/audio/samples_hp0.wav", 
    embeder="Audio", 
    config=config
)

```

You can also store the embeddings to a vector database and not keep them on memory. Here is an example of how to use the `PineconeAdapter` class:

```python
import embed_anything
import os

from embed_anything.vectordb import PineconeAdapter


# Initialize the PineconeEmbedder class
api_key = os.environ.get("PINECONE_API_KEY")
index_name = "anything"
pinecone_adapter = PineconeAdapter(api_key, index_name=index_name)

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
```


Supported Embedding Models:
---------------------------
- Text Embedding Models:
    - "OpenAI"
    - "Bert"
    - "Jina"

- Image Embedding Models:
    - "Clip"
    - "SigLip" (Coming Soon)

- Audio Embedding Models:
    - "Whisper"
"""

from .embed_anything import *
import embed_anything.vectordb

__doc__ = embed_anything.__doc__
if hasattr(embed_anything, "__all__"):
    __all__ = embed_anything.__all__
