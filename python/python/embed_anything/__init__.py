"""This module provides functions and classes for embedding queries, files, and
directories using different embedding models.

The module includes the following functions:

 - `embed_query`: Embeds the given query and returns an EmbedData object.
 - `embed_file`: Embeds the file at the given path and returns a list of EmbedData objects.
 - `embed_directory`: Embeds all the files in the given directory and returns a list of EmbedData objects.

The module also includes the `EmbedData` class, which represents the data of an embedded file.

Usage:
------

```python
import embed_anything
from embed_anything import EmbedData

#For text files

model = EmbeddingModel.from_pretrained_local(
    WhichModel.Bert, model_id="Hugging_face_link"
)
data = embed_anything.embed_file("test_files/test.pdf", embedder=model)


#For images
model = embed_anything.EmbeddingModel.from_pretrained_local(
    embed_anything.WhichModel.Clip,
    model_id="openai/clip-vit-base-patch16",
    # revision="refs/pr/15",
)
data: list[EmbedData] = embed_anything.embed_directory("test_files", embedder=model)
embeddings = np.array([data.embedding for data in data])
query = ["Photo of a monkey?"]
query_embedding = np.array(
    embed_anything.embed_query(query, embedder=model)[0].embedding
)
# For audio files
from embed_anything import (
    AudioDecoderModel,
    EmbeddingModel,
    embed_audio_file,
    TextEmbedConfig,
)
# choose any whisper or distilwhisper model from https://huggingface.co/distil-whisper or https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013
audio_decoder = AudioDecoderModel.from_pretrained_hf(
    "openai/whisper-tiny.en", revision="main", model_type="tiny-en", quantized=False
)
embedder = EmbeddingModel.from_pretrained_hf(
    embed_anything.WhichModel.Bert,
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    revision="main",
)
config = TextEmbedConfig(chunk_size=200, batch_size=32)
data = embed_anything.embed_audio_file(
    "test_files/audio/samples_hp0.wav",
    audio_decoder=audio_decoder,
    embedder=embedder,
    text_embed_config=config,
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
pinecone_adapter = PineconeAdapter(api_key)

try:
    pinecone_adapter.delete_index("anything")
except:
    pass

# Initialize the PineconeEmbedder class

pinecone_adapter.create_index(dimension=512, metric="cosine")

# bert_model = EmbeddingModel.from_pretrained_hf(
#     WhichModel.Bert, "sentence-transformers/all-MiniLM-L12-v2", revision="main"
# )

clip_model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Clip, "openai/clip-vit-base-patch16", revision="main"
)

embed_config = TextEmbedConfig(chunk_size=512, batch_size=32)


data = embed_anything.embed_image_directory(
    "test_files",
    embedder=clip_model,
    adapter=pinecone_adapter,
    # config=embed_config,
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

from ._embed_anything import *
from .vectordb import *
import platform
import os
import onnxruntime
import glob

path = os.path.dirname(onnxruntime.__file__) + "/capi/"

if path is None:
    print("onnxruntime is not installed. Install it using `pip install onnxruntime`")
else:
    if platform.system() == "Windows":
        # For Windows, look for DLL files
        dylib_path = glob.glob(os.path.join(path, "onnxruntime.dll"))
    elif platform.system() == "Darwin":
        dylib_path = glob.glob(os.path.join(path, "libonnxruntime*"))
    else:
        # For Linux, look for shared object files
        dylib_path = glob.glob(os.path.join(path, "libonnxruntime.so*"))

    if dylib_path:
        os.environ["ORT_DYLIB_PATH"] = dylib_path[0]
    else:
        print("onnxruntime dynamic library not found.")

__doc__ = _embed_anything.__doc__
if hasattr(_embed_anything, "__all__"):
    __all__ = _embed_anything.__all__
