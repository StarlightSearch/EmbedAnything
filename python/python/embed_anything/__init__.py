"""EmbedAnything: A high-performance, multimodal embedding pipeline.

This module provides functions and classes for embedding queries, files, and
directories using different embedding models. It supports text, images, audio,
PDFs, and other media types with various embedding backends (Candle, ONNX, Cloud).

Main Functions:
---------------
- `embed_query`: Embeds text queries and returns a list of EmbedData objects.
- `embed_file`: Embeds a single file and returns a list of EmbedData objects.
- `embed_directory`: Embeds all files in a directory and returns a list of EmbedData objects.
- `embed_image_directory`: Embeds all images in a directory.
- `embed_audio_file`: Embeds audio files using Whisper for transcription.
- `embed_webpage`: Embeds content from a webpage URL.

Main Classes:
-------------
- `EmbeddingModel`: Main class for loading and using embedding models.
- `EmbedData`: Represents embedded data with text, embedding vector, and metadata.
- `TextEmbedConfig`: Configuration for text embedding (chunking, batching, etc.).
- `ColpaliModel`: Specialized model for document/image-text embedding.
- `ColbertModel`: Model for late-interaction embeddings.
- `Reranker`: Model for re-ranking search results.
- `AudioDecoderModel`: Model for audio transcription (Whisper).

Usage Examples:
---------------

### Text Embedding

```python
from embed_anything import EmbeddingModel, WhichModel, TextEmbedConfig
import embed_anything

# Load a text embedding model
model = EmbeddingModel.from_pretrained_local(
    WhichModel.Bert, 
    model_id="sentence-transformers/all-MiniLM-L12-v2"
)

# Configure embedding parameters
config = TextEmbedConfig(
    chunk_size=1000,              # Characters per chunk
    batch_size=32,                # Process 32 chunks at once
    splitting_strategy="sentence"  # Split by sentences
)

# Embed a PDF file
data = embed_anything.embed_file("test_files/document.pdf", embedder=model, config=config)

# Access results
for item in data:
    print(f"Text: {item.text[:100]}...")
    print(f"Embedding dimension: {len(item.embedding)}")
    print(f"Metadata: {item.metadata}")
```

### Image Embedding

```python
import embed_anything
import numpy as np
from embed_anything import EmbedData, EmbeddingModel, WhichModel

# Load CLIP model for image-text embeddings
model = EmbeddingModel.from_pretrained_local(
    WhichModel.Clip,
    model_id="openai/clip-vit-base-patch16"
)

# Embed all images in a directory
data: list[EmbedData] = embed_anything.embed_image_directory(
    "test_files", 
    embedder=model
)

# Convert to numpy array for similarity search
embeddings = np.array([item.embedding for item in data])

# Embed a text query
query = ["Photo of a monkey?"]
query_embedding = np.array(
    embed_anything.embed_query(query, embedder=model)[0].embedding
)

# Calculate cosine similarity
similarities = np.dot(embeddings, query_embedding)
most_similar_idx = np.argmax(similarities)
print(f"Most similar image: {data[most_similar_idx].text}")
```

### Audio Embedding

```python
from embed_anything import (
    AudioDecoderModel,
    EmbeddingModel,
    embed_audio_file,
    TextEmbedConfig,
    WhichModel
)
import embed_anything

# Load Whisper model for audio transcription
# Choose from: https://huggingface.co/distil-whisper or 
# https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013
audio_decoder = AudioDecoderModel.from_pretrained_hf(
    "openai/whisper-tiny.en", 
    revision="main", 
    model_type="tiny-en", 
    quantized=False
)

# Load text embedding model for transcribed text
embedder = EmbeddingModel.from_pretrained_hf(
    WhichModel.Bert,
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    revision="main",
)

# Configure text embedding
config = TextEmbedConfig(chunk_size=200, batch_size=32)

# Embed audio file (transcribes then embeds)
data = embed_anything.embed_audio_file(
    "test_files/audio/samples_hp0.wav",
    audio_decoder=audio_decoder,
    embedder=embedder,
    text_embed_config=config,
)

# Access transcribed and embedded segments
for item in data:
    print(f"Transcribed text: {item.text}")
    print(f"Metadata: {item.metadata}")
```

### Vector Database Integration

Store embeddings directly to a vector database without keeping them in memory:

```python
import embed_anything
import os
from embed_anything import EmbeddingModel, WhichModel, TextEmbedConfig
from embed_anything.vectordb import PineconeAdapter

# Initialize Pinecone adapter
api_key = os.environ.get("PINECONE_API_KEY")
pinecone_adapter = PineconeAdapter(api_key)

# Create or use existing index
try:
    pinecone_adapter.delete_index("my-index")
except:
    pass

pinecone_adapter.create_index(
    dimension=512,      # Embedding dimension
    metric="cosine",    # Similarity metric
    index_name="my-index"
)

# Load embedding model
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Clip, 
    "openai/clip-vit-base-patch16"
)

# Embed images and stream directly to Pinecone
data = embed_anything.embed_image_directory(
    "test_files",
    embedder=model,
    adapter=pinecone_adapter,  # Streams to database
)

# Embeddings are now in Pinecone, not in memory
print("Embeddings stored in Pinecone!")
```

### ONNX Models (Faster Inference)

```python
from embed_anything import EmbeddingModel, WhichModel, ONNXModel, Dtype

# Load a pre-configured ONNX model (faster, lower memory)
model = EmbeddingModel.from_pretrained_onnx(
    WhichModel.Bert,
    model_id=ONNXModel.BGESmallENV15Q,  # Quantized BGE model
    dtype=Dtype.Q4F16
)

# Use like any other model
data = embed_anything.embed_file("test_files/document.pdf", embedder=model)
```

### Semantic Chunking

```python
from embed_anything import EmbeddingModel, WhichModel, TextEmbedConfig
import embed_anything

# Main embedding model
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Bert,
    model_id="sentence-transformers/all-MiniLM-L12-v2"
)

# Semantic encoder for chunk boundaries
semantic_encoder = EmbeddingModel.from_pretrained_hf(
    WhichModel.Jina,
    model_id="jinaai/jina-embeddings-v2-small-en"
)

# Configure semantic chunking
config = TextEmbedConfig(
    chunk_size=1000,
    batch_size=32,
    splitting_strategy="semantic",
    semantic_encoder=semantic_encoder
)

# Embed with semantic chunking
data = embed_anything.embed_file("test_files/document.pdf", embedder=model, config=config)
```

Supported Embedding Models:
---------------------------
- **Text Models**: BERT, Jina, Qwen3, Splade, ColBERT, Model2Vec
- **Image Models**: CLIP, SigLip
- **Audio Models**: Whisper, DistilWhisper
- **Document Models**: ColPali
- **Rerankers**: Jina Reranker, BGE Reranker, Qwen3 Reranker
- **Cloud Models**: OpenAI, Cohere, Gemini

For more examples and detailed documentation, visit:
https://embed-anything.com
"""

import platform
import os

from .vectordb import *
from ._embed_anything import *
import onnxruntime
import glob

path = os.path.dirname(onnxruntime.__file__) + "/capi/"

if path is None:
    print("onnxruntime is not installed. Install it using `pip install onnxruntime-gpu`")

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
