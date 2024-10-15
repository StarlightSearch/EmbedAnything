

<p align ="center">
<img width=400 src = "https://res.cloudinary.com/dltwftrgc/image/upload/v1712504276/Projects/EmbedAnything_500_x_200_px_a4l8xu.png">
</p>



<div align="center">

[![Downloads](https://static.pepy.tech/badge/embed-anything)](https://pepy.tech/project/embed-anything)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CowJrqZxDDYJzkclI-rbHaZHgL9C6K3p?usp=sharing)
[![license]( https://img.shields.io/badge/License-Apache-blue.svg)](https://opensource.org/licenses/Apache2.0)
[![package]( https://img.shields.io/badge/Package-PYPI-blue.svg)](https://pypi.org/project/embed-anything/)
[![discord](https://img.shields.io/discord/1213966302046064711?style=flat&logo=discord&link=https%3A%2F%2Fdiscord.gg%2FHGxDZxNt9G)](https://discord.gg/juETVTMdZu)
[![roadmap](https://img.shields.io/badge/Roadmap-1D9BF0?style=flat&logo=twitter&logoColor=white)](https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#roadmap)

</div>


<div align="center">

  <p align="center">
    <b>Generate and stream your embeddings with minimalist and lightning fast framework built in rust 🦀</b>
    <br />
    <a href="https://starlightsearch.github.io/EmbedAnything/references/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href=https://youtu.be/HLXIuznnXcI>View Demo</a>
    ·
    <a href="https://github.com/StarlightSearch/EmbedAnything/tree/main/examples">Examples</a>
    ·
    <a href="https://github.com/StarlightSearch/EmbedAnything/tree/main/examples/adapters">Vector Streaming Adapters</a>
    .
    <a href="https://huggingface.co/spaces/akshayballal/search_in_audio">Search in Audio Space</a>
    
  </p>
</div>


EmbedAnything is a minimalist yet highly performant, lightweight, lightening fast, multisource, multimodal and local embedding pipeline, built in rust. Whether you're working with text, images, audio, PDFs, websites, or other media, EmbedAnything simplifies the process of generating embeddings from various sources and streaming them to a vector database.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#the-benefit-of-rust-for-speed">Built With Rust</a></li>
        <li><a href="https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#why-candle">Why Candle?</a></li>
      </ul>
    </li>
    <li>
      <a href="https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#-getting-started">Getting Started</a>
      <ul>
        <li><a href="https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#-installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#-getting-started">Usage</a></li>
    <li><a href="https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#roadmap">Roadmap</a></li>
    <li><a href="https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#quick-start">Contributing</a></li>
    <li><a href="https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#Supported-Models">How to add custom model and chunk size</a></li>
    
  </ol>
</details>




## 🚀 Key Features

- **Local Embedding** : Works with local embedding models like BERT and JINA
- **Cloud Embedding Models:**: Supports OpenAI and Cohere.  
- **MultiModality** : Works with text sources like PDFs, txt, md, Images JPG and Audio, .WAV
- **Rust** : All the file processing is done in rust for speed and efficiency
- **Candle** : We have taken care of hardware acceleration as well, with Candle.
- **Python Interface:** Packaged as a Python library for seamless integration into your existing projects.
- **Scalable:** Store embeddings in a vector database for easy retrieval and scalability. 
- **Vector Streaming:** Continuously create and stream embeddings if you have low resource.

## 💡What is Vector Streaming

Vector Streaming enables you to process and generate embeddings for files and stream them, so if you have 10 GB of file, it can continuously generate embeddings Chunk by Chunk, that you can segment semantically, and store them in the vector database of your choice, Thus it eliminates bulk embeddings storage on RAM at once.

[![EmbedAnythingXWeaviate](https://github.com/StarlightSearch/EmbedAnything/blob/main/docs/assets/demo.gif)](https://www.youtube.com/watch?v=OJRWPLQ44Dw)

## 🦀 Why Embed Anything 

➡️Faster execution. <br />
➡️Memory Management: Rust enforces memory management simultaneously, preventing memory leaks and crashes that can plague other languages <br />
➡️True multithreading <br />
➡️Running language models or embedding models locally and efficiently <br />
➡️Candle allows inferences on CUDA-enabled GPUs right out of the box. <br />
➡️Decrease the memory usage of EmbedAnything.

# ⭐ Supported Models

We support a range of models, that can be supported by Candle, We have given a set of tested models but if you have specific usecase do mention it in the issue.

## How to add custom model and Chunk Size And Semantic Chunking.
```python
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Bert, model_id="model link from huggingface"
)
config = TextEmbedConfig(chunk_size=200, batch_size=32)
data = embed_anything.embed_file("file_address", embeder=model, config=config)
```


| Model  | Custom link |
| ------------- | ------------- |
| Jina  | jinaai/jina-embeddings-v2-base-en  |
|   | jinaai/jina-embeddings-v2-small-en  |
| Bert | sentence-transformers/all-MiniLM-L6-v2 |
|      | sentence-transformers/all-MiniLM-L12-v2 |
|      | sentence-transformers/paraphrase-MiniLM-L6-v2 |
| Clip | openai/clip-vit-base-patch32 | 
| Whisper| Most OpenAI Whisper from huggingface supported.


### For Semantic Chunking

```python
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Bert, model_id="sentence-transformers/all-MiniLM-L12-v2"
)

# with semantic encoder
semantic_encoder = EmbeddingModel.from_pretrained_hf(WhichModel.Jina, model_id = "jinaai/jina-embeddings-v2-small-en")
config = TextEmbedConfig(chunk_size=256, batch_size=32, splitting_strategy = "semantic", semantic_encoder=semantic_encoder)

```



# 🧑‍🚀 Getting Started

## 💚 Installation

`
pip install embed-anything`


# Usage



## ➡️ Usage For 0.3 and later version


### To use local embedding: we support Bert and Jina

```python
model = EmbeddingModel.from_pretrained_local(
    WhichModel.Bert, model_id="Hugging_face_link"
)
data = embed_anything.embed_file("test_files/test.pdf", embeder=model)
```



## For multimodal embedding: we support CLIP
### Requirements Directory with pictures you want to search for example we have test_files with images of cat, dogs etc

```python
import embed_anything
from embed_anything import EmbedData
model = embed_anything.EmbeddingModel.from_pretrained_local(
    embed_anything.WhichModel.Clip,
    model_id="openai/clip-vit-base-patch16",
    # revision="refs/pr/15",
)
data: list[EmbedData] = embed_anything.embed_directory("test_files", embeder=model)
embeddings = np.array([data.embedding for data in data])
query = ["Photo of a monkey?"]
query_embedding = np.array(
    embed_anything.embed_query(query, embeder=model)[0].embedding
)
similarities = np.dot(embeddings, query_embedding)
max_index = np.argmax(similarities)
Image.open(data[max_index].text).show()
```

## Audio Embedding using Whisper
### requirements:  Audio .wav files.


```python
import embed_anything
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
embeder = EmbeddingModel.from_pretrained_hf(
    embed_anything.WhichModel.Bert,
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    revision="main",
)
config = TextEmbedConfig(chunk_size=200, batch_size=32)
data = embed_anything.embed_audio_file(
    "test_files/audio/samples_hp0.wav",
    audio_decoder=audio_decoder,
    embeder=embeder,
    text_embed_config=config,
)
print(data[0].metadata)

```




## 🚧 Contributing to EmbedAnything



First of all, thank you for taking the time to contribute to this project. We truly appreciate your contributions, whether it's bug reports, feature suggestions, or pull requests. Your time and effort are highly valued in this project. 🚀

This document provides guidelines and best practices to help you to contribute effectively. These are meant to serve as guidelines, not strict rules. We encourage you to use your best judgment and feel comfortable proposing changes to this document through a pull request.



<li><a href="##-RoadMap">Roadmap</a></li>
<li><a href="##-Quick-Start">Quick Start</a></li>
<li><a href="##-Contributing-Guidelines">Guidelines</a></li>

# 🏎️ RoadMap 
One of the aims of EmbedAnything is to allow AI engineers to easily use state of the art embedding models on typical files and documents. A lot has already been accomplished here and these are the formats that we support right now and a few more have to be done. <br />

## 🖼️ Modalities and Source

We’re excited to share that we've expanded our platform to support multiple modalities, including:

- Audio files
- Markdowns
- Websites
- Images
- Custom model uploads <br />

This gives you the flexibility to work with various data types all in one place! 🌐 <br />

## 💜 Product
We’ve rolled out some major updates in version 0.3 to improve both functionality and performance. Here’s what’s new:

- Semantic Chunking: Optimized chunking strategy for better Retrieval-Augmented Generation (RAG) workflows.

- Streaming for Efficient Indexing: We’ve introduced streaming for memory-efficient indexing in vector databases. Want to know more? Check out our article on this feature here: https://www.analyticsvidhya.com/blog/2024/09/vector-streaming/

- Zero-Shot Applications: Explore our zero-shot application demos to see the power of these updates in action.

- Intuitive Functions: Version 0.3 includes a complete refactor for more intuitive functions, making the platform easier to use.

- Chunkwise Streaming: Instead of file-by-file streaming, we now support chunkwise streaming, allowing for more flexible and efficient data processing.

Check out the latest release :  and see how these features can supercharge your GenerativeAI pipeline! ✨




# 🚀Where are we heading  <br />


## ⚙️ Performance 
We've received quite a few questions about why we're using Candle, so here's a quick explanation:

One of the main reasons is that Candle doesn't require any specific ONNX format models, which means it can work seamlessly with any Hugging Face model. This flexibility has been a key factor for us. However, we also recognize that we’ve been compromising a bit on speed in favor of that flexibility.

What’s Next?
To address this, we’re excited to announce that we’re introducing ORT support along with our previous framework on hugging-face ,

➡️ Significantly faster performance</br >
- Stay tuned for these exciting updates! 🚀</br >


## 🫐Embeddings:

We had multimodality from day one for our infrastructure. We have already included it for websites, images and audios but we want to expand it further to.

☑️Graph embedding -- build deepwalks embeddings depth first and word to vec <br />
☑️Video Embedding <br/>
☑️ Yolo Clip <br/>


## 🌊Expansion to other Vector Adapters

We currently support a wide range of vector databases for streaming embeddings, including:

- Elastic: thanks to amazing and active Elastic team for the contribution <br/>
- Weaviate<br/>
- Pinecone<br/>
- Qdrant<br/>

But we're not stopping there! We're actively working to expand this list.

Want to Contribute?
If you’d like to add support for your favorite vector database, we’d love to have your help! Check out our contribution.md for guidelines, or feel free to reach out directly starlight-search@proton.me. Let's build something amazing together! 💡


