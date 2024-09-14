

<p align ="center">
<img width=400 src = "https://res.cloudinary.com/dltwftrgc/image/upload/v1712504276/Projects/EmbedAnything_500_x_200_px_a4l8xu.png">
</p>



<div align="center">

[![Downloads](https://static.pepy.tech/badge/embed-anything)](https://pepy.tech/project/embed-anything)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CowJrqZxDDYJzkclI-rbHaZHgL9C6K3p?usp=sharing)
[![license]( https://img.shields.io/badge/License-Apache-blue.svg)](https://opensource.org/licenses/Apache2.0)
[![package]( https://img.shields.io/badge/Package-PYPI-blue.svg)](https://pypi.org/project/embed-anything/)
[![discord](https://img.shields.io/discord/1213966302046064711?style=flat&logo=discord&link=https%3A%2F%2Fdiscord.gg%2FHGxDZxNt9G)](https://discord.gg/juETVTMdZu)

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
- **Cloud Embedding Models:**: Supports OpenAI. Mistral and Cohere Support coming soon.  
- **MultiModality** : Works with text sources like PDFs, txt, md, Images JPG and Audio, .WAV
- **Rust** : All the file processing is done in rust for speed and efficiency
- **Candle** : We have taken care of hardware acceleration as well, with Candle.
- **Python Interface:** Packaged as a Python library for seamless integration into your existing projects.
- **Scalable:** Store embeddings in a vector database for easy retrieval and scalability. 
- **Vector Streaming:** Continuously create and stream embeddings if you have low resource.

## 💡What is Vector Streaming

Vector Streaming enables you to process and generate embeddings for files and stream them, so if you have 10 GB of file, it can continuously generate embeddings file by file (Or chunk by chunk in future) and store them in the vector database of your choice, Thus it eliminates bulk embeddings storage on RAM at once.

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

## How to add custom model and Chunk Size.
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

## ➡️ Usage For 0.2

### To use local embedding: we support Bert and Jina

```python
import embed_anything
data = embed_anything.embed_file("file_path.pdf", embeder= "Bert")
embeddings = np.array([data.embedding for data in data])
```



## For multimodal embedding: we support CLIP
### Requirements Directory with pictures you want to search for example we have test_files with images of cat, dogs etc

```python
import embed_anything
data = embed_anything.embed_directory("directory_path", embeder= "Clip")
embeddings = np.array([data.embedding for data in data])

query = ["photo of a dog"]
query_embedding = np.array(embed_anything.embed_query(query, embeder= "Clip")[0].embedding)
similarities = np.dot(embeddings, query_embedding)
max_index = np.argmax(similarities)
Image.open(data[max_index].text).show()
```

## Audio Embedding using Whisper
### requirements:  Audio .wav files.


```python
import embed_anything
from embed_anything import JinaConfig, EmbedConfig, AudioDecoderConfig
import time

start_time = time.time()

# choose any whisper or distilwhisper model from https://huggingface.co/distil-whisper or https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013
audio_decoder_config = AudioDecoderConfig(
    decoder_model_id="openai/whisper-tiny.en",
    decoder_revision="main",
    model_type="tiny-en",
    quantized=False,
)
jina_config = JinaConfig(
    model_id="jinaai/jina-embeddings-v2-small-en", revision="main", chunk_size=100
)

config = EmbedConfig(jina=jina_config, audio_decoder=audio_decoder_config)
data = embed_anything.embed_file(
    "test_files/audio/samples_hp0.wav", embeder="Audio", config=config
)
print(data[0].metadata)
end_time = time.time()
print("Time taken: ", end_time - start_time)


```







## 🚧 Contributing to EmbedAnything



First of all, thank you for taking the time to contribute to this project. We truly appreciate your contributions, whether it's bug reports, feature suggestions, or pull requests. Your time and effort are highly valued in this project. 🚀

This document provides guidelines and best practices to help you to contribute effectively. These are meant to serve as guidelines, not strict rules. We encourage you to use your best judgment and feel comfortable proposing changes to this document through a pull request.



<li><a href="##-RoadMap">Roadmap</a></li>
<li><a href="##-Quick-Start">Quick Start</a></li>
<li><a href="##-Contributing-Guidelines">Guidelines</a></li>


## RoadMap 
One of the aims of EmbedAnything is to allow AI engineers to easily use state of the art embedding models on typical files and documents. A lot has already been accomplished here and these are the formats that we support right now and a few more have to be done. <br />
✅ Markdown, PDFs, and Website <br />
✅ WAV File <br />
✅ JPG, PNG, webp <br />
✅Add whisper for audio embeddings <br />
✅Custom model upload, anything that is available in candle <br />
✅Custom chunk size <br />
✅Pinecone Adapter, to directly save it on it. <br />
✅Zero-shot application <br />
✅Vector database integration via streaming adapters <br />
✅Refactoring for intuitive functions

Yet to do be done <br />
☑️Introducing chunkwise streaming instead of file <br />
☑️Graph embedding -- build deepwalks embeddings depth first and word to vec <br />
☑️Video Embedding
☑️ Yolo Clip
☑️ Add more Vector Database Adapters




