# 🏠 Home


<p align ="center">
<img width=400 src = "https://res.cloudinary.com/dltwftrgc/image/upload/v1712504276/Projects/EmbedAnything_500_x_200_px_a4l8xu.png">
</p>




<div align="center">

  <p align="center">
    <b>Supercharge your embedding pipeline with minimalist and lightening fast framework built in rust 🦀</b>
    <br />
    <a href="https://starlightsearch.github.io/EmbedAnything/references/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href=https://youtu.be/HLXIuznnXcI>View Demo</a>
    ·
    <a href="https://github.com/StarlightSearch/EmbedAnything/tree/main/examples">Examples</a>
    ·
    <a href="https://github.com/StarlightSearch/EmbedAnything/issues/new">Request Feature</a>
    .
    <a href="https://huggingface.co/spaces/akshayballal/search_in_audio">Search in Audio Space</a>
    
  </p>
</div>


EmbedAnything is a minimalist yet highly performant, lightweight, lightening fast, multisource, multimodal and local embedding pipeline, built in rust. Whether you're working with text, images, audio, PDFs, websites, or other media, EmbedAnything simplifies the process of generating embeddings from various sources and streaming them to a vector database.We support dense, sparse and late-interaction embeddings.

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


## 💡What is Vector Streaming

Vector Streaming enables you to process and generate embeddings for files and stream them, so if you have 10 GB of file, it can continuously generate embeddings Chunk by Chunk, that you can segment semantically, and store them in the vector database of your choice, Thus it eliminates bulk embeddings storage on RAM at once.

[![EmbedAnythingXWeaviate](https://github.com/StarlightSearch/EmbedAnything/blob/main/docs/assets/demo.gif)](https://www.youtube.com/watch?v=OJRWPLQ44Dw)

## 🚀 Key Features

- **Local Embedding** : Works with local embedding models like BERT and JINA
- **ColPali** : Support for ColPali in GPU version
- **Splade** : Support for sparse embeddings for hybrid
- **Cloud Embedding Models:**: Supports OpenAI and Cohere.  
- **MultiModality** : Works with text sources like PDFs, txt, md, Images JPG and Audio, .WAV
- **Rust** : All the file processing is done in rust for speed and efficiency
- **Candle** : We have taken care of hardware acceleration as well, with Candle.
- **Python Interface:** Packaged as a Python library for seamless integration into your existing projects.
- **Vector Streaming:** Continuously create and stream embeddings if you have low resource.



## 🦀 Why Embed Anything 

➡️Faster execution. <br />
➡️Memory Management: Rust enforces memory management simultaneously, preventing memory leaks and crashes that can plague other languages <br />
➡️True multithreading <br />
➡️Running language models or embedding models locally and efficiently <br />
➡️Candle allows inferences on CUDA-enabled GPUs right out of the box. <br />
➡️Decrease the memory usage of EmbedAnything.

## ⭐ Supported Models

We support a range of models, that can be supported by Candle, We have given a set of tested models but if you have specific usecase do mention it in the issue.


## 🧑‍🚀 Getting Started

### 📩 Installation

```bash
pip install embed-anything
```

For GPUs and using special models like ColPali <br/>

```bash
pip install embed-anything-gpu
```


### 📝 Usage


```python
model = EmbeddingModel.from_pretrained_local(
    WhichModel.Bert, model_id="sentence-transformers/all-MiniLM-L6-v2"
)
data = embed_anything.embed_file("test_files/test.pdf", embeder=model)
```


## Supported Models

| Model  | HF link |
| ------------- | ------------- | 
| Jina  | [Jina Models](https://huggingface.co/collections/jinaai/jina-embeddings-v2-65708e3ec4993b8fb968e744) | 
| Bert | All Bert based models |
| CLIP | openai/clip-* | 
| Whisper| [OpenAI Whisper models](https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013)|
| ColPali | vidore/colpali-v1.2-merged |
| Splade | [Splade Models] (https://huggingface.co/collections/naver/splade-667eb6df02c2f3b0c39bd248) and other Splade based models |

  

### ♠️ Splade Models

```python

model = EmbeddingModel.from_pretrained_hf(
    WhichModel.SparseBert, "naver/splade-v3")
```

### 👁️ ColPali Models 

```python
model: ColpaliModel = ColpaliModel.from_pretrained("vidore/colpali-v1.2-merged", None)
```

### 📷 Image Embeddings

*Requirements*: Directory with pictures you want to search for example we have `test_files` with images of cat, dogs etc

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

### 🔊 Audio Embedding using Whisper
*requirements*:  Audio .wav files.


```python
import embed_anything
from embed_anything import JinaConfig, EmbedConfig, AudioDecoderConfig
import time

start_time = time.time()

# choose any whisper or distilwhisper model 
# from https://huggingface.co/distil-whisper or 
# https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013
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