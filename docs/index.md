# Home


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


EmbedAnything is a minimalist yet highly performant, lightweight, lightening fast, multisource, multimodal and local embedding pipeline, built in rust. Whether you're working with text, images, audio, PDFs, websites, or other media, EmbedAnything simplifies the process of generating embeddings from various sources and storing them in a vector database.

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

### 💚 Installation

`
pip install embed-anything`


### 📝 Usage

To use local embedding: we support Bert and Jina

```python
import embed_anything
data = embed_anything.embed_file("file_path.pdf", embeder= "Bert")
embeddings = np.array([data.embedding for data in data])
```



#### 📷 Image Embeddings

*Requirements*: Directory with pictures you want to search for example we have `test_files` with images of cat, dogs etc

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

#### 🔊 Audio Embedding using Whisper
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

#### 🤗 Using embedding models from Hugging Face
```python
jina_config = JinaConfig(
    model_id="Custom link given below", revision="main", chunk_size=100
)
embed_config = EmbedConfig(jina=jina_config)
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

Yet to do be done <br />
☑️Vector Database: Add functionalities to integrate with any Vector Database <br />
☑️Graph embedding -- build deepwalks embeddings depth first and word to vec <br />
☑️Asynchronous chunks training


## ✔️ Code of Conduct:

Please read our [Code of Conduct] to understand the expectations we have for all contributors participating in this project. By participating, you agree to abide by our Code of Conduct.

## Quick Start

You can quickly get started with contributing by searching for issues with the labels **"Good First Issue"** or **"Help Needed"** in the [Issues Section]. If you think you can contribute, comment on the issue and we will assign it to you.  

To set up your development environment, please follow the steps mentioned below : 

1. Fork the repository from dev, We don't allow direct contribution to main


## Contributing Guidelines 
 
### 🔍 Reporting Bugs


1. Title describing the issue clearly and concisely with relevant labels
2. Provide a detailed description of the problem and the necessary steps to reproduce the issue.
3. Include any relevant logs, screenshots, or other helpful information supporting the issue.





