
<p align ="center">
<img width=400 src = "https://res.cloudinary.com/dltwftrgc/image/upload/v1712504276/Projects/EmbedAnything_500_x_200_px_a4l8xu.png">
</p>



<div align="center">

[![Downloads](https://static.pepy.tech/badge/embed-anything)](https://pepy.tech/project/embed-anything)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CowJrqZxDDYJzkclI-rbHaZHgL9C6K3p?usp=sharing)
[![license]( https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![license]( https://img.shields.io/badge/Package-PYPI-blue.svg)](https://pypi.org/project/embed-anything/)
[![license](https://img.shields.io/discord/1213966302046064711?style=flat&logo=discord&link=https%3A%2F%2Fdiscord.gg%2FHGxDZxNt9G)](https://discord.gg/juETVTMdZu)

</div>


<div align="center">

  <p align="center">
    <b>Minimalist and Robust Framework for local and multimodal embeddings built in Rust 🦀</b>
    <br />
    <a href="https://starlightsearch.github.io/EmbedAnything/references/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href=https://youtu.be/HLXIuznnXcI>View Demo</a>
    ·
    <a href="https://github.com/StarlightSearch/EmbedAnything/tree/main/examples">Examples</a>
    ·
    <a href="https://github.com/StarlightSearch/EmbedAnything/issues/new">Request Feature</a>
  </p>
</div>


EmbedAnything is a powerful Python library designed to streamline the creation and management of embedding pipelines. Whether you're working with text, images, audio, PDFs, websites, or other media, EmbedAnything simplifies the process of generating embeddings from various sources and storing them in a vector database.

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
  </ol>
</details>



## 🚀 Key Features

- **Local Embedding** : Works with local embedding models like BERT and JINA
- **MultiModality** : Works with text sources like PDFs, txt, md, Images JPG and Audio, .WAV
- **Rust** : All the file processing is done in rust for speed and efficiency
- **Candle** : We have taken care of hardware acceleration as well, with Candle.
- **Python Interface:** Packaged as a Python library for seamless integration into your existing projects.
- **Scalable:** Store embeddings in a vector database for easy retrieval and scalability.
- **OpenAI** Supports OpenAI and Whisper embeddings


## 🦀The Benefit of Rust for Speed
By using Rust for its core functionalities, EmbedAnything offers significant speed advantages:

➡️Faster execution. <br />
➡️Memory Management: Rust enforces memory management simultaneously, preventing memory leaks and crashes that can plague other languages <br />
➡️True multithreading.

## 🤗Why Candle? by Hugging face
➡️Running language models or embedding models locally and efficiently <br />
➡️Candle allows inferences on CUDA-enabled GPUs right out of the box. <br />
➡️Decrease the memory usage of EmbedAnything.








# 🧑‍🚀 Getting Started

## 💚 Installation

`
pip install embed-anything`

## Usage

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


## For OpenAI- Whisper
### requirements:  Please check if you already have the OpenAI key in the Environment variable.

```python
import embed_anything
import time

start_time = time.time()
data = embed_anything.embed_file(
    "file_path.wav", embeder="Whisper-Bert"
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

Yet to do be done <br />
☑️Vector Database: Add functionalities to integrate with any Vector Database <br />
☑️Graph embedding -- build deepwalks embeddings depth first and word to vec <br />
☑️Zero-shot application <br />
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




