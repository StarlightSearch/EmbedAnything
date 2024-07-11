# Welcome to EmbedAnything Docs


<p align ="center">
<img width=600 src = "https://res.cloudinary.com/dltwftrgc/image/upload/v1712504276/Projects/EmbedAnything_500_x_200_px_a4l8xu.png">
</p>


<p align="center">
    <b>Minimalist Framework for building local and multimodal embeddings built in Rust 🦀</b>
</p>

[![Downloads](https://static.pepy.tech/badge/embed-anything)](https://pepy.tech/project/embed-anything)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CowJrqZxDDYJzkclI-rbHaZHgL9C6K3p?usp=sharing)
[![license]( https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![license]( https://img.shields.io/badge/Package-PYPI-blue.svg)](https://pypi.org/project/embed-anything/)
[![license](https://img.shields.io/discord/1213966302046064711?style=flat&logo=discord&link=https%3A%2F%2Fdiscord.gg%2FHGxDZxNt9G)](https://discord.gg/juETVTMdZu)

EmbedAnything is a powerful python library designed to streamline the creation and management of embedding pipelines. Whether you're working with text, images, audio, or any other type of data., EmbedAnything makes it easy to generate embeddings from multiple sources and store them efficiently in a vector database.

## 🦀The Benefit of Rust for Speed
By using Rust for its core functionalities, EmbedAnything offers significant speed advantages:

➡️Rust is Compiled: Unlike Python, Rust compiles directly to machine code, resulting in faster execution. <br />
➡️Memory Management: Rust enforces memory management simultaneously, preventing memory leaks and crashes that can plague other languages <br />
➡️Rust achieves true multithreading.

## 🚀Why Candle?...
➡️Running language models or embedding models locally can be difficult, especially when you want to deploy a product that utilizes these models.  <br />
➡️If you use the transformers library from Hugging Face in Python, you will depend on PyTorch for tensor operations. <br />
➡️ This, in turn, has a dependency on Libtorch, which means that you will need to include the entire Libtorch library with your product.  <br />
➡️Also, Candle allows inferences on CUDA-enabled GPUs right out of the box. 
We will soon post on how we use Candle to increase the performance and decrease the memory usage of EmbedAnything.

## Examples
1. Image Search: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CowJrqZxDDYJzkclI-rbHaZHgL9C6K3p?usp=sharing)

[Watch the demo](https://youtu.be/HLXIuznnXcI)


## 🚀 Key Features

- **Local Embedding** Works with local embedding models like AllminiLM
- **MultiModality** Works with text and image and will soon expand to audio
- **Python Interface:** Packaged as a Python library for seamless integration into your existing projects.
- **Efficient:** Optimized for speed and performance, with core functionality written in Rust.
- **Scalable:** Store embeddings in a vector database for easy retrieval and scalability.
- **OpenAI** Works with openai as well




## 💚 Installation

`
pip install embed-anything`





# 🧑‍🚀 Getting Started

## For local models 

### To use local embedding: we support Bert and Jina

```python
import embed_anything
data = embed_anything.embed_file("filename.pdf", embeder= "Bert")
embeddings = np.array([data.embedding for data in data])
```


## For multimodal embedding: we support CLIP
### Requirements Directory with pictures you want to search for example we have test_files with images of cat, dogs etc

```python
import embed_anything
data = embed_anything.embed_directory("test_files", embeder= "Clip")
embeddings = np.array([data.embedding for data in data])

query = "photo of a dog"
query_embedding = np.array(embed_anything.embed_query(query, embeder= "Clip")[0].embedding)
similarities = np.dot(embeddings, query_embedding)
max_index = np.argmax(similarities)
Image.open(data[max_index].text).show()
```


## For OpenAI 
1. Please check if you already have the OpenAI key in the Environment variable.

### If you are using embed-anything==0.1.7 version (latest version)

```python
import embed_anything
data = embed_anything.embed_file("filename.pdf", embeder= "OpenAI")
embeddings = np.array([data.embedding for data in data])
```

#  🚧 Contributing to EmbedAnything

First of all, thank you for taking the time to contribute to this project. We truly appreciate your contributions, whether it's bug reports, feature suggestions, or pull requests. Your time and effort are highly valued in this project. 🚀

This document provides guidelines and best practices to help you to contribute effectively. These are meant to serve as guidelines, not strict rules. We encourage you to use your best judgment and feel comfortable proposing changes to this document through a pull request.

**********************************Table of Content:********************************** 
1. [Code of conduct]
2. [Quick Start]
3. [RoadMap]


## RoadMap 
☑️Graph embedding -- build deepwalks embeddings depth first and word to vec <br />
☑️Add whisper for audio embeddings <br />
☑️Zero-shot application <br />
☑️Asynchronous chunks training


## ✔️ Code of Conduct:

Please read our [Code of Conduct] to understand the expectations we have for all contributors participating in this project. By participating, you agree to abide by our Code of Conduct.

## 🚀 Quick Start

You can quickly get started with contributing by searching for issues with the labels **"Good First Issue"** or **"Help Needed"** in the [Issues Section]. If you think you can contribute, comment on the issue and we will assign it to you.  

To set up your development environment, please follow the steps mentioned below : 

1. Fork the repository and create a clone of the fork
2. Create a branch for a feature or a bug you are working on in your fork
3. If you are working with OpenAI make sure you have the keys

## Contributing Guidelines 
 
### 🔍 Reporting Bugs


1. Title describing the issue clearly and concisely with relevant labels
2. Provide a detailed description of the problem and the necessary steps to reproduce the issue.
3. Include any relevant logs, screenshots, or other helpful information supporting the issue.

### 💡 New Feature or Suggesting Enhancements



## ☑️ ToDo

- **Vector Database** Add functionalities to integrate with any Vector Database



