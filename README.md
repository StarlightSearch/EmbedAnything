
<p align ="center">
<img width=600 src = "https://res.cloudinary.com/dltwftrgc/image/upload/v1712504276/Projects/EmbedAnything_500_x_200_px_a4l8xu.png">
</p>


<p align="center">
    <b>Infra for building multimodal embeddings from unstructured sources, built in Rust for speed and robustness</b>
</p>

EmbedAnything is a powerful library designed to streamline the creation and management of embedding pipelines. Whether you're working with text, images, audio, or any other type of data., EmbedAnything makes it easy to generate embeddings from multiple sources and store them efficiently in a vector database.


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
from embed_anything import *
data = embed_anything.embed_file("filename.pdf", embeder= "Bert")
embeddings = np.array([data.embedding for data in data])
```


## For multimodal embedding: we support CLIP
### Requirements Directory with pictures you want to search for example we have test_files with images of cat, dogs etc

```python
from embed_anything import *
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


