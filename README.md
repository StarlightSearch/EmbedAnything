![image](https://github.com/StarlightSearch/EmbedAnything/assets/27013287/3eb92af8-d266-4096-a4c8-ecf9e84e2b44)# EmbedAnything
<p align ="center">
<img width=600 src = "https://res.cloudinary.com/dltwftrgc/image/upload/v1712504276/Projects/EmbedAnything_500_x_200_px_a4l8xu.png">
</p>


<p align="center">
    <b>Infra for building multimodal embeddings from unstructured sources, built in Rust for speed and robustness</b>
</p>

EmbedAnything is a powerful library designed to streamline the creation and management of embedding pipelines. Whether you're working with text, images, audio, or any other type of data., EmbedAnything makes it easy to generate embeddings from multiple sources and store them efficiently in a vector database.



[![Watch the demo]](https://youtu.be/HLXIuznnXcI)



## 🚀 Key Features

- **Local Embedding** Works with local and OpenAI embedding
- **MultiModality** Works with text and image and will soon expand to audio
- - **Python Interface:** Packaged as a Python library for seamless integration into your existing projects.
- **Efficient:** Optimized for speed and performance, with core functionality written in Rust.
- **Scalable:** Store embeddings in a vector database for easy retrieval and scalability.


## 💡ToDo

- **Vector Database** Add functionalities to integrate with any Vector Database

## 💚 Installation

`
pip install embed-anything`


Requirements:

Please check if you already have the OpenAI key in the Environment variable. We have only released the OpenAI embedder library so far. Please stay tuned for updates for the local embeddings as well.


## :astronaut: Get Started:

```python
import embed_anything
from embed_anything import EmbedData
data = embed_anything.embed_file("filename.pdf")
```

