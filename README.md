

<p align ="center">
<img width=400 src = "https://res.cloudinary.com/dltwftrgc/image/upload/v1712504276/Projects/EmbedAnything_500_x_200_px_a4l8xu.png">
</p>



<div align="center">

[![Downloads](https://static.pepy.tech/badge/embed-anything)](https://pepy.tech/project/embed-anything)
[![gpu](https://static.pepy.tech/badge/embed-anything-gpu)](https://www.pepy.tech/projects/embed-anything-gpu)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CowJrqZxDDYJzkclI-rbHaZHgL9C6K3p?usp=sharing)
[![roadmap](https://img.shields.io/badge/Discord-%235865F2.svg?style=flat&logo=discord&logoColor=white)](https://discord.gg/juETVTMdZu)
[![MkDocs](https://img.shields.io/badge/Blogs-F38020?.svg?logoColor=fff)](https://embed-anything.com/blog/)

</div>


<div align="center">

  <p align="center">
    <b> Inference, Ingestion, and Indexing in Rust 🦀</b>
    <br />
    <a href="https://embed-anything.com/references/">Python docs »</a>
    <br />
    <a href="https://docs.rs/embed_anything/latest/embed_anything/">Rust docs »</a>
    <br />
    <a href="https://colab.research.google.com/drive/1nXvd25hDYO-j7QGOIIC0M7MDpovuPCaD?usp=sharing"><strong>Benchmarks</strong></a>
    ·
    <a href="https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#%EF%B8%8Ffaq"><strong>FAQ</strong></a>
    ·
    <a href="https://github.com/StarlightSearch/EmbedAnything/tree/main/examples/adapters"><strong>Adapters</strong></a>
    .
    <a href="https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#-our-past-collaborations"><strong>Collaborations</strong></a>


    
  </p>
</div>


EmbedAnything is a minimalist, yet highly performant, lightning-fast, lightweight, multisource, multimodal, and local embedding pipeline built in Rust. Whether you're working with text, images, audio, PDFs, websites, or other media, EmbedAnything streamlines the process of generating embeddings from various sources and seamlessly streaming (memory-efficient-indexing) them to a vector database. It supports dense, sparse, ONNX, model2vec and late-interaction embeddings, offering flexibility for a wide range of use cases.

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

- **Candle Backend** : Supports BERT, Jina, ColPali, Splade, ModernBERT
- **ONNX Backend**: Supports BERT, Jina, ColPali, ColBERT Splade, Reranker, ModernBERT
- **Cloud Embedding Models:**: Supports OpenAI and Cohere.  
- **MultiModality** : Works with text sources like PDFs, txt, md, Images JPG and Audio, .WAV
- **Rust** : All the file processing is done in rust for speed and efficiency
- **GPU support** : We have taken care of hardware acceleration on GPU as well.
- **Python Interface:** Packaged as a Python library for seamless integration into your existing projects.
- **Vector Streaming:** Continuously create and stream embeddings if you have low resource.
- **No Dependency on Pytorch** Easy to deploy on cloud, as it comes with low memory footprint.

## 💡What is Vector Streaming

Vector Streaming enables you to process and generate embeddings for files and stream them, so if you have 10 GB of file, it can continuously generate embeddings Chunk by Chunk, that you can segment semantically, and store them in the vector database of your choice, Thus it eliminates bulk embeddings storage on RAM at once. 

The embedding process happens separetly from the main process, so as to maintain high performance enabled by rust MPSC, and no memory leak as embeddings are directly saved to vector database. Find our [blog](https://starlight-search.com/blog/2025/02/25/vector%20database/).

[![EmbedAnythingXWeaviate](https://res.cloudinary.com/dltwftrgc/image/upload/v1731166897/demo_o8auu4.gif)](https://www.youtube.com/watch?v=OJRWPLQ44Dw)

## 🦀 Why Embed Anything 

➡️Faster execution. <br />
➡️No Pytorch Dependency, thus low-memory footprint and easy to deploy on cloud. <br />
➡️Memory Management: Rust enforces memory management simultaneously, preventing memory leaks and crashes that can plague other languages <br />
➡️True multithreading <br />
➡️Running embedding models locally and efficiently <br />
➡️Candle allows inferences on CUDA-enabled GPUs right out of the box. <br />
➡️Decrease the memory usage. <br/>
➡️Supports range of models, Dense, Sparse, Late-interaction, ReRanker, ModernBert.

## 🍓 Our Past Collaborations:

We have collaborated with reputed enterprise like
[Elastic](https://www.youtube.com/live/OzQopxkxHyY?si=l6KasNNuCNOKky6f), [Weaviate](), [SingleStore](https://www.linkedin.com/events/buildingdomain-specificragappli7295319309566775297/theater/) [Milvus](https://milvus.io/docs/build_RAG_with_milvus_and_embedAnything.md) 
and [Analytics Vidya Datahours](https://community.analyticsvidhya.com/c/datahour/multimodal-embeddings-and-search-with-embed-anything-6adba0)

You can get in touch with us for further collaborations.

## Benchmarks

Only measures embedding model inference speed, on onnx-runtime. [Code](https://colab.research.google.com/drive/1nXvd25hDYO-j7QGOIIC0M7MDpovuPCaD?usp=sharing)

<img src="https://res.cloudinary.com/dltwftrgc/image/upload/v1730405688/embed_time_zusmua.png" width="500">

# ⭐ Supported Models

We support any hugging-face models on Candle. And We also support ONNX runtime for BERT and ColPali.

## How to add custom model on candle: from_pretrained_hf
```python
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Bert, model_id="model link from huggingface"
)
config = TextEmbedConfig(chunk_size=1000, batch_size=32)
data = embed_anything.embed_file("file_address", embedder=model, config=config)
```


| Model  | HF link |
| ------------- | ------------- | 
| Jina  | [Jina Models](https://huggingface.co/collections/jinaai/jina-embeddings-v2-65708e3ec4993b8fb968e744) | 
| Bert | All Bert based models |
| CLIP | openai/clip-* | 
| Whisper| [OpenAI Whisper models](https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013)|
| ColPali | starlight-ai/colpali-v1.2-merged-onnx|
| Colbert | answerdotai/answerai-colbert-small-v1, jinaai/jina-colbert-v2 and more |
| Splade | [Splade Models](https://huggingface.co/collections/naver/splade-667eb6df02c2f3b0c39bd248) and other Splade like models |
| Reranker | [Jina Reranker Models](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual), Xenova/bge-reranker |
| Model2Vec | model2vec, minishlab/potion-base-8M |
| Qwen3-Embedding | Qwen/Qwen3-Embedding-0.6B |


## Splade Models:

```python
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.SparseBert, "prithivida/Splade_PP_en_v1"
)
```

## ONNX-Runtime: from_pretrained_onnx

### BERT

```python
model = EmbeddingModel.from_pretrained_onnx(
  WhichModel.Bert, model_id="onnx_model_link"
)
```

### ColPali

```python
model: ColpaliModel = ColpaliModel.from_pretrained_onnx("starlight-ai/colpali-v1.2-merged-onnx", None)
```

### Colbert

```python
sentences = [
"The quick brown fox jumps over the lazy dog", 
"The cat is sleeping on the mat", "The dog is barking at the moon", 
"I love pizza", 
"The dog is sitting in the park"]

model = ColbertModel.from_pretrained_onnx("jinaai/jina-colbert-v2", path_in_repo="onnx/model.onnx")
embeddings = model.embed(sentences, batch_size=2)
```

### ModernBERT

```python
model = EmbeddingModel.from_pretrained_onnx(
    WhichModel.Bert, ONNXModel.ModernBERTBase, dtype = Dtype.Q4F16
)
```

### ReRankers
```python
reranker = Reranker.from_pretrained("jinaai/jina-reranker-v1-turbo-en", dtype=Dtype.F16)

results: list[RerankerResult] = reranker.rerank(["What is the capital of France?"], ["France is a country in Europe.", "Paris is the capital of France."], 2)
```

### Embed 4

```python
# Initialize the model once
model: EmbeddingModel = EmbeddingModel.from_pretrained_cloud(
    WhichModel.CohereVision, model_id="embed-v4.0"
)

```

### Qwen 3 - Embedding

```python
# Initialize the model once
model:EmbeddingModel = EmbeddingModel.from_pretrained_hf(
    WhichModel.Qwen3, model_id="Qwen/Qwen3-Embedding-0.6B"
)
```


## For Semantic Chunking

```python
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Bert, model_id="sentence-transformers/all-MiniLM-L12-v2"
)

# with semantic encoder
semantic_encoder = EmbeddingModel.from_pretrained_hf(WhichModel.Jina, model_id = "jinaai/jina-embeddings-v2-small-en")
config = TextEmbedConfig(chunk_size=1000, batch_size=32, splitting_strategy = "semantic", semantic_encoder=semantic_encoder)

```

## For late-chunking
```python
config = TextEmbedConfig(
    chunk_size=1000,
    batch_size=8,
    splitting_strategy="sentence",
    late_chunking=True,
)

# Embed a single file
data: list[EmbedData] = model.embed_file("test_files/attention.pdf", config=config)

```

# 🧑‍🚀 Getting Started

## 💚 Installation

`
pip install embed-anything
`<br/>

For GPUs and using special models like ColPali <br/>

`
pip install embed-anything-gpu
`

🚧❌ If it shows cuda error while running on windowns, run the following command:

```
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin")
```

# Usage



## ➡️ Usage For 0.3 and later version


### To use local embedding: we support Bert and Jina

```python
model = EmbeddingModel.from_pretrained_local(
    WhichModel.Bert, model_id="Hugging_face_link"
)
data = embed_anything.embed_file("test_files/test.pdf", embedder=model)
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
data: list[EmbedData] = embed_anything.embed_image_directory("test_files", embedder=model)
embeddings = np.array([data.embedding for data in data])
query = ["Photo of a monkey?"]
query_embedding = np.array(
    embed_anything.embed_query(query, embedder=model)[0].embedding
)
similarities = np.dot(embeddings, query_embedding)
max_index = np.argmax(similarities)
Image.open(data[max_index].text).show()
```

### Using ONNX Models

To use ONNX models, you can either use the `ONNXModel` enum or the `model_id` from the Hugging Face model.

```python
model = EmbeddingModel.from_pretrained_onnx(
  WhichModel.Bert, model_name = ONNXModel.AllMiniLML6V2Q
)
```

For some models, you can also specify the dtype to use for the model.

```python
model = EmbeddingModel.from_pretrained_onnx(
    WhichModel.Bert, ONNXModel.ModernBERTBase, dtype = Dtype.Q4F16
)
```

Using the above method is best to ensure that the model works correctly as these models are tested. But if you want to use other models, like finetuned models, you can use the `hf_model_id` and `path_in_repo` to load the model like below.

```python
model = EmbeddingModel.from_pretrained_onnx(
  WhichModel.Jina, hf_model_id = "jinaai/jina-embeddings-v2-small-en", path_in_repo="model.onnx"
)
```
To see all the ONNX models supported with model_name, see [here](/docs/guides/onnx_models.md)

## ⁉️FAQ

### Do I need to know rust to use or contribute to embedanything?
The answer is No. EmbedAnything provides you pyo3 bindings, so you can run any function in python without any issues. To contibute you should check out our guidelines and python folder example of adapters.

### How is it different from fastembed?

We provide both backends, candle and onnx. On top of it we also give an end-to-end pipeline, that is you can ingest different data-types and index to any vector database, and inference any model. Fastembed is just an onnx-wrapper.

### We've received quite a few questions about why we're using Candle.

One of the main reasons is that Candle doesn't require any specific ONNX format models, which means it can work seamlessly with any Hugging Face model. This flexibility has been a key factor for us. However, we also recognize that we’ve been compromising a bit on speed in favor of that flexibility.


## 🚧 Contributing to EmbedAnything

First of all, thank you for taking the time to contribute to this project. We truly appreciate your contributions, whether it's bug reports, feature suggestions, or pull requests. Your time and effort are highly valued in this project. 🚀

This document provides guidelines and best practices to help you to contribute effectively. These are meant to serve as guidelines, not strict rules. We encourage you to use your best judgment and feel comfortable proposing changes to this document through a pull request.



<li><a href="##-RoadMap">Roadmap</a></li>
<li><a href="##-Quick-Start">Quick Start</a></li>
<li><a href="##-Contributing-Guidelines">Guidelines</a></li>


# 🏎️ RoadMap 

## Accomplishments

One of the aims of EmbedAnything is to allow AI engineers to easily use state of the art embedding models on typical files and documents. A lot has already been accomplished here and these are the formats that we support right now and a few more have to be done. <br />

### Adding Fine-tuning 
One of the major goals of this year is to add finetuning these models on your data. Like a simple sentence transformer does.

### 🖼️ Modalities and Source

We’re excited to share that we've expanded our platform to support multiple modalities, including:

- [x] Audio files

- [x] Markdowns

- [x] Websites

- [x] Images

- [ ] Videos

- [ ] Graph

This gives you the flexibility to work with various data types all in one place! 🌐 <br />



### ⚙️ Performance 


We now support both candle and Onnx backend<br/>
➡️ Support for GGUF models </br >


### 🫐Embeddings:

We had multimodality from day one for our infrastructure. We have already included it for websites, images and audios but we want to expand it further to.

➡️ Graph embedding -- build deepwalks embeddings depth first and word to vec <br />
➡️ Video Embedding <br/>
➡️ Yolo Clip <br/>


### 🌊Expansion to other Vector Adapters

We currently support a wide range of vector databases for streaming embeddings, including:

- Elastic: thanks to amazing and active Elastic team for the contribution <br/>
- Weaviate <br/>
- Pinecone <br/>
- Qdrant <br/>
- Milvus<br/>
- Chroma <br/>

How to add an adpters: https://starlight-search.com/blog/2024/02/25/adapter-development-guide.md

### 💥 Create WASM demos to integrate embedanything directly to the browser. <br/>

### 💜 Add support for ingestion from remote sources
➡️ Support for S3 bucket </br >
➡️ Support for azure storage </br >
➡️ Support for google drive/dropbox</br >




But we're not stopping there! We're actively working to expand this list.

Want to Contribute?
If you’d like to add support for your favorite vector database, we’d love to have your help! Check out our contribution.md for guidelines, or feel free to reach out directly starlight-search@proton.me. Let's build something amazing together! 💡

## A big Thank you to all our StarGazers

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=StarlightSearch/EmbedAnything&type=Date)](https://star-history.com/#StarlightSearch/EmbedAnything&Date)
