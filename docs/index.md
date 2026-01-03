

<p align ="center">
<img width=400 src = "https://res.cloudinary.com/dltwftrgc/image/upload/v1712504276/Projects/EmbedAnything_500_x_200_px_a4l8xu.png">
</p>





<div align="center">

  <p align="center">
    <b> Highly Performant, Modular and Memory Safe</b>
    <br />
    <b> Ingestion, Inference and Indexing in Rust 🦀</b>
    <br />
    <a href="https://embed-anything.com/references/">Python docs »</a>
    <br />
    <a href="https://docs.rs/embed_anything/latest/embed_anything/">Rust docs »</a>
    <br />
    <a href="https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#benchmarks"><strong>Benchmarks</strong></a>
    ·
    <a href="https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#%EF%B8%8Ffaq"><strong>FAQ</strong></a>
    ·
    <a href="https://github.com/StarlightSearch/EmbedAnything/tree/main/examples/adapters"><strong>Adapters</strong></a>
    .
    <a href="https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#-our-past-collaborations"><strong>Collaborations</strong></a>
    .
     <a href="https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#-notebooks"><strong>Notebooks</strong></a>


    
  </p>
</div>


EmbedAnything is a minimalist, yet highly performant, modular, lightning-fast, lightweight, multisource, multimodal, and local embedding pipeline built in Rust. Whether you're working with text, images, audio, PDFs, websites, or other media, EmbedAnything streamlines the process of generating embeddings from various sources and seamlessly streaming (memory-efficient-indexing) them to a vector database. It supports dense, sparse, ONNX, model2vec and late-interaction embeddings, offering flexibility for a wide range of use cases.

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



<iframe src="https://preview.mailerlite.io/forms/1975233/173424452135028700/share" width="100%" height="600" frameborder="0" style="border: 1px solid #ccc; border-radius: 8px; margin: 20px 0;"></iframe>

## 🚀 Key Features


- **No Dependency on Pytorch**: Easy to deploy on cloud, comes with low memory footprint.
- **Highly Modular** : Choose any vectorDB adapter for RAG, with ~~1 line~~ 1 word of code
- **Candle Backend** : Supports BERT, Jina, ColPali, Splade, ModernBERT, Reranker, Qwen
- **ONNX Backend**: Supports BERT, Jina, ColPali, ColBERT Splade, Reranker, ModernBERT, Qwen
- **Cloud Embedding Models:**: Supports OpenAI, Cohere, and Gemini.
- **MultiModality** : Works with text sources like PDFs, txt, md, Images JPG and Audio, .WAV
- **GPU support** : Hardware acceleration on GPU as well.
- **Chunking** : In-built chunking methods like semantic, late-chunking
- **Vector Streaming:** Separate file processing, Indexing and Inferencing on different threads, reduces latency.

## 💡What is Vector Streaming

 Embedding models are computationally expensive and time-consuming. By separating document preprocessing from model inference, you can significantly reduce pipeline latency and improve throughput.

Vector streaming transforms a sequential bottleneck into an efficient, concurrent workflow.

The embedding process happens separetly from the main process, so as to maintain high performance enabled by rust MPSC, and no memory leak as embeddings are directly saved to vector database. Find our [blog](https://starlight-search.com/blog/2025/02/25/vector%20database/).

[![EmbedAnythingXWeaviate](https://res.cloudinary.com/dltwftrgc/image/upload/v1731166897/demo_o8auu4.gif)](https://www.youtube.com/watch?v=OJRWPLQ44Dw)

## 🦀 Why Embed Anything 

➡️Faster execution. <br />
➡️No Pytorch Dependency, thus low-memory footprint and easy to deploy on cloud. <br />
➡️True multithreading <br />
➡️Running embedding models locally and efficiently <br />
➡️In-built chunking methods like semantic, late-chunking <br/>
➡️Supports range of models, Dense, Sparse, Late-interaction, ReRanker, ModernBert.<br />
➡️Memory Management: Rust enforces memory management simultaneously, preventing memory leaks and crashes that can plague other languages <br />

## 🍓 Our Past Collaborations:

We have collaborated with reputed enterprise like
[Elastic](https://www.youtube.com/live/OzQopxkxHyY?si=l6KasNNuCNOKky6f), [Weaviate](https://www.linkedin.com/posts/sonam-pankaj_machinelearning-data-ai-activity-7238832243622768644-gB8c?utm_source=share&utm_medium=member_desktop&rcm=ACoAABlF_IAB4Y74d5JJwj0CUwpTkhuskE0PAt4), [SingleStore](https://www.linkedin.com/events/buildingdomain-specificragappli7295319309566775297/theater/), [Milvus](https://milvus.io/docs/build_RAG_with_milvus_and_embedAnything.md) 
and [Analytics Vidya Datahours](https://community.analyticsvidhya.com/c/datahour/multimodal-embeddings-and-search-with-embed-anything-6adba0)

You can get in touch with us for further collaborations.

## Benchmarks

### Inference Speed benchmarks.
Only measures embedding model inference speed, on onnx-runtime. [Code](https://colab.research.google.com/drive/1nXvd25hDYO-j7QGOIIC0M7MDpovuPCaD?usp=sharing)

<img src="https://res.cloudinary.com/dltwftrgc/image/upload/v1730405688/embed_time_zusmua.png" width="500">


Benchmarks with other fromeworks coming soon!! 🚀
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
| Model2Vec | model2vec, minishlab/potion-base-8M |
| Qwen3-Embedding | Qwen/Qwen3-Embedding-0.6B |
| Reranker | [Jina Reranker Models](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual), Xenova/bge-reranker, Qwen/Qwen3-Reranker-4B |




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



### ReRankers
```python
reranker = Reranker.from_pretrained("jinaai/jina-reranker-v1-turbo-en", dtype=Dtype.F16)

results: list[RerankerResult] = reranker.rerank(["What is the capital of France?"], ["France is a country in Europe.", "Paris is the capital of France."], 2)
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
## 📒 Notebooks


|   |   
| ------------- | 
| [End-to-End Retrieval and Reranking using VectorDB Adapters](https://colab.research.google.com/drive/1gct0lEplyW8VWGPXUgpLcQuMQeZDl6D5?usp=sharing)  | 
| [ColPali-Onnx](https://colab.research.google.com/drive/1yCVbpkoe53ymiCxG8ttJNbRhECy1Q-Du?usp=sharing)  | 
| [Adapters](https://github.com/StarlightSearch/EmbedAnything/tree/main/examples/adapters) |  |
| [Qwen3- Embedings](https://colab.research.google.com/drive/1OlUJwTtPvj28h5tCVerf6ebEnAf8kPAh?usp=sharing) | 
| [Benchmarks](https://colab.research.google.com/drive/1nXvd25hDYO-j7QGOIIC0M7MDpovuPCaD?usp=sharing) | 


## 🚀 Running EmbedAnything as an OpenAI-Compatible Server

You can run EmbedAnything as an OpenAI-compatible API server using Actix-web. This provides a production-ready, high-performance server for generating embeddings.

To learn more about running the server, API endpoints, and usage examples, see [Actix Server Guide](/docs/guides/actix_server.md)

# Usage

## ➡️ Usage For 0.3 and later version

```python
model = EmbeddingModel.from_pretrained_local(
    WhichModel.Bert, model_id="Hugging_face_link"
)
data = embed_anything.embed_file("test_files/test.pdf", embedder=model)
```



### Using ONNX Models

To use ONNX models, you can either use the `ONNXModel` enum or the `model_id` from the Hugging Face model.


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
If you’d like to add support for your favorite vector database, we’d love to have your help! Check out our contribution.md for guidelines, or feel free to reach out directly turingatverge@gmail.com . Let's build something amazing together! 💡




## A big Thank you to all our StarGazers

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=StarlightSearch/EmbedAnything&type=Date)](https://star-history.com/#StarlightSearch/EmbedAnything&Date)
