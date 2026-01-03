---
draft: false 
date: 2025-11-01
authors: 
 - sonam
slug: memory_leak
title: Memory Leak Explained!
---
Vector Streaming is popular for its low latency and high modularity features. But the best of it all is the memory leak that we are trying to bring it but haven’t been able to click with the audience yet. Why we named it vector streaming and how it plays a major role in memory management.


<!-- more -->

## Why Memory Management Matters When Working with Embeddings

When you run inference on an embedding model, you're converting text chunks into high-dimensional vector representations. In typical implementations, these embeddings accumulate in RAM during processing, which creates a significant risk of memory leaks—especially when dealing with large document collections or long-running processes.

### The Problem with Traditional Approaches

Consider a standard embedding pipeline: you load documents, chunk them, generate embeddings, and then... what happens to those embeddings while you're processing the next batch? They sit in memory. And if your indexing process is slow or if there's an interruption, those embeddings can pile up, consuming RAM until your application crashes or becomes unresponsive.

![image.png](https://royal-hygienic-522.notion.site/image/attachment%3A75ffa0ba-ba5d-4de1-a47d-1094320461b9%3Aimage.png?table=block&id=29281b6a-6bbe-802d-9360-d9c3dadc11b1&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=2000&userId=&cache=v2)

## What is vector streaming?

Embedding models are computationally expensive and time-consuming. By separating document preprocessing from model inference, you can significantly reduce pipeline latency and improve throughput.

Vector streaming transforms a sequential bottleneck into an efficient, concurrent workflow.

The embedding process occurs separately from the main process, allowing for high performance enabled by Rust's MPSC and preventing memory leaks, as embeddings are directly saved to a vector database. Find our [blog](https://starlight-search.com/blog/2025/02/25/vector%20database/).

### The Solution: Vector Streaming

The most effective way to prevent these memory issues is to **stream embeddings directly to your vector database** rather than buffering them in RAM. This approach ensures that once an embedding is generated, it's immediately persisted and can be freed from memory.

This is exactly what EmbedAnything accomplishes through its vector streaming architecture. By separating document preprocessing, model inference, and indexing onto different threads using Rust's MPSC (multi-producer, single-consumer) channels, embeddings flow directly from the model to your vector database without accumulating in memory.

### Benefits of This Approach

- **No memory leaks**: Embeddings are persisted immediately rather than queued in RAM
- **Reduced latency**: Concurrent processing of file parsing, embedding generation, and indexing
- **Better throughput**: Transform a sequential bottleneck into an efficient, parallel workflow
- **Production-ready**: Rust's memory safety guarantees prevent the crashes and leaks common in other languages

Vector streaming transforms what would typically be a memory-intensive operation into a lightweight, efficient pipeline—perfect for production environments where reliability and performance are critical.

HOW vector streaming:

The best way to create an inference pipeline with no memory leak is to save the embeddings generated directly on the vector database, rather than in RAM.

How do we do it? also extremely unique.

We set a buffer size that you can customize according to your hardware. And send each chunk, (to know how to set chunksize and buffer size because your retrieved results are highly based on these two factors, refer to this blog.)

Inference that is the conversation of chunks to embeddings and storage to vector databases and all other process happens on different threads, that gives us a unique way to be able to store the embeddings not on RAM but on to the vector database itself.

![image.png](https://royal-hygienic-522.notion.site/image/attachment%3A7c887312-ec4f-412a-8c20-2dbd4af195a1%3Aimage.png?table=block&id=29281b6a-6bbe-80d3-82d7-df11923a5068&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=1420&userId=&cache=v2)

Results of Vector Streaming have been tested on standard text data, with our Elastic Adapter available. We also support Weaviate, Milvus, Qdrant, Lance and many more and it’s super easy to add your favourite vector database, just read our [blog](https://embed-anything.com/blog/2025/09/15/semantic-late-chunking/).

We also have 

![image.png](https://royal-hygienic-522.notion.site/image/attachment%3Af16bb75d-3ac3-40ac-a106-9a1761a8059e%3Aimage.png?table=block&id=29381b6a-6bbe-8083-b7c8-ee21bf62d44c&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=1420&userId=&cache=v2)

## How to write it, an example.

There are three parts of writing vector streaming with an already existing adapter.

```python
# Initialize the embedding model

model = EmbeddingModel.from_pretrained_hf(

WhichModel.Bert,

model_id="sentence-transformers/all-MiniLM-L12-v2"

)

# Embed a PDF file

data = embed_anything.embed_file(

"test-file",

embedder=model,

adapter=pinecone_adapter,

)

# Embed all images in a directory

data = embed_anything.embed_image_directory(

"test_files",

embedder=model,

adapter=pinecone_adapter

)
```

We also released an Actix Server that can be used directly, more on that later. I hope you found this blog useful in understanding how memory leaks work and how our design helps in overcoming this issue.