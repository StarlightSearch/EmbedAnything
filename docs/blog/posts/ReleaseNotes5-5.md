---
draft: false 
date: 2025-05-25
authors: 
 - sonam
slug: release-notes-6
title: Release Notes 6.0
---

Super Excited to share the latest development in our library, which essentially giving you more embedding choices -- Cohere and siglip, new chunking method-- late chunking and more crates that facilitates amazing modality and maintainability for our rust codebase, --processor crate. so let's dive in.

<!-- more -->

## Late Chunking

The new 0.5.6 version adds Late Chunking to EmbedAnything, a technique introduced by Jina AI and Weaviate. 
Here's how we've implemented Late Chunking in EA:

𝗕𝗮𝘁𝗰𝗵 𝗮𝘀 𝗖𝗵𝘂𝗻𝗸 𝗚𝗿𝗼𝘂𝗽: In EmbedAnything, with late chunking enabled, the batch size determines the number of neighboring chunks that will be processed together.

𝗝𝗼𝗶𝗻𝘁 𝗘𝗺𝗯𝗲𝗱𝗱𝗶𝗻𝗴: The grouped chunks are fed into the embedding model as a single, larger input. This allows the model to capture relationships and dependencies between adjacent chunks.

𝗘𝗺𝗯𝗲𝗱𝗱𝗶𝗻𝗴 𝗦𝗽𝗹𝗶𝘁: After embedding, the combined output is divided back into the embeddings for the original, individual chunks.

𝗠𝗲𝗮𝗻 𝗣𝗼𝗼𝗹𝗶𝗻𝗴 (𝗽𝗲𝗿 𝗖𝗵𝘂𝗻𝗸): Mean pooling is then applied to each individual chunk's embedding, incorporating the contextual information learned during the joint embedding phase.

𝐾𝑒𝑦 𝐵𝑒𝑛𝑒𝑓𝑖𝑡𝑠:

𝗖𝗼𝗻𝘁𝗲𝘅𝘁-𝗔𝘄𝗮𝗿𝗲 𝗘𝗺𝗯𝗲𝗱𝗱𝗶𝗻𝗴𝘀: By embedding neighboring chunks together, we capture crucial contextual information that would be lost with independent chunking.

𝗢𝗽𝘁𝗶𝗺𝗶𝘇𝗲𝗱 𝗥𝗲𝘁𝗿𝗶𝗲𝘃𝗮𝗹 𝗣𝗲𝗿𝗳𝗼𝗿𝗺𝗮𝗻𝗰𝗲: Expect a significant improvement in the accuracy and relevance of your search results.

```python
model:EmbeddingModel = EmbeddingModel.from_pretrained_onnx(
    WhichModel.Jina, hf_model_id="jinaai/jina-embeddings-v2-small-en", path_in_repo="model.onnx"
)
config = TextEmbedConfig(
    chunk_size=1000,
    batch_size=8,
    splitting_strategy="sentence",
    late_chunking=True,
)

# Embed a single file
data: list[EmbedData] = model.embed_file("test_files/attention.pdf", config=config)
```


## Cohere Embed 4:

🧊 Single embedding per document, even for multimodal inputs
📚 Handles up to 128K tokens – perfect for long-form business documents
🗃️ Supports compressed vector formats (int8, binary) for real-world scalability
🌐 Multilingual across 100+ languages

The catch? It’s not open-source—and even if it were, the model would be quite hefty to run locally. But if you’re already using cloud-based embeddings like OpenAI’s, Embed v4 is worth testing.

```python
# Initialize the model once
model: EmbeddingModel = EmbeddingModel.from_pretrained_cloud(
    WhichModel.CohereVision, model_id="embed-v4.0"
)

```

## SigLIP

We already had Clip support but many of you asked for siglip support. It out performs clip for zero shot classification for smaller batch. It also has better memory efficinecy.

```python
# Load the model.
model = embed_anything.EmbeddingModel.from_pretrained_hf(
    embed_anything.WhichModel.Clip,
    model_id="google/siglip-base-patch16-224",
)
```

## Processor Crate:

This crate contains various "processors" that accepts files and produces a chunked, metadata-rich document description. This is especially helpful for retrieval-augmented generation! 

We have also received some additional cool feature requests on GitHub, which we would like to implement. If you want to help out please check out EmbedAnything on GitHub. We would love to have a contribution. 🚀



