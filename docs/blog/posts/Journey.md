---
draft: false 
date: 2024-12-15
authors: 
 - sonam
slug: embed-anything
title: The path ahead of EmbedAnything
---
In March, we set out to build a local file search app. We aimed to create a tool that would make file searching faster, more innovative, and more efficient. However, we quickly hit a roadblock: no high-performance backend fit our needs.

<!-- more -->

![image.png](https://royal-hygienic-522.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Ff1bf59bf-2c3f-4b4d-a5f9-109d041ef45a%2Faa8abe48-4210-494c-af98-458b6694b09a%2Fimage.png?table=block&id=15d81b6a-6bbe-80cc-883e-fcafd65e619d&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=1420&userId=&cache=v2)

### Short of backend

Initially, we experimented with LlamaIndex, hoping it would provide the required speed and reliability. Unfortunately, it fell short. Its performance didn’t meet our expectations, and its heavy dependencies added unnecessary complexity to our stack. We realized we needed a better solution.

Around the same time, we discovered **Candle**, a Rust-based framework for transformer model inference. Candle stood out with its remarkable speed and minimal dependency footprint. It was exactly what we were looking for a high-performing, lightweight backend that aligned with our vision for a seamless file search experience.

### Experimentation and Breakthroughs

Excited by Candle’s potential, we experimented to see how well it could handle our use case. The results were outstanding. Candle’s blazing-fast inference speeds and low resource demands enabled us to build a prototype that surpassed our initial performance goals.

With a working prototype, we decided to share it with the world. We knew a compelling demonstration could capture attention and validate our efforts. The next step was to make a splash with our launch.

### Demo Released

On **April 2nd**, we unveiled our demo online, carefully choosing the date to avoid confusion with April Fool’s Day. We created an engaging demo video to highlight the app’s capabilities and shared it on Twitter. What happened next exceeded all our expectations.

The demo received an overwhelming response. What began as a simple showcase of our prototype transformed into a pivotal moment for our project. In the next 30 days, we released it as an open-source project, seeing the demand and people’s interest.

[Demo](https://www.youtube.com/watch?v=HLXIuznnXcI)
### 0.2 released

Since then, we have never looked back. We kept embedding anything better and better. In the next three months, we released a more stable version, 0.2, with all the Python versions. It was running amazingly on AWS and could support multimodality.

At the same time, we realized that people wanted an end-to-end solution, not just an embedding generation platform. So we tried to integrate a vector database, but we realized that it would just make our library heavier and not give the value we were looking for, which was confirmed by this discussion opened on our GitHub.

[—GitHub discussion](https://github.com/StarlightSearch/EmbedAnything/discussions/44#discussion-6953627)

Akshay started looking for ways to index embeddings without being dependent on vector databases as a dependency, and he came up with a brilliant method that enhanced performance and made indexing extremely memory efficient.

And thus, vector streaming was born.

— [vector streaming blog](https://starlight-search.com/blog/2024/01/31/vector-streaming/)

### 0.3 release

It's time to release 0.3 because we underwent major code refactoring. All the major functions are refactored, making calling models more intuitive and optimized. Check out our docs and usage. We also added audio modality and different types of ingestions.

We only supported dense, so we expanded the types of embedding we could support. We went for sparse and started supporting ColPali, ColBert, ModernBert, Reranker, Jina V3.

## What We Got Right

We actively listened to our community and prioritized their needs in the library's development. When users requested support for sparse matrices in hybrid models, we delivered. When they wanted advanced indexing, we made it happen. During the critical three-month period between versions 0.2 and 0.4, our efforts were laser-focused on enhancing the product to meet and exceed expectations. 

We also released benches comparing it with other inference and to our suprise it's faster than libraries like sentence transformer and fastembed. Check out [Benches](https://colab.research.google.com/drive/1nXvd25hDYO-j7QGOIIC0M7MDpovuPCaD?usp=sharing).


We presented Embedanything at many conferences, like Pydata Global, Elastic, voxel 51 meetups, AI builders, etc. Additionally, we forged collaborations with major brands like Weaviate and Elastic, a strategy we’re excited to continue expanding in 2025.

[Elastic Collab](https://www.youtube.com/live/OzQopxkxHyY?si=shJ2hADyPPsYWmIF)


## What We Initially Got Wrong

In hindsight, one significant mistake was prematurely releasing the library before it was ready for production. As the saying goes, “You never get a second chance to make a first impression,” and this holds true even for open-source projects.

The library was unusable on macOS for the first three months, and we only released compatibility with Python 10. We didn’t focus enough on how we were rolling out updates, partly because we never anticipated the overwhelming rate of experimentation and interest it would receive right from the start.

I intended to foster a “build in public” project, encouraging collaboration and rapid iteration. I wanted to showcase how quickly we could improve and refine this amazing library. 

### In the year 2025

We are committed to applying everything we’ve learned from this journey and doubling down on what truly matters: our hero, the product. In the grand scheme of things, nothing else is as important. Moving forward, we’re also excited to announce even more collaborations with amazing brands, further expanding the impact and reach of our work.

Heartfelt thanks to all our amazing contributors and stargazers for your unwavering support and dedication to *embedanything*. Your continuous experimentation and feedback inspire us to keep refining and enhancing the library with every iteration. We deeply appreciate your efforts in making this journey truly collaborative. Let’s go from 100k+ to a million downloads this year!