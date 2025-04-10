
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

### 💜 Product
We’ve rolled out some major updates in version 0.3 to improve both functionality and performance. Here’s what’s new:

- Semantic Chunking: Optimized chunking strategy for better Retrieval-Augmented Generation (RAG) workflows.

- Streaming for Efficient Indexing: We’ve introduced streaming for memory-efficient indexing in vector databases. Want to know more? Check out our article on this feature here: https://www.analyticsvidhya.com/blog/2024/09/vector-streaming/

- Zero-Shot Applications: Explore our zero-shot application demos to see the power of these updates in action.

- Intuitive Functions: Version 0.3 includes a complete refactor for more intuitive functions, making the platform easier to use.

- Chunkwise Streaming: Instead of file-by-file streaming, we now support chunkwise streaming, allowing for more flexible and efficient data processing.

Check out the latest release :  and see how these features can supercharge your GenerativeAI pipeline! ✨

## 🚀Coming Soon  <br />

### ⚙️ Performance 
We've received quite a few questions about why we're using Candle, so here's a quick explanation:

One of the main reasons is that Candle doesn't require any specific ONNX format models, which means it can work seamlessly with any Hugging Face model. This flexibility has been a key factor for us. However, we also recognize that we’ve been compromising a bit on speed in favor of that flexibility.

What’s Next?
To address this, we’re excited to announce that we’re introducing Candle-ONNX along with our previous framework on hugging-face ,

➡️ Support for GGUF models </br >
- Significantly faster performance</br >
- Stay tuned for these exciting updates! 🚀</br >


### 🫐Embeddings:

We had multimodality from day one for our infrastructure. We have already included it for websites, images and audios but we want to expand it further to.

☑️Graph embedding -- build deepwalks embeddings depth first and word to vec <br />
☑️Video Embedding <br/>
☑️ Yolo Clip <br/>


### 🌊Expansion to other Vector Adapters

We currently support a wide range of vector databases for streaming embeddings, including:

- Elastic: thanks to amazing and active Elastic team for the contribution <br/>
- Weaviate<br/>
- Pinecone<br/>
- Qdrant<br/>
- Milvus<br/>

But we're not stopping there! We're actively working to expand this list.

Want to Contribute?
If you’d like to add support for your favorite vector database, we’d love to have your help! Check out our contribution.md for guidelines, or feel free to reach out directly starlight-search@proton.me. Let's build something amazing together! 💡
