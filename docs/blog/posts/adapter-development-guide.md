---
draft: false 
date: 2025-02-25
authors: 
 - sonam
 - akshay
slug: vector database
title: How to write adapters for your vector database.
---
We have received multiple requests to add different vector database for our [vector streaming](https://starlight-search.com/blog/2024/03/31/vector-streaming/). So We have decided to put a detailed guide for different vector databases out there. We are happy to accept pull-request.

<!-- more -->
## Creating Custom Adapters for EmbedAnything: A Step-by-Step Guide

In the world of machine learning and natural language processing, working with embeddings has become a fundamental task. The EmbedAnything library simplifies the process of generating embeddings from various data sources, but what if you want to store these embeddings in a specific database or service? This is where adapters come in. In this blog post, we'll walk through the process of creating a custom adapter for the EmbedAnything library, using the Pinecone vector database as an example.

## Understanding Adapters in EmbedAnything

Adapters serve as bridges between the EmbedAnything library and external services or databases. They handle the conversion and storage of embeddings, allowing you to seamlessly integrate EmbedAnything with your preferred storage solution.

## The Anatomy of an Adapter

Before diving into the code, let's understand the key components of an adapter:

1. **Initialization**: Setting up the connection to the external service
2. **Index Management**: Creating and deleting indices in the external service
3. **Data Conversion**: Transforming EmbedAnything's embedding format to the format required by the external service
4. **Data Storage**: Storing the converted embeddings in the external service

## Creating a Pinecone Adapter: Step by Step

Let's break down the process of creating a Pinecone adapter for EmbedAnything:

### Step 1: Set Up the Basic Class Structure

```python
from embed_anything import EmbedData, EmbeddingModel, WhichModel, TextEmbedConfig

class PineconeAdapter(Adapter):
    """
    Adapter class for interacting with Pinecone, a vector database service.
    """
    def __init__(self, api_key: str):
        """
        Initializes a new instance of the PineconeAdapter class.
        
        Args:
            api_key (str): The API key for accessing the Pinecone service.
        """
        super().__init__(api_key)
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = None
```

In this step, we're:
- Inheriting from the base `Adapter` class provided by EmbedAnything
- Initializing the Pinecone client using the provided API key
- Setting up an attribute to track the current index name

### Step 2: Implement Index Management Methods

```python
def create_index(
    self, 
    dimension: int, 
    metric: str = "cosine", 
    index_name: str = "anything",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
):
    """
    Creates a new index in Pinecone.
    
    Args:
        dimension (int): The dimensionality of the embeddings.
        metric (str, optional): The distance metric to use for similarity search. Defaults to "cosine".
        index_name (str, optional): The name of the index. Defaults to "anything".
        spec (ServerlessSpec, optional): The serverless specification for the index. Defaults to AWS in us-east-1 region.
    """
    self.index_name = index_name
    self.pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=spec
    )

def delete_index(self, index_name: str):
    """
    Deletes an existing index from Pinecone.
    
    Args:
        index_name (str): The name of the index to delete.
    """
    self.pc.delete_index(name=index_name)
```

These methods handle:
- Creating a new index in Pinecone with the specified dimensions and distance metric
- Deleting an existing index when needed
- Storing the current index name for later use

### Step 3: Implement Data Conversion Logic

```python
def convert(self, embeddings: List[EmbedData]) -> List[Dict]:
    """
    Converts a list of embeddings into the required format for upserting into Pinecone.
    
    Args:
        embeddings (List[EmbedData]): The list of embeddings to convert.
    
    Returns:
        List[Dict]: The converted data in the required format for upserting into Pinecone.
    """
    data_emb = []
    for embedding in embeddings:
        data_emb.append(
            {
                "id": str(uuid.uuid4()),
                "values": embedding.embedding,
                "metadata": {
                    "text": embedding.text,
                    "file": re.split(
                        r"/|\\", embedding.metadata.get("file_name", "")
                    )[-1],
                },
            }
        )
    return data_emb
```

This method:
- Takes a list of `EmbedData` objects from EmbedAnything
- Converts each embedding into the format expected by Pinecone
- Generates a unique ID for each embedding
- Extracts and formats metadata from the original embedding

### Step 4: Implement Storage Logic

```python
def upsert(self, data: List[Dict]):
    """
    Upserts data into the specified index in Pinecone.
    
    Args:
        data (List[Dict]): The data to upsert into Pinecone.
    
    Raises:
        ValueError: If the index has not been created before upserting data.
    """
    data = self.convert(data)
    if not self.index_name:
        raise ValueError("Index must be created before upserting data")
    self.pc.Index(name=self.index_name).upsert(data)
```

This method:
- Converts the input data using the `convert` method
- Checks if an index has been created before attempting to upsert data
- Upserts the converted data into the specified Pinecone index

## Using Your Custom Adapter

Once you've created your adapter, you can use it with EmbedAnything like this:

```python
# Initialize the PineconeEmbedder class
api_key = os.environ.get("PINECONE_API_KEY")
index_name = "anything"
pinecone_adapter = PineconeAdapter(api_key)

# Delete existing index if it exists
try:
    pinecone_adapter.delete_index("anything")
except:
    pass

# Create a new index
pinecone_adapter.create_index(dimension=512, metric="cosine")

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

print(data)
```

## Conclusion

Creating custom adapters for EmbedAnything allows you to seamlessly integrate the library with your preferred storage solutions. By following the step-by-step guide and best practices outlined in this blog post, you can create robust and efficient adapters that enhance your embedding workflow.

Remember, the key to a good adapter is clear documentation, robust error handling, and efficient data conversion. With these principles in mind, you can extend EmbedAnything to work with virtually any storage solution.

Happy coding!
