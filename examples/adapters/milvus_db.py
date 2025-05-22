from pymilvus import MilvusClient, DataType
import os
from typing import Dict, List

import embed_anything
from embed_anything.vectordb import Adapter
from embed_anything import EmbedData, EmbeddingModel, WhichModel

print("Milvus Vector DB - Adapter")

# Default embedding dimension
EMBEDDINGS_DIM = 384
# Maximum VARCHAR field length for text content
TEXT_CONTENT_VARCHARS = 4098

# Type annotation for embeddings
VectorEmbeddings = List[List[EmbedData]]

class MilvusVectorAdapter(Adapter):
    def __init__(self, uri: str = './milvus.db', token: str = '', collection_name: str = "embed_anything_collection"):
        """
        Initialize the MilvusVectorAdapter.
        
        Args:
            uri (str): The URI to connect to, comes in the form of
                "https://address:port" for Milvus or Zilliz Cloud service,
                or "path/to/local/milvus.db" for the lite local Milvus. Defaults to
                "./milvus.db".
            token (str): The token for log in. Defaults to "".
            collection_name (str): Name of the collection to use. Defaults to
                "embed_anything_collection".
        """
        self.collection_name = collection_name
        self.client = MilvusClient(uri=uri, token=token)
        print("Ok - Milvus DB connection established.")

    def create_index(self, dimension: int = EMBEDDINGS_DIM):
        """
        Create a collection and index for embeddings.
        
        Args:
            dimension: Dimension of the embedding vectors.
            **kwargs: Additional parameters for index creation.
        """
        # Delete collection if it exists
        if self.client.has_collection(self.collection_name):
            self.delete_index()
        
        # Create collection schema
        schema = self.client.create_schema(auto_id=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="embeddings",
            datatype=DataType.FLOAT_VECTOR,
            dim=dimension
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=TEXT_CONTENT_VARCHARS
        )
        schema.add_field(
            field_name="file_name",
            datatype=DataType.VARCHAR,
            max_length=255
        )
        schema.add_field(
            field_name="modified",
            datatype=DataType.VARCHAR,
            max_length=50
        )
        schema.add_field(
            field_name="created",
            datatype=DataType.VARCHAR,
            max_length=50
        )
        
        # Create the collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema
        )
        
        # Create the index
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embeddings",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 1024}
        )
        
        # Apply the index
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )
        
        # Load the collection
        self.client.load_collection(
            collection_name=self.collection_name
        )
        
        print(f"Collection '{self.collection_name}' created with index.")

    def convert(self, embeddings: List[EmbedData]) -> List[Dict]:
        """
        Convert EmbedData objects to a format compatible with Milvus.
        
        Args:
            embeddings: List of EmbedData objects.
            
        Returns:
            List of dictionaries with data formatted for Milvus.
        """
        ret_data = []
        for i, embedding in enumerate(embeddings):
            data_dict = {
                "embeddings": embedding.embedding,
                "text": embedding.text,
                "file_name": embedding.metadata["file_name"],
                "modified": embedding.metadata["modified"],
                "created": embedding.metadata["created"],
            }
            ret_data.append(data_dict)
        
        print(f"Converted {len(ret_data)} embeddings for insertion.")
        return ret_data

    def delete_index(self):
        """
        Delete the collection and its index.
        """
        try:
            self.client.drop_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' dropped.")
        except Exception as e:
            print(f"Failed to drop collection: {e}")
   

    def upsert(self, data: List[EmbedData]):
        """
        Insert or update embeddings in the collection.
        
        Args:
            data: List of EmbedData objects to insert.
        """
        # Convert data to Milvus format
        formatted_data = self.convert(data)
        
        # Insert data
        self.client.insert(
            collection_name=self.collection_name,
            data=formatted_data
        )
        
        print(f"Successfully inserted {len(formatted_data)} embeddings.")
    




if __name__ == "__main__":
    # Initialize the MilvusVectorAdapter class
    index_name = "embed_anything_milvus_collection"
    milvus_adapter = MilvusVectorAdapter(uri='./milvus.db', collection_name=index_name)

    # Delete existing index if it exists
    try:
        milvus_adapter.delete_index(index_name)
    except:
        pass

    # Create a new index
    milvus_adapter.create_index()

    # Initialize the embedding model
    model = EmbeddingModel.from_pretrained_hf(
        WhichModel.Bert, 
        model_id="sentence-transformers/all-MiniLM-L12-v2"
    )

    # Embed a PDF file
    data = embed_anything.embed_file(
        "path/to/your/file.pdf",
        embedder=model,
        adapter=milvus_adapter,
    )