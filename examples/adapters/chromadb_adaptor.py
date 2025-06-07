import embed_anything
import os
from typing import Dict, List, Optional
from embed_anything import EmbedData
from embed_anything.vectordb import Adapter
from uuid import uuid4
import chromadb

class ChromaAdapter(Adapter):
    def __init__(self, db_path: str, embedding_dimension: int):
        self.db_path = db_path
        self.dimension = embedding_dimension
        
        os.makedirs(db_path, exist_ok=True)
        # Initialize ChromaDB with persistence path
        self.client = chromadb.PersistentClient(path=self.db_path)
        
    def create_index(self, table_name: str):
        # Get or create collection with provided name
        self.collection = self.client.get_or_create_collection(
            name=table_name
        )
    
    def convert(self, embeddings: List[EmbedData]) -> Dict[str, List]:
        # Format data for ChromaDB's expected structure
        ids = []
        documents = []
        embeddings_list = []
        metadatas = []
        
        for embedding in embeddings:
            id = str(uuid4())
            ids.append(id)
            documents.append(embedding.text)
            embeddings_list.append(embedding.embedding)
            
            metadata = {
                "file_name": embedding.metadata["file_name"],
                "modified": embedding.metadata["modified"],
                "created": embedding.metadata["created"]
            }
            metadatas.append(metadata)
        
        return {
            "ids": ids,
            "documents": documents,
            "embeddings": embeddings_list,
            "metadatas": metadatas
        }
    
    def delete_index(self, table_name: str):
        # Remove collection if it exists
        self.client.delete_collection(name=table_name)
        
    
    def upsert(self, data: List[EmbedData]):
        # Add documents and embeddings to collection
        converted_data = self.convert(data)
        
        self.collection.add(
            ids=converted_data["ids"],
            documents=converted_data["documents"],
            embeddings=converted_data["embeddings"],
            metadatas=converted_data["metadatas"]
        )


def main():
    # Initialize adapter and model
    chroma_adapter = ChromaAdapter(db_path="tmp/chromadb", embedding_dimension=384)
    
    model = embed_anything.EmbeddingModel.from_pretrained_hf(
        embed_anything.WhichModel.Bert, 
        model_id="sentence-transformers/all-MiniLM-L12-v2"
    )
    
    # Create collection and embed documents
    chroma_adapter.create_index("docs")
    
    data = embed_anything.embed_file(
        "MoE.pdf",
        embedder=model,
        adapter=chroma_adapter,
    )
    
    # Example search
    query_embedding = embed_anything.embed_query(['what is mistral'], embedder=model)[0].embedding
    
    results = chroma_adapter.collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    for doc in results["documents"][0]:
        print(doc)

if __name__ == "__main__":
    main()