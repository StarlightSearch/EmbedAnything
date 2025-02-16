import embed_anything
import os
from typing import Dict, List
from embed_anything import EmbedData
from embed_anything.vectordb import Adapter
from uuid import uuid4
import lancedb

class LanceAdapter(Adapter):
    def __init__(self, db_path: str, embedding_dimension: int):

        import pyarrow as pa  # For schema definition

        self.db_path = db_path
        self.connection = lancedb.connect(self.db_path)
        self.dimension = embedding_dimension
        
        # Define schema using pyarrow
        self.schema = pa.schema([
            pa.field("embeddings", pa.list_(pa.float32(), self.dimension)),
            pa.field("text", pa.string()),
            pa.field("file_name", pa.string()),
            pa.field("modified", pa.string()),
            pa.field("created", pa.string())
        ])

    def create_index(self, table_name: str):
        self.table_name = table_name
        self.connection = lancedb.connect(self.db_path)
        self.table = self.connection.create_table(table_name, schema=self.schema)


    def convert(self, embeddings: List[List[EmbedData]]) -> List[Dict]:
        data = []
        for embedding in embeddings:

            data.append(
                {
                    "text": embedding.text,
                    "embeddings": embedding.embedding,
                    "file_name": embedding.metadata["file_name"],
                    "modified": embedding.metadata["modified"],
                    "created": embedding.metadata["created"],
                }
            )
        return data
    
    def delete_index(self, table_name: str):
        self.connection.drop_table(table_name)

    def upsert(self, data: EmbedData):
        self.table.add(self.convert(data))


def main():
    # Initialize adapter
    lance_adapter = LanceAdapter(db_path="tmp/lancedb", embedding_dimension=384)
    
    # Initialize model
    model = embed_anything.EmbeddingModel.from_pretrained_hf(
        embed_anything.WhichModel.Bert, 
        model_id="sentence-transformers/all-MiniLM-L12-v2"
    )
    
    # Create index and embed data
    if "docs" in lance_adapter.connection.table_names():
        lance_adapter.delete_index("docs")
    lance_adapter.create_index("docs")
    
    data = embed_anything.embed_file(
        "test_files/attention.pdf",
        embedder=model,
        adapter=lance_adapter,
    )
    
    # Example search
    query_vec = embed_anything.embed_query(['attention'], embedder=model)[0].embedding
    docs = lance_adapter.table.search(query_vec).limit(5).to_pandas()["text"]
    print(docs[2])

if __name__ == "__main__":
    main()
