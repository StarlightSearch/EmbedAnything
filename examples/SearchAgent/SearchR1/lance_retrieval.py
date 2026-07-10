from typing import List, Dict, Optional
import os
import numpy as np
import warnings

from embed_anything import (
    EmbeddingModel,
    TextEmbedConfig,
    WhichModel,
    embed_query,
    ONNXModel,
    Dtype,
)
import lancedb
import embed_anything
from tqdm import tqdm


class LanceRetriever:
    """LanceDB-based retriever using embed_anything for embeddings."""
    
    def __init__(self, config):
        """
        Initialize the LanceDB retriever.
        
        Args:
            config: Configuration object with the following attributes:
                - lancedb_path: Path to LanceDB database (default: "tmp/lancedb")
                - table_name: Name of the table in LanceDB (default: "docs")
                - retrieval_topk: Default number of results to return (default: 10)
                - corpus_path: Path to corpus file (optional, for loading document metadata)
        """
        self.config = config
        self.lancedb_path = getattr(config, 'lancedb_path', 'tmp/lancedb')
        self.table_name = getattr(config, 'table_name', 'docs')
        self.topk = getattr(config, 'retrieval_topk', 10)
        self.corpus_path = getattr(config, 'corpus_path', None)
        
        # Initialize embedding model
        self.model = EmbeddingModel.from_pretrained_onnx(
            WhichModel.Bert, 
            ONNXModel.ModernBERTBase, 
            dtype=Dtype.Q4F16
        )
        
        # Connect to LanceDB
        self.connection = lancedb.connect(self.lancedb_path)
        
        # Load table
        if self.table_name not in self.connection.table_names():
            raise ValueError(f"Table '{self.table_name}' not found in LanceDB. Please create it first using embed_store().")
        
        self.table = self.connection.open_table(self.table_name)
        
        # Optionally load corpus for metadata
        self.corpus = None
        if self.corpus_path and os.path.exists(self.corpus_path):
            try:
                import datasets
                self.corpus = datasets.load_dataset(
                    'json',
                    data_files=self.corpus_path,
                    split="train",
                    num_proc=4
                )
            except Exception as e:
                warnings.warn(f"Could not load corpus from {self.corpus_path}: {e}")
    
    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query into an embedding vector.
        
        Args:
            query: Query string to encode
            
        Returns:
            numpy array of the embedding vector
        """
        embedding_data = embed_query(
            [query], 
            embedder=self.model, 
            config=TextEmbedConfig(batch_size=256)
        )[0]
        return np.array(embedding_data.embedding, dtype=np.float32)
    
    def _encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        Encode multiple queries into embedding vectors.
        
        Args:
            queries: List of query strings to encode
            
        Returns:
            numpy array of shape (len(queries), embedding_dim)
        """
        embedding_data_list = embed_query(
            queries,
            embedder=self.model,
            config=TextEmbedConfig(batch_size=256)
        )
        embeddings = [np.array(ed.embedding, dtype=np.float32) for ed in embedding_data_list]
        return np.array(embeddings)
    
    def _search(self, query: str, num: Optional[int] = None, return_score: bool = False) -> List[Dict]:
        """
        Search for similar documents in LanceDB.
        
        Args:
            query: Query string
            num: Number of results to return (default: self.topk)
            return_score: Whether to return similarity scores
            
        Returns:
            List of document dictionaries, optionally with scores
        """
        if num is None:
            num = self.topk
        
        # Encode query
        query_vector = self._encode_query(query)
        
        # Search in LanceDB
        results = self.table.search(query_vector).limit(num).to_pandas()
        
        if len(results) == 0:
            if return_score:
                return [], []
            else:
                return []
        
        # Convert results to list of dictionaries
        doc_results = []
        scores = []
        
        for _, row in results.iterrows():
            doc = {
                'text': row.get('text', ''),
                'id': row.get('id', ''),
            }
            
            # If corpus is loaded, try to get full document metadata
            if self.corpus is not None and 'id' in row:
                try:
                    doc_idx = int(row['id'])
                    if doc_idx < len(self.corpus):
                        full_doc = self.corpus[doc_idx]
                        # Merge with corpus data
                        doc.update(full_doc)
                except (ValueError, IndexError):
                    pass
            
            doc_results.append(doc)
            
            # Extract score if available (LanceDB returns _distance, convert to similarity)
            if '_distance' in row:
                # Convert distance to similarity score (lower distance = higher similarity)
                # Using negative distance as similarity score
                score = float(-row['_distance'])
                scores.append(score)
            elif return_score:
                scores.append(0.0)  # Default score if not available
        
        if return_score:
            return doc_results, scores
        else:
            return doc_results
    
    def _batch_search(self, query_list: List[str], num: Optional[int] = None, return_score: bool = False):
        """
        Batch search for similar documents in LanceDB.
        
        Args:
            query_list: List of query strings
            num: Number of results per query (default: self.topk)
            return_score: Whether to return similarity scores
            
        Returns:
            List of lists of document dictionaries, optionally with scores
        """
        if isinstance(query_list, str):
            query_list = [query_list]
        
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        
        # Process queries in batch
        for query in tqdm(query_list, desc='LanceDB retrieval'):
            query_results, query_scores = self._search(query, num, return_score=True)
            results.append(query_results)
            if return_score:
                scores.append(query_scores)
        
        if return_score:
            return results, scores
        else:
            return results
    
    def search(self, query: str, num: Optional[int] = None, return_score: bool = False):
        """Public interface for single search."""
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: Optional[int] = None, return_score: bool = False):
        """Public interface for batch search."""
        return self._batch_search(query_list, num, return_score)


def embed_store(ds, lancedb_path: str = "tmp/lancedb", table_name: str = "docs"):
    """
    Store embeddings in LanceDB.
    
    Args:
        ds: Dataset with 'train' split containing items with 'Response' field
        lancedb_path: Path to LanceDB database
        table_name: Name of the table to create
    """
    # Initialize embedding model
    model = EmbeddingModel.from_pretrained_onnx(
        WhichModel.Bert, 
        ONNXModel.ModernBERTBase, 
        dtype=Dtype.Q4F16
    )
    
    # Connect to LanceDB
    connection = lancedb.connect(lancedb_path)
    
    # Drop existing table if it exists
    if table_name in connection.table_names():
        connection.drop_table(table_name)
    
    data = []
    
    # Process dataset and create embeddings
    for (i, item) in enumerate(tqdm(ds['train'], desc='Creating embeddings')):
        # Get the embeddings
        embedding_data = embed_query(
            [str(item['Response'])], 
            embedder=model, 
            config=TextEmbedConfig(batch_size=256)
        )[0]
        
        # Create a document dictionary
        doc = {
            "vector": embedding_data.embedding,
            "text": embedding_data.text,
            "id": str(i)
        }
        data.append(doc)
    
    # Create table with all data
    table = connection.create_table(table_name, data=data)
    
    print(f"Created table '{table_name}' with {len(data)} documents in {lancedb_path}")
    return table

