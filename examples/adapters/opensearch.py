"""
EmbedAnything OpenSearch Adapter
Provides a clean interface for semantic search using EmbedAnything and OpenSearch
"""

import urllib3
from typing import List, Dict, Any
from opensearchpy import OpenSearch, helpers
from embed_anything import EmbedData, EmbeddingModel, TextEmbedConfig, WhichModel
import numpy as np

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class OpenSearchAdapter:
    """
    Adapter class to integrate EmbedAnything with OpenSearch
    """
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 9200,
                 index_name: str = "anything",
                 http_auth=None,
                 use_ssl: bool = False):
        """
        Initialize the adapter
        
        Args:
            host: OpenSearch host
            port: OpenSearch port
            index_name: Name of the OpenSearch index
            http_auth: Optional tuple of (username, password)
            use_ssl: Whether to use SSL connection
        """
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=http_auth,
            use_ssl=use_ssl,
            verify_certs=False,
            ssl_show_warn=False
        )
        self.index_name = index_name
    
    def create_index(self, 
                     dimension: int, 
                     metric: str = "l2", 
                     mappings: Dict = {}, 
                     settings: Dict = {}, 
                     **kwargs):
        """
        Create OpenSearch index with vector field mapping
        
        Args:
            dimension: Dimension of the embeddings
            metric: Distance metric (l2, cosinesimil, innerproduct)
            mappings: Custom mappings to override defaults
            settings: Custom settings to override defaults
            **kwargs: Additional arguments including index_name override
        """
        if "index_name" in kwargs:
            self.index_name = kwargs["index_name"]
        
        # Default mappings
        default_mappings = {
            "properties": {
                "text": {"type": "text"},
                "embeddings": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": metric,
                        "engine": "lucene",
                        "parameters": {
                            "m": 16,
                            "ef_construction": 200
                        }
                    }
                },
                "metadata": {"type": "object"}
            }
        }
        
        # Default settings
        default_settings = {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "knn": True
            }
        }
        
        # Use provided mappings/settings or defaults
        final_mappings = mappings if mappings else default_mappings
        final_settings = settings if settings else default_settings
        
        self.client.indices.create(
            index=self.index_name,
            body={
                "mappings": final_mappings,
                "settings": final_settings
            }
        )
    
    def convert(self, embeddings: List[Dict]) -> List[Dict]:
        """
        Convert EmbedData objects to OpenSearch document format
        
        Args:
            embeddings: List of EmbedData objects
            
        Returns:
            List of dictionaries formatted for OpenSearch
        """
        data = []
        for embedding in embeddings:
            data.append({
                "text": embedding["text"],
                "embeddings": embedding["embedding"],
                "metadata": embedding["metadata"]
            })
        return data
    
    def delete_index(self, index_name: str):
        """
        Delete an OpenSearch index
        
        Args:
            index_name: Name of the index to delete
        """
        self.client.indices.delete(index=index_name)
    
    def gendata(self, data: List[Dict]):
        """
        Generator function for bulk operations
        
        Args:
            data: List of documents to index
            
        Yields:
            Individual documents for bulk indexing
        """
        for doc in data:
            yield {
                "_index": self.index_name,
                "_source": doc
            }
    
    def upsert(self, data: List[Dict]):
        """
        Upsert documents into OpenSearch
        
        Args:
            data: List of EmbedData objects to upsert
        """
        converted_data = self.convert(data)
        helpers.bulk(
            client=self.client, 
            actions=self.gendata(converted_data)
        )


# Initialize model and config
model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Jina, model_id="jinaai/jina-embeddings-v2-small-en"
)
config = TextEmbedConfig(
    chunk_size=1000, batch_size=32, splitting_strategy="sentence"
)

# Initialize OpenSearch client
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9201}],
    http_auth=None,
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False
)

index_name = 'my-vectors'

def semantic_search(query, k=2):
    # Generate query embedding
    query_embedding = model.embed_query(
        [query], config=config  # Pass query as a list
    )
    
    # Prepare search body
    search_body = {
        "size": k,
        "query": {
            "knn": {
                "embeddings": {
                    "vector": query_embedding[0].embedding,  # Embedding is already a list
                    "k": k
                }
            }
        }
    }
    
    # Execute search
    response = client.search(
        body=search_body,
        index=index_name
    )
    
    # Process and return results
    results = []
    for hit in response['hits']['hits']:
        results.append({
            'text': hit['_source']['text'],
            'score': hit['_score']
        })
    
    return results

# Initialize adapter
adapter = OpenSearchAdapter(
    host='localhost',
    port=9201,  # Updated port to match docker-compose
    index_name='my-vectors'
)

# Create sample data
sample_texts = [
    "The quick brown fox jumps over the lazy dog",
    "A fast orange fox leaps over a sleepy canine",
    "Machine learning is transforming the world",
    "Artificial intelligence is revolutionizing technology"
]

# Create sample data objects
embed_data_list = [
    {
        "text": text,
        "embedding": model.embed_query([text], config=config)[0].embedding,  # Pass text as a list
        "metadata": {
            "file_name": "sample.txt",
            "modified": "2024-03-19",
            "created": "2024-03-19"
        }
    }
    for text in sample_texts
]

# Create index for 512-dimensional vectors (Jina model dimension)
if not client.indices.exists(index_name):
    adapter.create_index(dimension=512, metric='l2')
else:
    print(f"Index '{index_name}' already exists. Skipping creation.")

# Upsert data
try:
    adapter.upsert(embed_data_list)
    print("Successfully indexed sample data!")
except Exception as e:
    print(f"Error during upsert: {e}")

queries = [
    "Tell me about foxes"
]

# Test single query
try:
    print(f"\nRunning semantic search for query: '{queries[0]}'")
    results = semantic_search(queries[0])
    print("Results:")
    if results:
        for i, res in enumerate(results, 1):
            print(f"{i}. Text: {res['text']} (Score: {res['score']})")
    else:
        print("No results found.")
except Exception as e:
    print(f"Error during search: {e}")

# Note: Uncomment the following line if you want to delete the index
# adapter.delete_index('my-vectors')