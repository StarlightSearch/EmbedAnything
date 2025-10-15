"""
Qwen3 Reranker Examples

This file demonstrates how to use the Qwen3 reranking model in EmbedAnything.
The Qwen3 reranker is specifically designed for document relevance scoring and ranking.

Key Features:
- High-quality relevance scoring
- Support for multiple queries and documents
- Batch processing capabilities
- ONNX optimization for performance
"""

from embed_anything import Reranker, Dtype, RerankerResult, DocumentRank
import time

def format_query(query: str, instruction=None):
    """You may add instruction to get better results in specific fields."""
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    return f"<Instruct>: {instruction}\n<Query>: {query}\n"

def format_document(doc: str):
    return f"<Document>: {doc}"

def basic_qwen3_reranking():
    """Basic example of using Qwen3 reranker for simple document ranking."""
    print("=== Basic Qwen3 Reranking ===")

    # Initialize the Qwen3 reranker
    # Using the ONNX-optimized version for better performance
    reranker = Reranker.from_pretrained(
        "zhiqing/Qwen3-Reranker-0.6B-ONNX", 
        dtype=Dtype.F32
    )

    # Define a query and candidate documents
    query = ["What is artificial intelligence?"]
    documents = [
        "Artificial intelligence is a field of computer science.",
        "Machine learning is a subset of AI.",
        "The weather is sunny today.",
        "AI systems can learn from data.",
        "Deep learning uses neural networks.",
        "Pizza is a popular Italian food."
    ]

    # Format query and documents
    query = [format_query(x) for x in query]
    documents = [format_document(x) for x in documents]

    # Rerank documents and get top 3 results
    results = reranker.rerank(query, documents, 2)

    # Display results
    for result in results:
        print(f"Query: {result.query}")
        print("Top ranked documents:")
        for doc in result.documents:
            print(f"  Rank {doc.rank}: {doc.document}")
            print(f"    Relevance Score: {doc.relevance_score:.4f}")
        print()

def multi_query_reranking():
    """Example of reranking documents for multiple queries simultaneously."""
    print("=== Multi-Query Reranking ===")

    reranker = Reranker.from_pretrained(
        "zhiqing/Qwen3-Reranker-0.6B-ONNX", 
        dtype=Dtype.F32
    )

    # Multiple queries
    queries = [
        "How to make coffee?",
        "What is machine learning?",
        "Tell me about cats"
    ]

    # Shared document collection
    documents = [
        "Coffee is made by brewing ground coffee beans with hot water.",
        "Machine learning enables computers to learn without explicit programming.",
        "Cats are domesticated mammals and popular pets.",
        "You can make coffee using a French press or drip machine.",
        "Neural networks are a key component of modern AI.",
        "Cats have been living with humans for thousands of years.",
        "The weather is nice today.",
        "Deep learning is a subset of machine learning.",
        "Coffee beans come from the Coffea plant.",
        "Cats are known for their independent nature."
    ]

    # Format queries and documents
    queries = [format_query(x) for x in queries]
    documents = [format_document(x) for x in documents]

    # Rerank for all queries at once
    results = reranker.rerank(queries, documents, top_k=3)

    # Display results for each query
    for result in results:
        print(f"Query: {result.query}")
        print("Top documents:")
        for doc in result.documents:
            print(f"  {doc.document} (Score: {doc.relevance_score:.4f})")
        print()

def custom_scoring_example():
    """Example of using compute_scores for custom ranking logic."""
    print("=== Custom Scoring with compute_scores ===")

    reranker = Reranker.from_pretrained(
        "zhiqing/Qwen3-Reranker-0.6B-ONNX", 
        dtype=Dtype.F32
    )

    query = ["What are the benefits of exercise?"]
    documents = [
        "Exercise improves cardiovascular health.",
        "Regular physical activity boosts mood.",
        "The movie was entertaining.",
        "Exercise helps with weight management.",
        "Cooking is a useful skill.",
        "Physical activity increases energy levels.",
        "Exercise promotes better sleep quality."
    ]

    # Format query and documents
    query = [format_query(x) for x in query]
    documents = [format_document(x) for x in documents]

    # Get raw scores for custom processing
    scores = reranker.compute_scores(query, documents, batch_size=4)

    print(f"Raw relevance scores for query: '{query[0]}'")
    print("-" * 50)

    # Create custom ranking based on scores
    doc_scores = list(zip(documents, scores[0]))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    for i, (doc, score) in enumerate(doc_scores):
        print(f"{i+1:2d}. Score: {score:6.4f} | {doc}")

    print()

def performance_benchmark():
    """Benchmark the performance of Qwen3 reranking."""
    print("=== Performance Benchmark ===")

    reranker = Reranker.from_pretrained(
        "zhiqing/Qwen3-Reranker-0.6B-ONNX", 
        dtype=Dtype.F32
    )

    # Create a larger dataset for benchmarking
    queries = ["What is technology?"] * 5  # 5 identical queries
    documents = [
        "Technology refers to the application of scientific knowledge.",
        "Computers are examples of modern technology.",
        "The internet has revolutionized communication.",
        "Smartphones combine multiple technologies.",
        "Artificial intelligence is advancing rapidly.",
        "Renewable energy technologies are important.",
        "Medical technology saves lives.",
        "Transportation technology has evolved significantly.",
        "Educational technology enhances learning.",
        "Space technology enables exploration."
    ] * 10  # 100 total documents

    # Format queries and documents
    queries = [format_query(x) for x in queries]
    documents = [format_document(x) for x in documents]

    print(f"Benchmarking with {len(queries)} queries and {len(documents)} documents...")

    # Warm up
    _ = reranker.compute_scores(queries[:1], documents[:10], batch_size=4)

    # Benchmark
    start_time = time.time()
    results = reranker.rerank(queries, documents, top_k=5)
    end_time = time.time()

    processing_time = end_time - start_time
    docs_per_second = len(documents) / processing_time

    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Documents processed per second: {docs_per_second:.1f}")
    print(f"Total documents processed: {len(documents)}")
    print()

def search_and_rerank_pipeline():
    """Example of a complete search and rerank pipeline."""
    print("=== Search and Rerank Pipeline ===")

    reranker = Reranker.from_pretrained(
        "zhiqing/Qwen3-Reranker-0.6B-ONNX", 
        dtype=Dtype.F32
    )

    # Simulate a search query
    search_query = ["How to learn Python programming?"]

    # Simulate candidate documents from a vector search
    candidate_docs = [
        "Python is a high-level programming language.",
        "To learn Python, start with basic syntax.",
        "The weather is cloudy today.",
        "Python has excellent libraries for data science.",
        "Machine learning with Python is popular.",
        "Practice coding regularly to improve skills.",
        "Python is great for beginners.",
        "Online tutorials are helpful for learning.",
        "The capital of Japan is Tokyo.",
        "Python supports object-oriented programming.",
        "Start with simple projects when learning.",
        "Python has a large and active community."
    ]

    print(f"Search Query: {search_query[0]}")
    print(f"Found {len(candidate_docs)} candidate documents")
    print("\nReranking documents by relevance...")

    # Format search_query and documents
    search_query = [format_query(x) for x in search_query]
    documents = [format_document(x) for x in documents]

    # Rerank the candidates
    reranked_results = reranker.rerank(search_query, candidate_docs, top_k=5)

    print("\nTop 5 most relevant documents:")
    print("-" * 60)

    for result in reranked_results:
        for doc in result.documents:
            print(f"Rank {doc.rank:2d} (Score: {doc.relevance_score:.4f}):")
            print(f"  {doc.document}")
            print()

if __name__ == "__main__":
    print("Qwen3 Reranker Examples")
    print("=" * 50)
    print()

    try:
        # Run all examples
        basic_qwen3_reranking()
        multi_query_reranking()
        custom_scoring_example()
        performance_benchmark()
        search_and_rerank_pipeline()

        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have the required dependencies installed:")
        print("  pip install embed-anything")
        print("  pip install onnxruntime")
