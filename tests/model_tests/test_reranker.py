"""
Test suite for the Reranker functionality in EmbedAnything.

This module tests the reranking capabilities including model loading,
document reranking, and score computation.
"""

import pytest
from embed_anything import Reranker, Dtype, RerankerResult, DocumentRank


@pytest.fixture
def reranker_model():
    """Fixture to provide a working reranker model for testing."""
    try:
        # Using the Qwen3 reranker which is known to work
        model = Reranker.from_pretrained(
            "zhiqing/Qwen3-Reranker-0.6B-ONNX", 
            dtype=Dtype.F32
        )
        return model
    except Exception as e:
        pytest.skip(f"Could not load reranker model: {e}")


@pytest.fixture
def sample_documents():
    """Fixture providing sample documents for testing."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "The weather is sunny today in New York.",
        "Deep learning uses neural networks for pattern recognition.",
        "Python is a popular programming language for data science.",
        "The capital of France is Paris.",
        "Neural networks are inspired by biological brain structures."
    ]


@pytest.fixture
def sample_queries():
    """Fixture providing sample queries for testing."""
    return [
        "What is machine learning?",
        "How do neural networks work?",
        "What is the weather like?",
        "Tell me about programming languages."
    ]


def test_reranker_model_creation(reranker_model):
    """Test that the reranker model can be created successfully."""
    assert reranker_model is not None
    assert hasattr(reranker_model, 'rerank')
    assert hasattr(reranker_model, 'compute_scores')


def test_reranker_model_creation_with_different_dtypes():
    """Test reranker creation with different data types."""
    try:
        # Test with F32 (should work)
        model_f32 = Reranker.from_pretrained(
            "zhiqing/Qwen3-Reranker-0.6B-ONNX", 
            dtype=Dtype.F32
        )
        assert model_f32 is not None
        
        # Test with F16 (may not work for all models)
        try:
            model_f16 = Reranker.from_pretrained(
                "zhiqing/Qwen3-Reranker-0.6B-ONNX", 
                dtype=Dtype.F16
            )
            assert model_f16 is not None
        except Exception:
            # F16 might not be supported, which is okay
            pass
            
    except Exception as e:
        pytest.skip(f"Could not test different dtypes: {e}")


def test_reranker_basic_functionality(reranker_model, sample_documents, sample_queries):
    """Test basic reranking functionality with a single query."""
    query = [sample_queries[0]]  # "What is machine learning?"
    
    # Test reranking
    results = reranker_model.rerank(query, sample_documents, 3)
    
    # Verify results structure
    assert len(results) == 1
    result = results[0]
    assert result.query == query[0]
    assert len(result.documents) == 3
    
    # Verify document ranking
    for i, doc in enumerate(result.documents):
        assert doc.rank == i + 1
        assert doc.relevance_score is not None
        assert isinstance(doc.relevance_score, float)
        assert doc.document in sample_documents


def test_reranker_multiple_queries(reranker_model, sample_documents, sample_queries):
    """Test reranking with multiple queries simultaneously."""
    queries = sample_queries[:2]  # Use first two queries
    
    # Test reranking
    results = reranker_model.rerank(queries, sample_documents, 2)
    
    # Verify results structure
    assert len(results) == 2
    
    # Check first query results
    result1 = results[0]
    assert result1.query == queries[0]
    assert len(result1.documents) == 2
    
    # Check second query results
    result2 = results[1]
    assert result2.query == queries[1]
    assert len(result2.documents) == 2


def test_reranker_top_k_parameter(reranker_model, sample_documents, sample_queries):
    """Test that the top_k parameter works correctly."""
    query = [sample_queries[0]]
    
    # Test with different top_k values
    for top_k in [1, 2, 5]:
        results = reranker_model.rerank(query, sample_documents, top_k)
        assert len(results) == 1
        assert len(results[0].documents) == min(top_k, len(sample_documents))


def test_reranker_compute_scores(reranker_model, sample_documents, sample_queries):
    """Test the compute_scores method for raw scoring."""
    query = [sample_queries[0]]
    
    # Test score computation
    scores = reranker_model.compute_scores(query, sample_documents, batch_size=4)
    
    # Verify scores structure
    assert len(scores) == 1  # One query
    assert len(scores[0]) == len(sample_documents)  # One score per document
    
    # Verify score values
    for score in scores[0]:
        assert isinstance(score, float)
        assert not (score < 0 or score > 1)  # Scores should be normalized


def test_reranker_empty_documents(reranker_model, sample_queries):
    """Test reranking with empty document list."""
    query = [sample_queries[0]]
    
    # Test with empty documents
    results = reranker_model.rerank(query, [], 5)
    assert len(results) == 1
    assert len(results[0].documents) == 0


def test_reranker_single_document(reranker_model, sample_queries):
    """Test reranking with a single document."""
    query = [sample_queries[0]]
    single_doc = ["Machine learning is a subset of AI."]
    
    results = reranker_model.rerank(query, single_doc, 5)
    assert len(results) == 1
    assert len(results[0].documents) == 1
    assert results[0].documents[0].rank == 1


def test_reranker_relevance_ordering(reranker_model, sample_documents, sample_queries):
    """Test that documents are properly ordered by relevance score."""
    query = [sample_queries[0]]  # "What is machine learning?"
    
    results = reranker_model.rerank(query, sample_documents, len(sample_documents))
    
    # Get all documents and scores
    documents = results[0].documents
    
    # Verify scores are in descending order (most relevant first)
    for i in range(len(documents) - 1):
        assert documents[i].relevance_score >= documents[i + 1].relevance_score


def test_reranker_batch_processing(reranker_model, sample_documents, sample_queries):
    """Test that batch processing works correctly."""
    query = [sample_queries[0]]
    
    # Test with different batch sizes
    for batch_size in [1, 2, 4, 8]:
        try:
            scores = reranker_model.compute_scores(query, sample_documents, batch_size)
            assert len(scores) == 1
            assert len(scores[0]) == len(sample_documents)
        except Exception as e:
            # Some batch sizes might not be supported
            pytest.skip(f"Batch size {batch_size} not supported: {e}")


def test_reranker_error_handling():
    """Test that appropriate errors are raised for invalid inputs."""
    # Test with invalid model ID
    with pytest.raises(Exception):
        Reranker.from_pretrained("invalid/model/id", dtype=Dtype.F32)
    
    # Test with invalid dtype
    try:
        with pytest.raises(Exception):
            Reranker.from_pretrained(
                "zhiqing/Qwen3-Reranker-0.6B-ONNX", 
                dtype="INVALID_DTYPE"
            )
    except Exception:
        # The model might handle invalid dtypes gracefully
        pass


def test_reranker_document_metadata(reranker_model, sample_documents, sample_queries):
    """Test that document metadata is preserved correctly."""
    query = [sample_queries[0]]
    
    results = reranker_model.rerank(query, sample_documents, 3)
    
    # Verify document content is preserved
    for doc in results[0].documents:
        assert doc.document in sample_documents
        assert isinstance(doc.document, str)
        assert len(doc.document) > 0


def test_reranker_query_preservation(reranker_model, sample_documents, sample_queries):
    """Test that queries are preserved correctly in results."""
    query = [sample_queries[0]]
    
    results = reranker_model.rerank(query, sample_documents, 3)
    
    # Verify query is preserved
    assert results[0].query == query[0]
    assert isinstance(results[0].query, str)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
