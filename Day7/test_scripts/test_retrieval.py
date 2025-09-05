"""
Test the retrieval module functionality.
"""

import sys
import os
import numpy as np

# Add the parent directory to the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.retrieval import DocumentRetriever, SearchResult
from src.data_loader import Document

def test_search_result_class():
    """Test the SearchResult data class."""
    print("\n📊 Testing SearchResult class...")
    
    # Create a test document first
    test_doc = Document(
        id="test_1",
        title="Test Article",
        content="This is a preview of the test article content...",
        url="https://en.wikipedia.org/wiki/Test"
    )
    
    # Create a test search result
    result = SearchResult(
        document=test_doc,
        similarity_score=0.8542,
        rank=1
    )
    
    print(f"✓ SearchResult created: {result.document.title}")
    print(f"  Similarity: {result.similarity_score}")
    print(f"  Rank: {result.rank}")
    print(f"  URL: {result.document.url}")
    
    # Test string representation
    result_str = str(result)
    print(f"✓ String representation: {result_str}")
    
    return True

def test_document_retriever_initialization():
    """Test DocumentRetriever initialization."""
    print("\n🔍 Testing DocumentRetriever initialization...")
    
    # Create test documents
    documents = [
        Document(
            id="ml_1",
            title="Machine Learning",
            content="Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence.",
            url="https://en.wikipedia.org/wiki/Machine_learning"
        ),
        Document(
            id="dl_1",
            title="Deep Learning",
            content="Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            url="https://en.wikipedia.org/wiki/Deep_learning"
        ),
        Document(
            id="cv_1",
            title="Computer Vision",
            content="Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.",
            url="https://en.wikipedia.org/wiki/Computer_vision"
        )
    ]
    
    # Create corresponding embeddings (mock data for testing)
    embeddings = np.array([
        [1.0, 0.0, 0.0],    # ML embedding
        [0.8, 0.6, 0.0],    # DL embedding (similar to ML)
        [0.0, 0.0, 1.0]     # CV embedding (different)
    ])
    
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    print(f"Test documents: {len(documents)}")
    print(f"Test embeddings shape: {embeddings.shape}")
    
    # Initialize retriever
    retriever = DocumentRetriever(documents, embeddings)
    
    print("✓ DocumentRetriever initialized")
    print(f"✓ Document count: {len(retriever.documents)}")
    print(f"✓ Embeddings shape: {retriever.embeddings.shape}")
    
    # Check that documents and embeddings match
    assert len(retriever.documents) == retriever.embeddings.shape[0], "Document/embedding count mismatch"
    
    return True

def test_search_functionality():
    """Test search functionality."""
    print("\n🔎 Testing search functionality...")
    
    # Create test documents with different content
    documents = [
        Document(
            id="ml_test_1",
            title="Machine Learning Basics",
            content="Machine learning algorithms learn patterns from data to make predictions. Common algorithms include linear regression, decision trees, and neural networks.",
            url="https://example.com/ml"
        ),
        Document(
            id="dnn_test_1",
            title="Deep Neural Networks",
            content="Deep neural networks consist of multiple layers of interconnected nodes. They are particularly effective for complex pattern recognition tasks.",
            url="https://example.com/dnn"
        ),
        Document(
            id="prep_test_1",
            title="Data Preprocessing",
            content="Data preprocessing involves cleaning, transforming, and preparing raw data for analysis. It includes handling missing values and feature scaling.",
            url="https://example.com/preprocessing"
        ),
        Document(
            id="graphics_test_1",
            title="Computer Graphics",
            content="Computer graphics involves creating and manipulating visual content using computers. It includes 3D modeling, rendering, and animation techniques.",
            url="https://example.com/graphics"
        )
    ]
    
    # Create mock embeddings that reflect content similarity
    embeddings = np.array([
        [1.0, 0.8, 0.2],    # ML - similar to neural networks
        [0.9, 1.0, 0.1],    # Neural networks - similar to ML
        [0.5, 0.3, 1.0],    # Data preprocessing - somewhat related to ML
        [0.1, 0.1, 0.2]     # Graphics - different from others
    ])
    
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    retriever = DocumentRetriever(documents, embeddings)
    
    # Test query embedding (similar to ML/Neural networks)
    query_embedding = np.array([0.95, 0.9, 0.1])
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    print(f"Query embedding: {query_embedding}")
    
    # Test search with different parameters
    test_cases = [
        {"top_k": 2, "min_similarity": 0.0, "expected_count": 2},
        {"top_k": 5, "min_similarity": 0.0, "expected_count": 4},  # All documents
        {"top_k": 3, "min_similarity": 0.9, "expected_count": 2},  # High similarity threshold - only top 2 should be >0.9
        {"top_k": 1, "min_similarity": 0.0, "expected_count": 1},  # Only top result
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n  Test case {i}: top_k={case['top_k']}, min_similarity={case['min_similarity']}")
        
        results = retriever.retrieve_top_k(
            query_embedding=query_embedding,
            top_k=case["top_k"],
            min_similarity=case["min_similarity"]
        )
        
        print(f"    Results: {len(results)} (expected: {case['expected_count']})")
        
        # Check result count
        assert len(results) == case["expected_count"], f"Wrong result count: {len(results)} vs {case['expected_count']}"
        
        # Check result ordering (should be by similarity score, descending)
        for j in range(len(results) - 1):
            assert results[j].similarity_score >= results[j + 1].similarity_score, "Results not ordered by similarity"
        
        # Check similarity threshold
        for result in results:
            assert result.similarity_score >= case["min_similarity"], f"Result below similarity threshold: {result.similarity_score}"
        
        # Display results
        for j, result in enumerate(results):
            print(f"      {j + 1}. {result.document.title} (score: {result.similarity_score:.3f})")
        
        print(f"    ✓ Test case {i} passed")
    
    return True

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🛡️ Testing edge cases...")
    
    # Create minimal test setup
    documents = [
        Document(
            id="edge_1", 
            title="Test Doc", 
            content="Test content", 
            url="https://example.com/test"
        )
    ]
    embeddings = np.array([[1.0, 0.0, 0.0]])
    
    retriever = DocumentRetriever(documents, embeddings)
    query_embedding = np.array([1.0, 0.0, 0.0])
    
    # Test top_k larger than document count
    print("Testing top_k > document count...")
    results = retriever.retrieve_top_k(query_embedding, top_k=10, min_similarity=0.0)
    assert len(results) == 1, "Should return all available documents"
    print("✓ Large top_k handled correctly")
    
    # Test very high similarity threshold
    print("Testing high similarity threshold...")
    results = retriever.retrieve_top_k(query_embedding, top_k=5, min_similarity=0.99)
    assert len(results) <= 1, "High threshold should filter most results"
    print(f"✓ High threshold returned {len(results)} results")
    
    # Test zero similarity threshold
    print("Testing zero similarity threshold...")
    results = retriever.retrieve_top_k(query_embedding, top_k=5, min_similarity=0.0)
    assert len(results) == 1, "Should return all documents with zero threshold"
    print("✓ Zero threshold handled correctly")
    
    # Test top_k = 0
    print("Testing top_k = 0...")
    results = retriever.retrieve_top_k(query_embedding, top_k=0, min_similarity=0.0)
    assert len(results) == 0, "Should return no results for top_k=0"
    print("✓ Zero top_k handled correctly")
    
    # Test exact similarity match
    print("Testing exact similarity match...")
    results = retriever.retrieve_top_k(query_embedding, top_k=1, min_similarity=0.0)
    assert len(results) == 1, "Should find exact match"
    assert abs(results[0].similarity_score - 1.0) < 1e-6, f"Should have perfect similarity: {results[0].similarity_score}"
    print(f"✓ Exact match similarity: {results[0].similarity_score:.6f}")
    
    return True

def test_content_preview():
    """Test content preview generation."""
    print("\n📝 Testing content preview...")
    
    # Create document with long content
    long_content = "This is a very long document content. " * 50  # ~1900 characters
    document = Document(
        id="long_1",
        title="Long Document",
        content=long_content,
        url="https://example.com/long"
    )
    
    embeddings = np.array([[1.0, 0.0, 0.0]])
    retriever = DocumentRetriever([document], embeddings)
    
    query_embedding = np.array([1.0, 0.0, 0.0])
    results = retriever.retrieve_top_k(query_embedding, top_k=1, min_similarity=0.0)
    
    assert len(results) == 1, "Should return one result"
    
    # The SearchResult contains the full document, not a preview
    result_content = results[0].document.content
    print(f"Original content length: {len(long_content)}")
    print(f"Result content length: {len(result_content)}")
    print(f"Result content start: {result_content[:100]}...")
    
    # Check that we get the full document content
    assert result_content == long_content, "Should return full document content"
    
    # The document title should be preserved
    assert results[0].document.title == "Long Document", "Title should be preserved"
    
    print("✓ Document content and metadata preserved correctly")
    
    return True

def main():
    """Run all retrieval tests."""
    print("🧪 Testing Retrieval Module")
    print("=" * 50)
    
    tests = [
        test_search_result_class,
        test_document_retriever_initialization,
        test_search_functionality,
        test_edge_cases,
        test_content_preview
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} passed")
            else:
                failed += 1
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} failed with error: {e}")
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All retrieval tests passed!")
        return True
    else:
        print("❌ Some retrieval tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)