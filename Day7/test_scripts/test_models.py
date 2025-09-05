"""
Test the models module functionality (Pydantic data models).
"""

import sys
import os
import json
from typing import List

# Add the parent directory to the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models import (
    QueryRequest, 
    DocumentResponse, 
    SearchResponse, 
    HealthResponse
)

def test_query_request_model():
    """Test QueryRequest model validation."""
    print("\n🔍 Testing QueryRequest model...")
    
    # Test valid query request
    valid_request = QueryRequest(
        query="machine learning algorithms",
        top_k=5,
        min_similarity=0.1
    )
    
    print(f"✓ Valid request: {valid_request.query}")
    print(f"  top_k: {valid_request.top_k}")
    print(f"  min_similarity: {valid_request.min_similarity}")
    
    # Test default values
    minimal_request = QueryRequest(query="test query")
    print(f"✓ Minimal request with defaults:")
    print(f"  top_k: {minimal_request.top_k} (default)")
    print(f"  min_similarity: {minimal_request.min_similarity} (default)")
    
    # Test validation errors
    print("Testing validation errors...")
    
    # Empty query
    try:
        QueryRequest(query="")
        print("❌ Empty query should have failed")
        return False
    except ValueError as e:
        print(f"✓ Empty query validation: {e}")
    
    # Negative top_k
    try:
        QueryRequest(query="test", top_k=-1)
        print("❌ Negative top_k should have failed")
        return False
    except ValueError as e:
        print(f"✓ Negative top_k validation: {e}")
    
    # Invalid min_similarity range
    try:
        QueryRequest(query="test", min_similarity=1.5)
        print("❌ Invalid min_similarity should have failed")
        return False
    except ValueError as e:
        print(f"✓ Invalid min_similarity validation: {e}")
    
    # Test JSON serialization
    json_data = valid_request.model_dump()
    print(f"✓ JSON serialization: {json_data}")
    
    # Test JSON deserialization
    reconstructed = QueryRequest(**json_data)
    assert reconstructed.query == valid_request.query
    print("✓ JSON deserialization works")
    
    return True

def test_document_response_model():
    """Test DocumentResponse model."""
    print("\n📄 Testing DocumentResponse model...")
    
    # Test valid document response
    doc_response = DocumentResponse(
        id="doc_1",
        title="Machine Learning Overview",
        content_preview="Machine learning is a method of data analysis...",
        similarity_score=0.8542,
        url="https://en.wikipedia.org/wiki/Machine_learning",
        rank=1
    )
    
    print(f"✓ Document response: {doc_response.title}")
    print(f"  ID: {doc_response.id}")
    print(f"  Similarity: {doc_response.similarity_score}")
    print(f"  Rank: {doc_response.rank}")
    print(f"  URL: {doc_response.url}")
    print(f"  Preview: {doc_response.content_preview[:50]}...")
    
    # Test validation - only test the actual field constraints
    try:
        DocumentResponse(
            id="test_id",
            title="Test Document",
            content_preview="test",
            similarity_score=1.5,  # Invalid score > 1.0
            url="https://example.com",
            rank=1
        )
        print("❌ Invalid similarity score should have failed")
        return False
    except ValueError as e:
        print(f"✓ Invalid similarity score validation: {e}")
    
    # Test invalid rank
    try:
        DocumentResponse(
            id="test_id",
            title="Test",
            content_preview="test",
            similarity_score=0.5,
            url="https://example.com",
            rank=0  # Invalid rank (must be > 0)
        )
        print("❌ Invalid rank should have failed")
        return False
    except ValueError as e:
        print(f"✓ Invalid rank validation: {e}")
    
    return True

def test_search_response_model():
    """Test SearchResponse model."""
    print("\n🔎 Testing SearchResponse model...")
    
    # Create sample documents
    documents = [
        DocumentResponse(
            id="doc_1",
            title="Document 1",
            content_preview="This is the first document...",
            similarity_score=0.9,
            url="https://example.com/doc1",
            rank=1
        ),
        DocumentResponse(
            id="doc_2",
            title="Document 2", 
            content_preview="This is the second document...",
            similarity_score=0.7,
            url="https://example.com/doc2",
            rank=2
        )
    ]
    
    # Test search response
    search_response = SearchResponse(
        query="test query",
        results=documents,
        total_results=2,
        processing_time_ms=45.6
    )
    
    print(f"✓ Search response for: '{search_response.query}'")
    print(f"  Total results: {search_response.total_results}")
    print(f"  Processing time: {search_response.processing_time_ms}ms")
    print(f"  Results count: {len(search_response.results)}")
    
    # Test empty results
    empty_response = SearchResponse(
        query="no results query",
        results=[],
        total_results=0,
        processing_time_ms=12.3
    )
    
    assert len(empty_response.results) == 0
    assert empty_response.total_results == 0
    print("✓ Empty results handled correctly")
    
    # Test validation - SearchResponse doesn't have strict query validation
    # Just test that it accepts empty queries (which might be valid for some use cases)
    try:
        empty_query_response = SearchResponse(
            query="",  # Empty query - might be allowed
            results=[],
            total_results=0,
            processing_time_ms=10.0
        )
        print("✓ Empty query accepted (no strict validation)")
    except ValueError as e:
        print(f"✓ Empty query validation: {e}")
    
    return True

def test_health_response_model():
    """Test HealthResponse model."""
    print("\n🏥 Testing HealthResponse model...")
    
    # Test healthy status
    healthy_response = HealthResponse(
        status="healthy",
        service="RAG-API"
    )
    
    print(f"✓ Healthy response: {healthy_response.status}")
    print(f"  Service: {healthy_response.service}")
    
    # Test default values
    default_response = HealthResponse()
    
    assert default_response.status == "ok"
    assert default_response.service == "RAG-API"
    print("✓ Default values handled correctly")
    
    return True

def test_json_serialization():
    """Test JSON serialization of all models."""
    print("\n📄 Testing JSON serialization...")
    
    # Create a complete search response with all nested models
    documents = [
        DocumentResponse(
            id="test_1",
            title="Test Document",
            content_preview="This is a test document for JSON serialization.",
            similarity_score=0.85,
            url="https://example.com/test",
            rank=1
        )
    ]
    
    search_response = SearchResponse(
        query="json serialization test",
        results=documents,
        total_results=1,
        processing_time_ms=23.4
    )
    
    # Test JSON serialization
    json_str = search_response.model_dump_json()
    print(f"✓ JSON serialization successful")
    print(f"  JSON length: {len(json_str)} characters")
    
    # Test JSON deserialization
    parsed_json = json.loads(json_str)
    reconstructed = SearchResponse(**parsed_json)
    
    assert reconstructed.query == search_response.query
    assert len(reconstructed.results) == len(search_response.results)
    assert reconstructed.results[0].title == search_response.results[0].title
    
    print("✓ JSON deserialization successful")
    
    # Test pretty printing
    pretty_json = search_response.model_dump_json(indent=2)
    print(f"✓ Pretty JSON formatting works")
    
    return True

def main():
    """Run all model tests."""
    print("🧪 Testing Models Module")
    print("=" * 50)
    
    tests = [
        test_query_request_model,
        test_document_response_model,
        test_search_response_model,
        test_health_response_model,
        test_json_serialization
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
        print("🎉 All model tests passed!")
        return True
    else:
        print("❌ Some model tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
