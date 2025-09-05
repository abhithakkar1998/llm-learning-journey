"""
Test the complete RAG pipeline functionality.
"""

import sys
import os
# Add the parent directory to the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.rag_pipeline import RAGPipeline
from src.models import QueryRequest

def test_rag_pipeline_integration():
    """Test the complete RAG pipeline functionality."""
    print("🚀 Testing RAG Pipeline Integration")
    print("=" * 50)
    
    try:
        # Get the correct data directory path within Day7 folder
        current_dir = os.path.dirname(os.path.dirname(__file__))  # Go up one level from test_scripts
        data_dir = os.path.join(current_dir, "data")
        
        # Initialize pipeline
        print(f"\n1. Creating RAG Pipeline...")
        print(f"   Data directory: {data_dir}")
        pipeline = RAGPipeline(data_dir=data_dir)
        
        # Check initial status
        status = pipeline.get_status()
        print(f"Initial status: {status}")
        
        # Initialize corpus
        print("\n2. Initializing corpus...")
        doc_count = pipeline.initialize_corpus()
        print(f"✓ Loaded {doc_count} documents")
        
        # Verify we have documents
        if doc_count == 0:
            print("❌ No documents loaded!")
            return False
        
        # Check status after initialization
        status = pipeline.get_status()
        print(f"Status after init: {status}")
        
        # Verify pipeline is initialized
        if not status.get('initialized', False):
            print("❌ Pipeline not properly initialized!")
            return False
        
        # Test search functionality
        print("\n3. Testing search functionality...")
        
        test_queries = [
            "machine learning algorithms",
            "deep neural networks", 
            "data preprocessing techniques",
            "computer vision applications"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Search {i}: '{query}' ---")
            
            # Create query request
            query_request = QueryRequest(
                query=query,
                top_k=3,
                min_similarity=0.1
            )
            
            # Execute search
            response = pipeline.search(query_request)
            
            print(f"Processing time: {response.processing_time_ms:.1f}ms")
            print(f"Found {response.total_results} results:")
            
            # Verify we got results
            if response.total_results == 0:
                print(f"⚠️  No results for query: {query}")
            
            for j, result in enumerate(response.results, 1):
                print(f"  {j}. {result.title}")
                print(f"     Similarity: {result.similarity_score:.3f}")
                print(f"     Preview: {result.content_preview[:100]}...")
                print()
        
        print("✅ RAG Pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ RAG Pipeline test failed with error: {e}")
        return False


def main():
    """Run RAG pipeline integration tests."""
    print("🧪 Testing RAG Pipeline Integration")
    print("=" * 50)
    
    tests = [
        test_rag_pipeline_integration
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
        print("🎉 All RAG pipeline tests passed!")
        return True
    else:
        print("❌ Some RAG pipeline tests failed!")
        return False

if __name__ == "__main__":
    main()
