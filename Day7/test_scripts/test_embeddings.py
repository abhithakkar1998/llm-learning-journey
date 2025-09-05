"""
Test the embeddings module functionality.
"""

import sys
import os
import numpy as np
import tempfile
import json

# Add the parent directory to the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.embeddings import EmbeddingGenerator
from src.data_loader import Document

def test_embedding_generator_initialization():
    """Test EmbeddingGenerator initialization."""
    print("\n🤖 Testing EmbeddingGenerator initialization...")
    
    # Test default initialization
    generator = EmbeddingGenerator()
    print(f"✓ Default generator created with model: {generator.model_name}")
    print(f"✓ Device: {generator.device}")
    
    # Check that model is loaded
    assert hasattr(generator, 'model'), "Model not loaded"
    assert hasattr(generator, 'embedding_dim'), "Embedding dimension not available"
    print("✓ Model loaded successfully")
    
    # Test custom model name
    try:
        custom_generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        print(f"✓ Custom model generator created: {custom_generator.model_name}")
    except Exception as e:
        print(f"⚠️  Custom model test skipped: {e}")
    
    return True

def test_encode_query():
    """Test query encoding functionality."""
    print("\n🔍 Testing query encoding...")
    
    generator = EmbeddingGenerator()
    
    # Test single query
    query = "machine learning algorithms"
    embedding = generator.encode_query(query)
    
    print(f"Query: '{query}'")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding type: {type(embedding)}")
    
    # Check embedding properties
    assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
    assert len(embedding.shape) == 2, "Query embedding should be 2D (1, embedding_dim)"
    assert embedding.shape[0] == 1, "Query embedding should have batch size 1"
    assert embedding.shape[1] > 0, "Embedding should have positive dimension"
    
    # For consistency with similarity calculations, convert to 1D
    embedding_1d = embedding.flatten()
    
    # Check that embedding is normalized (unit length)
    magnitude = np.linalg.norm(embedding_1d)
    print(f"Embedding magnitude: {magnitude:.6f}")
    assert abs(magnitude - 1.0) < 1e-5, f"Embedding not normalized: {magnitude}"
    
    print("✓ Query encoding successful")
    
    # Test different queries produce different embeddings
    queries = [
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "computer vision"
    ]
    
    embeddings = []
    for q in queries:
        emb = generator.encode_query(q)
        embeddings.append(emb.flatten())  # Convert to 1D for comparison
    
    # Check that embeddings are different
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = np.dot(embeddings[i], embeddings[j])
            print(f"Similarity '{queries[i]}' vs '{queries[j]}': {similarity:.3f}")
            assert similarity < 0.99, "Embeddings too similar"
    
    print("✓ Different queries produce different embeddings")
    return True

def test_batch_encode_documents():
    """Test batch document encoding."""
    print("\n📚 Testing batch document encoding...")
    
    generator = EmbeddingGenerator()
    
    # Create test documents
    documents = [
        Document(
            id="doc_1",
            title="Machine Learning Basics",
            content="Machine learning is a method of data analysis that automates analytical model building.",
            url="https://example.com/ml"
        ),
        Document(
            id="doc_2",
            title="Deep Neural Networks",
            content="Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
            url="https://example.com/dl"
        ),
        Document(
            id="doc_3",
            title="Computer Vision",
            content="Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images.",
            url="https://example.com/cv"
        )
    ]
    
    print(f"Encoding {len(documents)} documents...")
    
    # Test batch encoding
    embeddings = generator.generate_embeddings(documents, batch_size=2)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Expected shape: ({len(documents)}, {generator.embedding_dim})")
    
    # Check embedding properties
    assert embeddings.shape[0] == len(documents), "Wrong number of embeddings"
    assert embeddings.shape[1] > 0, "Wrong embedding dimension"
    
    # Check that embeddings are normalized
    magnitudes = np.linalg.norm(embeddings, axis=1)
    print(f"Embedding magnitudes: {magnitudes}")
    assert np.allclose(magnitudes, 1.0, atol=1e-5), "Embeddings not normalized"
    
    print("✓ Batch encoding successful")
    
    # Test empty list
    try:
        empty_embeddings = generator.generate_embeddings([])
        print("⚠️  Empty list should have raised ValueError")
    except ValueError:
        print("✓ Empty document list handled correctly")
    except Exception as e:
        print(f"✓ Empty document list handled with exception: {type(e).__name__}")
    
    return True

def test_caching_functionality():
    """Test embedding caching functionality."""
    print("\n💾 Testing caching functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        generator = EmbeddingGenerator(cache_dir=temp_dir)
        
        # Create test documents
        documents = [
            Document(
                id="cache_1",
                title="Test Doc 1",
                content="This is the first test document for caching.",
                url="https://example.com/1"
            ),
            Document(
                id="cache_2",
                title="Test Doc 2", 
                content="This is the second test document for caching.",
                url="https://example.com/2"
            )
        ]
        
        print(f"Cache directory: {temp_dir}")
        
        # Generate embeddings
        embeddings = generator.generate_embeddings(documents)
        
        # Save embeddings
        metadata = {
            "document_count": len(documents),
            "test_run": True
        }
        
        saved_path = generator.save_embeddings(
            embeddings, 
            filename="test_embeddings.npy",
            metadata=metadata
        )
        
        print("✓ Embeddings saved to cache")
        
        # Check that files exist
        assert os.path.exists(saved_path), "Embeddings file not created"
        
        # Load embeddings
        loaded_embeddings, loaded_metadata = generator.load_embeddings("test_embeddings.npy")
        
        print("✓ Embeddings loaded from cache")
        print(f"Loaded embeddings shape: {loaded_embeddings.shape}")
        
        # Verify loaded data
        assert np.array_equal(embeddings, loaded_embeddings), "Embeddings don't match"
        assert loaded_metadata['document_count'] == len(documents), "Metadata mismatch"
        assert loaded_metadata['test_run'] == True, "Custom metadata mismatch"
        
        print("✓ Cache data verified")
    
    return True

def test_error_handling():
    """Test error handling in embeddings module."""
    print("\n🛡️ Testing error handling...")
    
    generator = EmbeddingGenerator()
    
    # Test encoding invalid input
    try:
        result = generator.encode_query("")
        print(f"Empty query result shape: {result.shape}")
        print("✓ Empty query handled")
    except Exception as e:
        print(f"✓ Empty query caused expected error: {e}")
    
    # Test loading non-existent cache
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            fake_embeddings = os.path.join(temp_dir, "fake.npy")
            embeddings, metadata = generator.load_embeddings("fake.npy")
            print("⚠️  Should have raised FileNotFoundError")
        except FileNotFoundError:
            print("✓ Non-existent cache files handled correctly")
        except Exception as e:
            print(f"✓ Non-existent cache handled with exception: {type(e).__name__}")
    
    # Test invalid save parameters
    try:
        generator.save_embeddings(None)
        print("⚠️  Should have failed to save None embeddings")
    except (ValueError, AttributeError):
        print("✓ Invalid save parameters handled correctly")
    except Exception as e:
        print(f"✓ Invalid save parameters handled with exception: {type(e).__name__}")
    
    return True

def main():
    """Run all embeddings tests."""
    print("🧪 Testing Embeddings Module")
    print("=" * 50)
    
    tests = [
        test_embedding_generator_initialization,
        test_encode_query,
        test_batch_encode_documents,
        test_caching_functionality,
        test_error_handling
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
        print("🎉 All embeddings tests passed!")
        return True
    else:
        print("❌ Some embeddings tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
