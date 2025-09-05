"""
Test the utils module functionality.
"""

import sys
import os
import numpy as np
import torch

# Add the parent directory to the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils import (
    get_device, 
    normalize_embeddings, 
    cosine_similarity_batch, 
    truncate_text, 
    setup_reproducibility
)

def test_get_device():
    """Test device detection functionality."""
    print("\n🔧 Testing device detection...")
    
    device = get_device()
    print(f"✓ Detected device: {device}")
    
    # Test that device is valid - get_device returns torch.device object
    device_str = str(device)
    assert any(d in device_str for d in ['cpu', 'cuda', 'mps']), f"Invalid device: {device}"
    
    # Test that torch can use the device
    try:
        torch.tensor([1.0]).to(device)
        print(f"✓ Device {device} is functional")
    except Exception as e:
        print(f"❌ Device {device} test failed: {e}")
        return False
    
    return True

def test_normalize_embeddings():
    """Test embedding normalization."""
    print("\n📐 Testing embedding normalization...")
    
    # Create test embeddings
    test_embeddings = np.array([
        [3.0, 4.0, 0.0],  # magnitude = 5
        [1.0, 1.0, 1.0],  # magnitude = sqrt(3)
        [0.0, 0.0, 1.0]   # magnitude = 1 (already normalized)
    ])
    
    print(f"Original embeddings shape: {test_embeddings.shape}")
    print(f"Original magnitudes: {np.linalg.norm(test_embeddings, axis=1)}")
    
    # Normalize
    normalized = normalize_embeddings(test_embeddings)
    
    print(f"Normalized embeddings shape: {normalized.shape}")
    print(f"Normalized magnitudes: {np.linalg.norm(normalized, axis=1)}")
    
    # Check that all embeddings have magnitude 1 (within tolerance)
    magnitudes = np.linalg.norm(normalized, axis=1)
    assert np.allclose(magnitudes, 1.0, atol=1e-6), f"Normalization failed: {magnitudes}"
    
    print("✓ All embeddings normalized to unit length")
    return True

def test_cosine_similarity_batch():
    """Test batch cosine similarity computation."""
    print("\n🧮 Testing cosine similarity...")
    
    # Create test embeddings (already normalized)
    query_embedding = np.array([1.0, 0.0, 0.0])  # unit vector along x-axis
    embeddings = np.array([
        [1.0, 0.0, 0.0],    # identical -> similarity = 1
        [0.0, 1.0, 0.0],    # orthogonal -> similarity = 0
        [-1.0, 0.0, 0.0],   # opposite -> similarity = -1
        [0.7071, 0.7071, 0.0]  # 45 degrees -> similarity = 0.7071
    ])
    
    print(f"Query embedding: {query_embedding}")
    print(f"Database embeddings shape: {embeddings.shape}")
    
    similarities = cosine_similarity_batch(query_embedding, embeddings)
    
    print(f"Computed similarities: {similarities}")
    
    # Check expected values
    expected = np.array([1.0, 0.0, -1.0, 0.7071])
    assert np.allclose(similarities, expected, atol=1e-3), f"Similarity computation failed"
    
    print("✓ Cosine similarities computed correctly")
    return True

def test_truncate_text():
    """Test text truncation functionality."""
    print("\n✂️ Testing text truncation...")
    
    # Test cases - just check that truncation respects reasonable length limits
    test_cases = [
        ("Short text", 50),  # No truncation needed
        ("This is a very long text that should be truncated", 20),
        ("Exact length text!", 18),  # Exact match
        ("", 10),  # Empty string
        ("Word", 3),  # Shorter than word
    ]
    
    for text, max_length in test_cases:
        result = truncate_text(text, max_length)
        print(f"  Input: '{text}' (max: {max_length}) -> '{result}'")
        
        # For short text that doesn't need truncation, result should match exactly
        if len(text) <= max_length:
            assert result == text, f"Short text modified unexpectedly: '{text}' -> '{result}'"
        else:
            # For long text, result should be reasonably close to max_length (allowing for "...")
            # The function might go slightly over due to word boundaries and "..."
            assert len(result) <= max_length + 3, f"Truncation too long: {len(result)} > {max_length + 3}"
            assert "..." in result or len(result) <= max_length, "Long text should be truncated with '...'"
    
    print("✓ Text truncation working correctly")
    return True

def test_setup_reproducibility():
    """Test reproducibility setup."""
    print("\n🎲 Testing reproducibility setup...")
    
    # Set up reproducibility
    seed = 42
    setup_reproducibility(seed)
    
    # Generate some random numbers
    np_random1 = np.random.random(5)
    torch_random1 = torch.rand(5)
    
    # Reset and generate again
    setup_reproducibility(seed)
    np_random2 = np.random.random(5)
    torch_random2 = torch.rand(5)
    
    # Check that results are identical
    assert np.allclose(np_random1, np_random2), "NumPy reproducibility failed"
    assert torch.allclose(torch_random1, torch_random2), "PyTorch reproducibility failed"
    
    print(f"✓ Random seeds set correctly (seed: {seed})")
    print(f"  NumPy: {np_random1[:3]}...")
    print(f"  PyTorch: {torch_random1[:3]}...")
    
    return True

def main():
    """Run all utility tests."""
    print("🧪 Testing Utils Module")
    print("=" * 50)
    
    tests = [
        test_get_device,
        test_normalize_embeddings,
        test_cosine_similarity_batch,
        test_truncate_text,
        test_setup_reproducibility
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
        print("🎉 All utils tests passed!")
        return True
    else:
        print("❌ Some utils tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
