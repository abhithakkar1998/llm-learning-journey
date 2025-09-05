"""
Utility functions for the RAG pipeline.
Contains reusable helper functions for device detection, normalization, etc.
"""

import torch
import numpy as np
from typing import Union, List
from sklearn.preprocessing import normalize


def get_device() -> torch.device:
    """
    Detect the best available device for computation.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device: The optimal device for computation
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    else:
        return torch.device("cpu")


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit vectors for cosine similarity.
    
    This converts embeddings so their magnitude is 1, making cosine similarity
    equivalent to a simple dot product (much faster computation).
    
    Args:
        embeddings: Array of embeddings to normalize (n_docs, embedding_dim)
        
    Returns:
        Normalized embeddings with L2 norm = 1
    """
    return normalize(embeddings, norm='l2', axis=1)


def cosine_similarity_batch(query_embedding: np.ndarray, 
                          document_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and document embeddings efficiently.
    
    For normalized vectors: cosine_similarity = dot_product
    This is much faster than computing the full cosine similarity formula.
    
    Args:
        query_embedding: Single query embedding (1, embedding_dim)
        document_embeddings: Document embeddings (n_docs, embedding_dim)
        
    Returns:
        Similarity scores (n_docs,) - higher values = more similar
    """
    # Ensure both are normalized
    query_norm = normalize_embeddings(query_embedding.reshape(1, -1))
    doc_norm = normalize_embeddings(document_embeddings)
    
    # Compute dot product (equivalent to cosine similarity for normalized vectors)
    similarities = np.dot(query_norm, doc_norm.T).flatten()
    return similarities


def truncate_text(text: str, max_length: int = 512) -> str:
    """
    Truncate text to maximum length while preserving word boundaries.
    
    This prevents cutting words in half, which could affect embedding quality.
    
    Args:
        text: Input text to truncate
        max_length: Maximum character length
        
    Returns:
        Truncated text with "..." if truncated
    """
    if len(text) <= max_length:
        return text
    
    # Find the last complete word within the limit
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    # If we found a space reasonably close to the limit, use it
    if last_space > max_length * 0.8:  
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."


def setup_reproducibility(seed: int = 42) -> None:
    """
    Set random seeds for reproducible results across the pipeline.
    
    This ensures consistent results when running experiments or demos.
    
    Args:
        seed: Random seed value (42 is a common default)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
