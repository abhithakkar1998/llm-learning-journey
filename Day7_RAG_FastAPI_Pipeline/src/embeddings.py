"""
Embedding generation module using sentence transformers.
Handles document embedding creation, caching, and management.
"""

import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import logging
from .utils import get_device, normalize_embeddings, setup_reproducibility
from .data_loader import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Handles document embedding generation using sentence transformers.
    
    This class manages:
    - Loading and initializing sentence transformer models
    - Batch processing for efficient embedding generation
    - Caching embeddings to disk for reuse
    - Device optimization (GPU/CPU)
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_dir: str = "data"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: HuggingFace model identifier for sentence transformers
            cache_dir: Directory to cache embeddings and model files
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = get_device()
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set up reproducibility
        setup_reproducibility(42)
        
        # Initialize the model
        logger.info(f"Loading sentence transformer: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=str(self.device))
        
        # Model info
        embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded successfully - Embedding dimension: {embedding_dim}")
        
        # Store for later use
        self.embedding_dim = embedding_dim
    
    def generate_embeddings(self, 
                          documents: List[Document], 
                          batch_size: int = 32,
                          normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of documents.
        
        This method:
        1. Extracts text content from Document objects
        2. Processes in batches for memory efficiency
        3. Uses sentence transformer to create embeddings
        4. Optionally normalizes for cosine similarity
        
        Args:
            documents: List of Document objects to embed
            batch_size: Number of documents to process at once
            normalize: Whether to normalize embeddings for cosine similarity
            
        Returns:
            numpy array of embeddings (n_documents, embedding_dim)
        """
        if not documents:
            raise ValueError("No documents provided for embedding generation")
        
        logger.info(f"Generating embeddings for {len(documents)} documents")
        logger.info(f"Batch size: {batch_size}, Normalize: {normalize}")
        
        # Extract text content from documents
        texts = [doc.content for doc in documents]
        
        # Generate embeddings in batches for memory efficiency
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_end = min(i + batch_size, len(texts))
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} "
                       f"(documents {i+1}-{batch_end})")
            
            # Generate embeddings for this batch
            with torch.no_grad():  # Save memory during inference
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False  # We handle progress ourselves
                )
            
            all_embeddings.append(batch_embeddings)
        
        # Combine all batches
        embeddings = np.vstack(all_embeddings)
        
        # Normalize if requested (recommended for cosine similarity)
        if normalize:
            embeddings = normalize_embeddings(embeddings)
            logger.info("✓ Embeddings normalized for cosine similarity")
        
        logger.info(f"✓ Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single query string.
        
        This method is optimized for query processing:
        1. Handles single query strings efficiently
        2. Returns normalized embeddings for similarity search
        3. Uses the same model as document embeddings for consistency
        
        Args:
            query: User's search query string
            normalize: Whether to normalize for cosine similarity (recommended)
            
        Returns:
            numpy array of query embedding (1, embedding_dim)
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace")
        
        query = query.strip()
        logger.info(f"Encoding query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        # Generate embedding for the query
        with torch.no_grad():
            query_embedding = self.model.encode(
                [query],  # SentenceTransformer expects a list
                convert_to_numpy=True,
                show_progress_bar=False
            )
        
        # Normalize if requested
        if normalize:
            query_embedding = normalize_embeddings(query_embedding)
        
        logger.info(f"✓ Query encoded to shape: {query_embedding.shape}")
        return query_embedding
    
    def save_embeddings(self, 
                       embeddings: np.ndarray, 
                       filename: str = "embeddings.npy",
                       metadata: Optional[dict] = None) -> str:
        """
        Save embeddings to disk for future use.
        
        This method:
        1. Saves embeddings as efficient NumPy binary format
        2. Optionally saves metadata (model info, generation params)
        3. Creates cache directory if needed
        4. Returns the full path for verification
        
        Args:
            embeddings: NumPy array of embeddings to save
            filename: Output filename (should end with .npy)
            metadata: Optional dict with generation info
            
        Returns:
            Full path to saved embeddings file
        """
        if embeddings is None or embeddings.size == 0:
            raise ValueError("Cannot save empty or None embeddings")
        
        # Ensure filename has .npy extension
        if not filename.endswith('.npy'):
            filename = filename + '.npy'
        
        filepath = os.path.join(self.cache_dir, filename)
        
        # Save embeddings in efficient binary format
        np.save(filepath, embeddings)
        
        # Save metadata if provided
        if metadata is not None:
            metadata_path = filepath.replace('.npy', '_metadata.json')
            import json
            
            # Add standard metadata
            full_metadata = {
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'shape': embeddings.shape,
                'dtype': str(embeddings.dtype),
                **metadata  # User-provided metadata
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2)
            
            logger.info(f"✓ Metadata saved to: {metadata_path}")
        
        logger.info(f"✓ Embeddings saved to: {filepath}")
        logger.info(f"  Shape: {embeddings.shape}, Size: {embeddings.nbytes / (1024*1024):.2f} MB")
        
        return filepath
    
    def load_embeddings(self, filename: str = "embeddings.npy") -> tuple[np.ndarray, Optional[dict]]:
        """
        Load embeddings from disk cache.
        
        This method:
        1. Loads embeddings from NumPy binary format
        2. Optionally loads associated metadata
        3. Validates the loaded data
        4. Returns both embeddings and metadata
        
        Args:
            filename: Input filename to load from
            
        Returns:
            Tuple of (embeddings_array, metadata_dict)
            
        Raises:
            FileNotFoundError: If embeddings file doesn't exist
            ValueError: If loaded embeddings are invalid
        """
        # Ensure filename has .npy extension
        if not filename.endswith('.npy'):
            filename = filename + '.npy'
        
        filepath = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")
        
        # Load embeddings
        try:
            embeddings = np.load(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load embeddings from {filepath}: {e}")
        
        # Validate loaded embeddings
        if embeddings.size == 0:
            raise ValueError(f"Loaded embeddings are empty from {filepath}")
        
        # Try to load metadata
        metadata = None
        metadata_path = filepath.replace('.npy', '_metadata.json')
        
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"✓ Metadata loaded from: {metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        logger.info(f"✓ Embeddings loaded from: {filepath}")
        logger.info(f"  Shape: {embeddings.shape}, Size: {embeddings.nbytes / (1024*1024):.2f} MB")
        
        # Validate embedding dimension matches current model
        if embeddings.shape[1] != self.embedding_dim:
            logger.warning(f"Embedding dimension mismatch: loaded={embeddings.shape[1]}, "
                          f"current model={self.embedding_dim}")
        
        return embeddings, metadata
