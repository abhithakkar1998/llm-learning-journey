"""
Document retrieval module for semantic search.
Implements similarity-based document retrieval using embeddings.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass
from .utils import cosine_similarity_batch
from .data_loader import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Data class for individual search results.
    
    Contains both the document and its relevance score for easy handling.
    """
    document: Document
    similarity_score: float
    rank: int
    
    def __post_init__(self):
        """Ensure similarity score is valid."""
        if not (0.0 <= self.similarity_score <= 1.0):
            logger.warning(f"Unusual similarity score: {self.similarity_score}")


class DocumentRetriever:
    """
    Handles semantic document retrieval using pre-computed embeddings.
    
    This class manages:
    - Document corpus and their embeddings
    - Similarity computation between queries and documents
    - Ranking and filtering of search results
    - Result formatting and presentation
    """
    
    def __init__(self, 
                 documents: List[Document], 
                 embeddings: np.ndarray):
        """
        Initialize the document retriever.
        
        Args:
            documents: List of Document objects to search through
            embeddings: Pre-computed document embeddings (n_docs, embedding_dim)
            
        Raises:
            ValueError: If documents and embeddings don't match in length
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError(f"Documents ({len(documents)}) and embeddings ({embeddings.shape[0]}) "
                           f"length mismatch")
        
        if len(documents) == 0:
            raise ValueError("Cannot initialize retriever with empty document set")
        
        self.documents = documents
        self.embeddings = embeddings
        
        logger.info(f"DocumentRetriever initialized with {len(documents)} documents")
        logger.info(f"Embedding shape: {embeddings.shape}")
        
        # Pre-compute some statistics for logging
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        self.is_normalized = np.allclose(embedding_norms, 1.0, atol=1e-3)
        
        if self.is_normalized:
            logger.info("✓ Embeddings are normalized (ready for cosine similarity)")
        else:
            logger.warning("Embeddings may not be normalized - similarity scores might be affected")

    def retrieve_top_k(self,
                       query_embedding: np.ndarray,
                       top_k: int = 5,
                       min_similarity: float = 0.1) -> List[SearchResult]:
        """
        Retrieve the top-k most similar documents for a given query embedding.

        This method:
        1. Computes similarity between the query and all documents
        2. Sorts documents by relevance
        3. Filters out low-similarity results
        4. Returns a ranked list of SearchResult objects

        Args:
            query_embedding: Embedding of the user's search query
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity score to be considered a match

        Returns:
            A list of SearchResult objects, ranked by similarity
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Compute cosine similarity between query and all documents
        similarities = cosine_similarity_batch(query_embedding, self.embeddings)

        # Get indices of top-k results
        # np.argsort returns indices that would sort the array
        # [::-1] reverses it for descending order
        top_indices = np.argsort(similarities)[::-1]

        # Filter and collect results
        results = []
        for rank, idx in enumerate(top_indices):
            score = similarities[idx]

            # Stop if score is below threshold or we have enough results
            if score < min_similarity or len(results) >= top_k:
                break

            result = SearchResult(
                document=self.documents[idx],
                similarity_score=float(score),
                rank=rank + 1
            )
            results.append(result)

        logger.info(f"Retrieved {len(results)} documents for query")
        return results
