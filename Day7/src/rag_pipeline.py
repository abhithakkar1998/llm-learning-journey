"""
RAG pipeline orchestration module.
Coordinates data loading, embedding generation, and document retrieval.
"""

import os
import time
import logging
import numpy as np
from typing import List, Optional, Tuple
from .data_loader import WikipediaDataLoader, Document, create_sample_corpus
from .embeddings import EmbeddingGenerator
from .retrieval import DocumentRetriever, SearchResult
from .models import QueryRequest, SearchResponse, DocumentResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main orchestrator for the RAG system.
    
    This class coordinates all components:
    - Data loading and management
    - Embedding generation and caching
    - Document retrieval and search
    - High-level API interface
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the RAG pipeline.
        
        Args:
            data_dir: Directory for data storage and caching
            model_name: Sentence transformer model to use
        """
        self.data_dir = data_dir
        self.model_name = model_name
        
        # Initialize components
        self.data_loader = WikipediaDataLoader(data_dir)
        self.embedding_generator = EmbeddingGenerator(model_name, data_dir)
        
        # These will be initialized when corpus is loaded
        self.documents: Optional[List[Document]] = None
        self.retriever: Optional[DocumentRetriever] = None
        
        # Track initialization state
        self.is_initialized = False
        
        logger.info(f"RAGPipeline created with data_dir: {data_dir}")
        logger.info(f"Using model: {model_name}")
    
    def initialize_corpus(self, use_cache: bool = True) -> int:
        """
        Initialize the document corpus for the RAG system.
        
        This method:
        1. Loads or creates a Wikipedia corpus
        2. Generates embeddings (with caching)
        3. Sets up the document retriever
        4. Marks the pipeline as ready for search
        
        Args:
            use_cache: Whether to use cached embeddings if available
            
        Returns:
            Number of documents loaded
        """
        logger.info("Initializing RAG corpus...")
        start_time = time.time()
        
        # Step 1: Load or create document corpus
        logger.info("Loading document corpus...")
        self.documents = create_sample_corpus(self.data_dir)
        logger.info(f"✓ Loaded {len(self.documents)} documents")
        
        # Step 2: Generate or load embeddings
        embeddings_file = "wikipedia_embeddings.npy"
        embeddings_path = os.path.join(self.data_dir, embeddings_file)
        
        if use_cache and os.path.exists(embeddings_path):
            logger.info("Loading cached embeddings...")
            embeddings, metadata = self.embedding_generator.load_embeddings(embeddings_file)
            
            # Verify embedding count matches document count
            if embeddings.shape[0] != len(self.documents):
                logger.warning(f"Embedding count ({embeddings.shape[0]}) doesn't match document count ({len(self.documents)})")
                logger.info("Regenerating embeddings...")
                embeddings = self._generate_and_cache_embeddings(embeddings_file)
        else:
            logger.info("Generating new embeddings...")
            embeddings = self._generate_and_cache_embeddings(embeddings_file)
        
        # Step 3: Initialize retriever
        logger.info("Initializing document retriever...")
        self.retriever = DocumentRetriever(self.documents, embeddings)
        
        # Mark as initialized
        self.is_initialized = True
        initialization_time = time.time() - start_time
        
        logger.info(f"✓ RAG pipeline initialized successfully in {initialization_time:.2f}s")
        logger.info(f"Ready to search {len(self.documents)} documents")
        
        return len(self.documents)
    
    def _generate_and_cache_embeddings(self, filename: str) -> np.ndarray:
        """Generate embeddings and save them to cache."""
        embeddings = self.embedding_generator.generate_embeddings(
            self.documents,
            batch_size=16,
            normalize=True
        )
        
        # Save with metadata
        metadata = {
            "num_documents": len(self.documents),
            "document_ids": [doc.id for doc in self.documents],
            "generation_timestamp": time.time()
        }
        
        self.embedding_generator.save_embeddings(embeddings, filename, metadata)
        return embeddings
    
    def search(self, query_request: QueryRequest) -> SearchResponse:
        """
        Execute end-to-end semantic search.
        
        This method:
        1. Validates the pipeline is initialized
        2. Encodes the user query
        3. Retrieves similar documents
        4. Formats results for API response
        
        Args:
            query_request: Pydantic model with search parameters
            
        Returns:
            SearchResponse with results and metadata
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize_corpus() first.")
        
        start_time = time.time()
        
        # Step 1: Encode the query
        query_embedding = self.embedding_generator.encode_query(query_request.query)
        
        # Step 2: Retrieve similar documents
        search_results = self.retriever.retrieve_top_k(
            query_embedding,
            top_k=query_request.top_k,
            min_similarity=query_request.min_similarity
        )
        
        # Step 3: Convert to API response format
        document_responses = []
        for result in search_results:
            doc_response = DocumentResponse(
                id=result.document.id,
                title=result.document.title,
                content_preview=self._create_content_preview(result.document.content),
                url=result.document.url,
                similarity_score=result.similarity_score,
                rank=result.rank
            )
            document_responses.append(doc_response)
        
        # Step 4: Create final response
        processing_time_ms = (time.time() - start_time) * 1000
        
        response = SearchResponse(
            query=query_request.query,
            results=document_responses,
            total_results=len(document_responses),
            processing_time_ms=processing_time_ms
        )
        
        logger.info(f"Search completed: '{query_request.query}' -> {len(document_responses)} results in {processing_time_ms:.1f}ms")
        return response
    
    def _create_content_preview(self, content: str, max_length: int = 200) -> str:
        """Create a preview of document content for API responses."""
        if len(content) <= max_length:
            return content
        
        # Find last complete sentence within limit
        preview = content[:max_length]
        last_period = preview.rfind('.')
        
        if last_period > max_length * 0.7:  # If period is reasonably close to end
            return preview[:last_period + 1]
        else:
            return preview + "..."
    
    def get_status(self) -> dict:
        """Get pipeline status information."""
        return {
            "initialized": self.is_initialized,
            "model_name": self.model_name,
            "document_count": len(self.documents) if self.documents else 0,
            "data_directory": self.data_dir
        }
    
    def reload_data(self, force_regenerate: bool = False) -> int:
        """
        Reload data and optionally regenerate embeddings.
        
        Useful for updating the corpus with new documents or 
        refreshing embeddings with a different model.
        
        Args:
            force_regenerate: Force regeneration of embeddings even if cached
            
        Returns:
            Number of documents loaded
        """
        logger.info("Reloading RAG data...")
        
        # Reset state
        self.is_initialized = False
        self.documents = None
        self.retriever = None
        
        # Re-initialize with appropriate cache settings
        return self.initialize_corpus(use_cache=not force_regenerate)
