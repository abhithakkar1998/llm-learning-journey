"""
FastAPI application for the RAG Pipeline.

This module provides REST API endpoints for semantic search functionality
using the RAG pipeline we've built.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from .rag_pipeline import RAGPipeline
from .models import QueryRequest, SearchResponse, HealthResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: RAGPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown logic:
    - Initialize RAG pipeline on startup
    - Clean up resources on shutdown
    """
    # Startup
    logger.info("🚀 Starting RAG API server...")
    global pipeline
    
    try:
        # Get configuration from environment variables
        data_dir = os.environ.get("RAG_DATA_DIR", "data")
        model_name = os.environ.get("RAG_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        skip_init = os.environ.get("RAG_SKIP_INIT", "false").lower() == "true"
        
        # Initialize pipeline
        pipeline = RAGPipeline(data_dir=data_dir, model_name=model_name)
        
        if not skip_init:
            # Initialize corpus (this may take a few minutes on first run)
            logger.info("Initializing RAG corpus...")
            doc_count = pipeline.initialize_corpus()
            logger.info(f"✓ RAG pipeline ready with {doc_count} documents")
        else:
            logger.info("⚠️  Pipeline initialization skipped")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG API server...")


# Create FastAPI app with lifespan management
app = FastAPI(
    title="RAG Pipeline API",
    description="Semantic search API using Retrieval-Augmented Generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware for web frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "search": "/search"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the RAG pipeline and system health.
    """
    try:
        if pipeline is None:
            return HealthResponse(
                status="unhealthy",
                message="Pipeline not initialized"
            )
        
        # Get pipeline status
        pipeline_status = pipeline.get_status()
        
        # Determine if this is first-time initialization
        data_dir = pipeline.data_dir
        embeddings_path = os.path.join(data_dir, "wikipedia_embeddings.npy")
        corpus_path = os.path.join(data_dir, "wikipedia_corpus.json")
        
        is_first_time = not (os.path.exists(embeddings_path) and os.path.exists(corpus_path))
        
        if not pipeline_status["initialized"]:
            status_msg = "first_time_init" if is_first_time else "initializing"
            message = "First-time setup in progress (may take 5-10 minutes)" if is_first_time else "Pipeline is initializing"
            
            return HealthResponse(
                status=status_msg,
                message=message,
                details={
                    **pipeline_status,
                    "is_first_time_init": is_first_time,
                    "expected_duration": "5-10 minutes" if is_first_time else "30-60 seconds"
                }
            )
        
        return HealthResponse(
            status="healthy",
            message="RAG pipeline is ready",
            details={
                **pipeline_status,
                "is_first_time_init": False
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            message=f"Health check error: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse)
async def search_documents(query_request: QueryRequest):
    """
    Semantic search endpoint.
    
    Performs semantic search across the document corpus using the query.
    
    Args:
        query_request: Search parameters including query text, top_k, and similarity threshold
        
    Returns:
        SearchResponse with ranked results and metadata
        
    Raises:
        HTTPException: If pipeline is not ready or search fails
    """
    try:
        # Validate pipeline is ready
        if pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline not initialized"
            )
        
        if not pipeline.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline is still initializing. Please try again in a moment."
            )
        
        # Validate query
        if not query_request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        # Execute search
        logger.info(f"Search request: '{query_request.query}' (top_k={query_request.top_k})")
        
        response = pipeline.search(query_request)
        
        logger.info(f"Search completed: {response.total_results} results in {response.processing_time_ms:.1f}ms")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@app.post("/reload")
async def reload_pipeline(background_tasks: BackgroundTasks, force_regenerate: bool = False):
    """
    Reload the RAG pipeline data.
    
    Useful for updating the corpus or refreshing embeddings.
    This operation runs in the background to avoid blocking the API.
    
    Args:
        force_regenerate: Whether to force regeneration of embeddings
        
    Returns:
        Confirmation message
    """
    try:
        if pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="Pipeline not available"
            )
        
        # Schedule reload in background
        background_tasks.add_task(
            _reload_pipeline_background,
            force_regenerate
        )
        
        return {
            "message": "Pipeline reload started in background",
            "force_regenerate": force_regenerate
        }
        
    except Exception as e:
        logger.error(f"Reload request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Reload failed: {str(e)}"
        )


async def _reload_pipeline_background(force_regenerate: bool):
    """Background task for reloading pipeline."""
    try:
        logger.info(f"Starting background pipeline reload (force_regenerate={force_regenerate})")
        
        start_time = time.time()
        doc_count = pipeline.reload_data(force_regenerate=force_regenerate)
        elapsed_time = time.time() - start_time
        
        logger.info(f"✓ Pipeline reloaded successfully: {doc_count} documents in {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Background reload failed: {e}")


@app.get("/status")
async def get_pipeline_status():
    """
    Get detailed pipeline status.
    
    Returns comprehensive information about the pipeline state,
    including initialization status, document count, and configuration.
    """
    try:
        if pipeline is None:
            return {"status": "not_initialized", "pipeline": None}
        
        pipeline_status = pipeline.get_status()
        
        return {
            "status": "ready" if pipeline_status["initialized"] else "initializing",
            "pipeline": pipeline_status,
            "api": {
                "version": "1.0.0",
                "endpoints": ["/", "/health", "/search", "/reload", "/status"]
            }
        }
        
    except Exception as e:
        logger.error(f"Status request failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors with helpful message."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "Check /docs for available endpoints",
            "available_endpoints": ["/", "/health", "/search", "/reload", "/status"]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors with logging."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Please check the server logs for details"
        }
    )


if __name__ == "__main__":
    # This allows running the API directly with: python -m src.api
    import uvicorn
    
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
