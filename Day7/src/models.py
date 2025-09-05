"""
Pydantic models for API request/response validation.
Defines data structures for FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

# --- Request Models ---

class QueryRequest(BaseModel):
    """
    Model for a user's search query request.
    FastAPI will use this to validate the incoming JSON body of a POST request.
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="The search query text from the user."
    )
    top_k: int = Field(
        5,
        gt=0,
        le=20,
        description="The number of top results to return."
    )
    min_similarity: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="The minimum similarity score for a result to be included."
    )

# --- Response Models ---

class DocumentResponse(BaseModel):
    """
    Model for a single document returned in the search results.
    """
    id: str
    title: str
    content_preview: str = Field(..., description="A short preview of the document content.")
    url: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., gt=0)

    class Config:
        # This allows the model to be created from arbitrary class instances,
        # which is useful when converting our internal SearchResult objects.
        from_attributes = True


class SearchResponse(BaseModel):
    """
    The final response model for a search query.
    This is the structure of the JSON that will be sent back to the client.
    """
    query: str
    results: List[DocumentResponse]
    total_results: int
    processing_time_ms: float = Field(..., description="Time taken to process the request in milliseconds.")

class HealthResponse(BaseModel):
    """
    Model for the API health check endpoint.
    """
    status: str = "ok"
    service: str = "RAG-API"

