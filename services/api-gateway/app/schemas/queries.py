"""
Query schemas for the RAG System API.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class QueryRequest(BaseModel):
    """Basic query request schema."""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True


class AdvancedQueryRequest(BaseModel):
    """Advanced query request schema."""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    enable_reranking: bool = True
    enable_query_expansion: bool = True
    document_types: Optional[List[str]] = None
    date_range: Optional[Dict[str, str]] = None
    semantic_threshold: Optional[float] = None


class QueryResultItem(BaseModel):
    """Individual query result item schema."""
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = {}
    document_title: Optional[str] = None
    document_filename: Optional[str] = None
    chunk_index: Optional[int] = None
    page_number: Optional[int] = None


class QueryResponse(BaseModel):
    """Query response schema."""
    query: str
    results: List[QueryResultItem]
    total: int
    processing_time: float
    reranked: bool = False
    query_expanded: bool = False
    filters_applied: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}


class SimilarDocumentItem(BaseModel):
    """Similar document item schema."""
    document_id: str
    title: Optional[str] = None
    filename: str
    similarity_score: float
    metadata: Dict[str, Any] = {}
    snippet: Optional[str] = None


class SimilarDocumentsResponse(BaseModel):
    """Similar documents response schema."""
    document_id: str
    similar_documents: List[SimilarDocumentItem]
    total: int
    processing_time: float
    algorithm: str = "vector_similarity"


class QueryAnalyticsItem(BaseModel):
    """Query analytics item schema."""
    query: str
    count: int
    avg_score: Optional[float] = None
    avg_processing_time: Optional[float] = None
    last_used: Optional[datetime] = None


class QueryAnalyticsResponse(BaseModel):
    """Query analytics response schema."""
    period: Dict[str, Any]
    summary: Dict[str, Any]
    popular_queries: List[QueryAnalyticsItem]
    timeline: List[Dict[str, Any]] = []
    trends: Dict[str, Any] = {}


class QueryLogEntry(BaseModel):
    """Query log entry schema."""
    id: str
    user_id: str
    query: str
    timestamp: datetime
    processing_time: float
    result_count: int
    filters_used: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None


class QuerySuggestion(BaseModel):
    """Query suggestion schema."""
    suggestion: str
    confidence: float
    type: str = "completion"  # completion, correction, related
    metadata: Dict[str, Any] = {}


class QuerySuggestionsResponse(BaseModel):
    """Query suggestions response schema."""
    query: str
    suggestions: List[QuerySuggestion]
    total: int
    processing_time: float
