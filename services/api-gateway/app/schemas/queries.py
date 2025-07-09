"""
Pydantic schemas for query-related API endpoints.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union

from pydantic import BaseModel, Field, validator


class QueryRequest(BaseModel):
    """Schema for basic query request."""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class AdvancedQueryRequest(QueryRequest):
    """Schema for advanced query request with additional options."""
    enable_reranking: bool = Field(default=True)
    enable_query_expansion: bool = Field(default=False)
    document_ids: Optional[List[str]] = Field(default=None)
    date_range: Optional[Dict[str, str]] = Field(default=None)
    semantic_search_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_search_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    
    @validator('semantic_search_weight', 'keyword_search_weight')
    def validate_weights(cls, v, values):
        # Check that weights sum to 1.0 (approximately)
        if 'semantic_search_weight' in values:
            total = v + values['semantic_search_weight']
            if abs(total - 1.0) > 0.01:
                raise ValueError('Search weights must sum to 1.0')
        return v


class SearchResult(BaseModel):
    """Schema for individual search result."""
    document_id: str
    chunk_id: str
    content: str
    score: float
    document_title: Optional[str] = None
    document_filename: str
    chunk_index: int
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    highlights: Optional[List[str]] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


class QueryResponse(BaseModel):
    """Schema for query response."""
    query: str
    results: List[SearchResult]
    total_results: int
    max_score: float
    avg_score: float
    processing_time_ms: int
    used_models: Dict[str, str] = Field(default_factory=dict)
    query_expansion: Optional[List[str]] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


class SimilarDocumentsRequest(BaseModel):
    """Schema for finding similar documents."""
    document_id: str
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.7, ge=0.0, le=1.0)
    include_content: bool = Field(default=False)


class SimilarDocumentResult(BaseModel):
    """Schema for similar document result."""
    document_id: str
    title: Optional[str] = None
    filename: str
    similarity_score: float
    category: Optional[str] = None
    created_at: datetime
    chunk_matches: int = 0
    content_preview: Optional[str] = None
    
    class Config:
        from_attributes = True


class SimilarDocumentsResponse(BaseModel):
    """Schema for similar documents response."""
    source_document_id: str
    similar_documents: List[SimilarDocumentResult]
    total_found: int
    processing_time_ms: int
    
    class Config:
        from_attributes = True


class QueryAnalyticsRequest(BaseModel):
    """Schema for query analytics request."""
    start_date: Optional[str] = Field(default=None)
    end_date: Optional[str] = Field(default=None)
    granularity: str = Field(default="day", regex="^(hour|day|week|month)$")
    user_id: Optional[str] = Field(default=None)
    query_type: Optional[str] = Field(default=None)


class QueryAnalyticsResponse(BaseModel):
    """Schema for query analytics response."""
    total_queries: int
    unique_users: int
    avg_response_time_ms: float
    success_rate: float
    popular_queries: List[Dict[str, Union[str, int]]]
    queries_over_time: List[Dict[str, Union[str, int]]]
    query_types: Dict[str, int]
    
    class Config:
        from_attributes = True


class QueryLogResponse(BaseModel):
    """Schema for query log response."""
    id: str
    query_text: str
    query_type: str
    user_id: str
    result_count: int
    max_score: Optional[float] = None
    response_time_ms: int
    success: bool
    created_at: datetime
    
    class Config:
        from_attributes = True
