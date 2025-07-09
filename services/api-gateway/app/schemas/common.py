"""
Common schemas used across the application.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Union

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str = Field(..., regex="^(healthy|unhealthy|degraded)$")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, str] = Field(default_factory=dict)
    version: str = "1.0.0"
    
    class Config:
        from_attributes = True


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: Dict[str, Any] = Field(...)
    
    class Config:
        from_attributes = True
        
    @classmethod
    def create(cls, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Create error response."""
        error_data = {
            "code": code,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        if details:
            error_data["details"] = details
        return cls(error=error_data)


class SuccessResponse(BaseModel):
    """Schema for success responses."""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class StatsResponse(BaseModel):
    """Schema for system statistics response."""
    total_documents: int = 0
    total_chunks: int = 0
    total_queries: int = 0
    total_users: int = 0
    storage_used_bytes: int = 0
    processing_queue_size: int = 0
    active_sessions: int = 0
    uptime_seconds: int = 0
    
    # Performance metrics
    avg_query_time_ms: Optional[float] = None
    avg_processing_time_ms: Optional[float] = None
    success_rate: Optional[float] = None
    
    # Resource usage
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    disk_usage_percent: Optional[float] = None
    
    # Recent activity
    queries_last_hour: int = 0
    uploads_last_hour: int = 0
    errors_last_hour: int = 0
    
    class Config:
        from_attributes = True


class PaginationParams(BaseModel):
    """Schema for pagination parameters."""
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)
    
    class Config:
        from_attributes = True


class SortParams(BaseModel):
    """Schema for sorting parameters."""
    sort_by: str = Field(default="created_at")
    sort_order: str = Field(default="desc", regex="^(asc|desc)$")
    
    class Config:
        from_attributes = True


class FilterParams(BaseModel):
    """Schema for filtering parameters."""
    category: Optional[str] = None
    status: Optional[str] = None
    user_id: Optional[str] = None
    search: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    tags: Optional[List[str]] = None
    
    class Config:
        from_attributes = True


class ConfigResponse(BaseModel):
    """Schema for configuration response."""
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    max_query_length: int
    max_file_size_mb: int
    allowed_mime_types: List[str]
    features: Dict[str, bool]
    rate_limits: Dict[str, int]
    
    class Config:
        from_attributes = True


class MetricsResponse(BaseModel):
    """Schema for metrics response."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metrics: Dict[str, Union[int, float, str]]
    
    class Config:
        from_attributes = True


class BatchOperationResponse(BaseModel):
    """Schema for batch operation response."""
    total_requested: int
    successful: int
    failed: int
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        from_attributes = True
