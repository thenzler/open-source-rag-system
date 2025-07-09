"""
Common schemas for the RAG System API.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    services: Dict[str, str]
    timestamp: Optional[datetime] = None


class StatsResponse(BaseModel):
    """System statistics response schema."""
    documents: Dict[str, Any]
    chunks: Dict[str, Any]
    queries: Dict[str, Any]
    system: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: Dict[str, Any]


class MessageResponse(BaseModel):
    """Simple message response schema."""
    message: str
    details: Optional[Dict[str, Any]] = None
