"""
Pydantic schemas for document-related API endpoints.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, validator


class DocumentBase(BaseModel):
    """Base document schema."""
    filename: str = Field(..., min_length=1, max_length=255)
    title: Optional[str] = Field(None, max_length=500)
    category: Optional[str] = Field(None, max_length=100)
    tags: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DocumentUploadRequest(DocumentBase):
    """Schema for document upload request."""
    pass


class DocumentResponse(DocumentBase):
    """Schema for document response."""
    id: str
    original_filename: str
    content_type: str
    file_size: int
    status: str
    processing_progress: float = 0.0
    error_message: Optional[str] = None
    summary: Optional[str] = None
    language: str = "en"
    user_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    view_count: int = 0
    query_count: int = 0
    last_accessed: Optional[datetime] = None
    chunk_count: int = 0
    
    class Config:
        from_attributes = True
        
    @classmethod
    def from_orm(cls, obj):
        """Create from ORM object."""
        return cls(
            id=str(obj.id),
            filename=obj.filename,
            original_filename=obj.original_filename,
            content_type=obj.content_type,
            file_size=obj.file_size,
            title=obj.title,
            category=obj.category,
            tags=obj.tags or [],
            metadata=obj.metadata or {},
            status=obj.status,
            processing_progress=obj.processing_progress,
            error_message=obj.error_message,
            summary=obj.summary,
            language=obj.language,
            user_id=obj.user_id,
            created_at=obj.created_at,
            updated_at=obj.updated_at,
            processed_at=obj.processed_at,
            view_count=obj.view_count,
            query_count=obj.query_count,
            last_accessed=obj.last_accessed,
            chunk_count=len(obj.chunks) if obj.chunks else 0
        )


class DocumentListResponse(BaseModel):
    """Schema for document list response."""
    documents: List[DocumentResponse]
    total: int
    skip: int
    limit: int
    
    class Config:
        from_attributes = True


class DocumentChunkResponse(BaseModel):
    """Schema for document chunk response."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    chunk_type: str = "text"
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime
    retrieval_count: int = 0
    
    class Config:
        from_attributes = True


class DocumentStatusResponse(BaseModel):
    """Schema for document processing status."""
    document_id: str
    status: str
    processing_progress: float
    error_message: Optional[str] = None
    chunks_created: int = 0
    vectors_indexed: int = 0
    estimated_completion: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class DocumentUpdateRequest(BaseModel):
    """Schema for document update request."""
    title: Optional[str] = Field(None, max_length=500)
    category: Optional[str] = Field(None, max_length=100)
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('tags')
    def validate_tags(cls, v):
        if v is not None and len(v) > 20:
            raise ValueError('Maximum 20 tags allowed')
        return v


class DocumentStatsResponse(BaseModel):
    """Schema for document statistics."""
    total_documents: int
    total_chunks: int
    total_size_bytes: int
    documents_by_status: Dict[str, int]
    documents_by_category: Dict[str, int]
    processing_queue_size: int
    average_processing_time_minutes: Optional[float] = None
    
    class Config:
        from_attributes = True
