"""
Document schemas for the RAG System API.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from datetime import datetime
import uuid


class DocumentResponse(BaseModel):
    """Document response schema."""
    id: str
    filename: str
    original_filename: str
    content_type: str
    file_size: int
    title: Optional[str] = None
    summary: Optional[str] = None
    status: str
    processing_progress: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}
    category: Optional[str] = None
    tags: List[str] = []
    language: str = "en"
    user_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    view_count: int = 0
    query_count: int = 0
    last_accessed: Optional[datetime] = None
    chunk_count: int = 0
    
    @classmethod
    def from_orm(cls, document):
        """Create from ORM object."""
        return cls(
            id=str(document.id),
            filename=document.filename,
            original_filename=document.original_filename,
            content_type=document.content_type,
            file_size=document.file_size,
            title=document.title,
            summary=document.summary,
            status=document.status,
            processing_progress=document.processing_progress or 0.0,
            error_message=document.error_message,
            metadata=document.metadata or {},
            category=document.category,
            tags=document.tags or [],
            language=document.language or "en",
            user_id=document.user_id,
            created_at=document.created_at,
            updated_at=document.updated_at,
            processed_at=document.processed_at,
            view_count=document.view_count or 0,
            query_count=document.query_count or 0,
            last_accessed=document.last_accessed,
            chunk_count=len(document.chunks) if hasattr(document, 'chunks') and document.chunks else 0
        )


class DocumentUploadRequest(BaseModel):
    """Document upload request schema."""
    metadata: Optional[Dict[str, Any]] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None


class DocumentListResponse(BaseModel):
    """Document list response schema."""
    documents: List[DocumentResponse]
    total: int
    skip: int
    limit: int


class DocumentChunkResponse(BaseModel):
    """Document chunk response schema."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    chunk_type: str = "text"
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    page_number: Optional[int] = None
    vector_id: Optional[str] = None
    embedding_model: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: Optional[datetime] = None
    retrieval_count: int = 0
    last_retrieved: Optional[datetime] = None


class DocumentStatusResponse(BaseModel):
    """Document processing status response schema."""
    document_id: str
    status: str
    progress: float
    error_message: Optional[str] = None
    created_at: datetime
    processed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None


class DocumentDeleteResponse(BaseModel):
    """Document deletion response schema."""
    message: str
    document_id: str
    deleted_chunks: int
    deleted_vectors: int
    file_deleted: bool = True
