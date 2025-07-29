"""
Data models for repositories
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentStatus(Enum):
    """Document processing status"""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    PROCESSED = "processed"  # Legacy status from old system
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"

@dataclass
class Tenant:
    """Multi-tenant organization entity"""
    id: Optional[int] = None
    name: str = ""
    slug: str = ""  # URL-safe identifier
    domain: Optional[str] = None  # Custom domain for tenant
    is_active: bool = True
    created_at: Optional[datetime] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    limits: Dict[str, Any] = field(default_factory=dict)  # Storage, API limits etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "domain": self.domain,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "settings": self.settings,
            "limits": self.limits
        }

@dataclass
class Document:
    """Document entity with multi-tenancy support"""
    id: Optional[int] = None
    tenant_id: int = 1  # Default tenant for backward compatibility
    filename: str = ""
    original_filename: str = ""
    file_path: str = ""
    content_type: str = ""
    file_size: int = 0
    status: DocumentStatus = DocumentStatus.UPLOADING
    upload_timestamp: Optional[datetime] = None
    processing_timestamp: Optional[datetime] = None
    completion_timestamp: Optional[datetime] = None
    uploader: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing results
    text_content: Optional[str] = None
    chunk_count: int = 0
    embedding_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_path": self.file_path,
            "content_type": self.content_type,
            "file_size": self.file_size,
            "status": self.status.value if self.status else None,
            "upload_timestamp": self.upload_timestamp.isoformat() if self.upload_timestamp else None,
            "processing_timestamp": self.processing_timestamp.isoformat() if self.processing_timestamp else None,
            "completion_timestamp": self.completion_timestamp.isoformat() if self.completion_timestamp else None,
            "uploader": self.uploader,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata,
            "chunk_count": self.chunk_count,
            "embedding_count": self.embedding_count
        }

@dataclass
class DocumentChunk:
    """Text chunk entity"""
    id: Optional[int] = None
    document_id: int = 0
    chunk_index: int = 0
    text_content: str = ""
    character_count: int = 0
    word_count: int = 0
    start_char: int = 0
    end_char: int = 0
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text_content": self.text_content,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

@dataclass
class Embedding:
    """Vector embedding entity"""
    id: Optional[int] = None
    chunk_id: int = 0
    document_id: int = 0
    embedding_vector: List[float] = field(default_factory=list)
    embedding_model: str = ""
    vector_dimension: int = 0
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "embedding_model": self.embedding_model,
            "vector_dimension": self.vector_dimension,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

@dataclass
class SearchResult:
    """Search result entity"""
    chunk: DocumentChunk
    document: Document
    similarity_score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chunk": self.chunk.to_dict(),
            "document": self.document.to_dict(),
            "similarity_score": self.similarity_score,
            "rank": self.rank,
            "metadata": self.metadata
        }

@dataclass
class User:
    """User entity with multi-tenancy support"""
    id: Optional[int] = None
    tenant_id: int = 1  # Default tenant for backward compatibility
    username: str = ""
    email: str = ""
    password_hash: str = ""
    role: str = "user"
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding password)"""
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "metadata": self.metadata
        }

@dataclass
class QueryLog:
    """Query history entity"""
    id: Optional[int] = None
    query_text: str = ""
    user_id: Optional[int] = None
    result_count: int = 0
    processing_time: float = 0.0
    method: str = ""  # "vector_search", "llm_generated", etc.
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "query_text": self.query_text,
            "user_id": self.user_id,
            "result_count": self.result_count,
            "processing_time": self.processing_time,
            "method": self.method,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata
        }