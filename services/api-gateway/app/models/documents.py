"""
Document and DocumentChunk models.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, String, Text, DateTime, Integer, Float, JSON, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class Document(Base):
    """Document model for storing uploaded documents."""
    
    __tablename__ = "documents"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=False)
    file_size = Column(Integer, nullable=False)
    
    # Content
    title = Column(String(500))
    content = Column(Text)
    summary = Column(Text)
    
    # Processing status
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    processing_progress = Column(Float, default=0.0)
    error_message = Column(Text)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    category = Column(String(100))
    tags = Column(JSON, default=list)
    language = Column(String(10), default="en")
    
    # User and timestamps
    user_id = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True))
    
    # File information
    file_path = Column(String(500))
    file_hash = Column(String(64))  # SHA-256 hash
    
    # Analytics
    view_count = Column(Integer, default=0)
    query_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True))
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_documents_user_id', 'user_id'),
        Index('idx_documents_status', 'status'),
        Index('idx_documents_category', 'category'),
        Index('idx_documents_created_at', 'created_at'),
        Index('idx_documents_file_hash', 'file_hash'),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": str(self.id),
            "filename": self.filename,
            "original_filename": self.original_filename,
            "content_type": self.content_type,
            "file_size": self.file_size,
            "title": self.title,
            "summary": self.summary,
            "status": self.status,
            "processing_progress": self.processing_progress,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "category": self.category,
            "tags": self.tags,
            "language": self.language,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "view_count": self.view_count,
            "query_count": self.query_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "chunk_count": len(self.chunks) if self.chunks else 0
        }


class DocumentChunk(Base):
    """DocumentChunk model for storing processed document chunks."""
    
    __tablename__ = "document_chunks"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Content
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_type = Column(String(50), default="text")  # text, heading, table, etc.
    
    # Position information
    start_position = Column(Integer)
    end_position = Column(Integer)
    page_number = Column(Integer)
    
    # Vector information
    vector_id = Column(String(100))  # ID in vector database
    embedding_model = Column(String(100))
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Analytics
    retrieval_count = Column(Integer, default=0)
    last_retrieved = Column(DateTime(timezone=True))
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    # Indexes
    __table_args__ = (
        Index('idx_chunks_document_id', 'document_id'),
        Index('idx_chunks_vector_id', 'vector_id'),
        Index('idx_chunks_chunk_index', 'document_id', 'chunk_index'),
        Index('idx_chunks_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "content": self.content,
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "page_number": self.page_number,
            "vector_id": self.vector_id,
            "embedding_model": self.embedding_model,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "retrieval_count": self.retrieval_count,
            "last_retrieved": self.last_retrieved.isoformat() if self.last_retrieved else None
        }
