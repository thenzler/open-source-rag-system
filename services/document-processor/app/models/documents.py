"""
Document models for database.
"""
import enum
from datetime import datetime
from typing import Optional, Dict, Any
import uuid

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    Text, JSON, Enum, ForeignKey, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class ProcessingStatus(str, enum.Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Document(Base):
    """Document model."""
    __tablename__ = "documents"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # File information
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    mime_type = Column(String(128), nullable=False)
    file_size = Column(Integer, nullable=False)
    checksum = Column(String(64), nullable=False)  # SHA-256 hash
    
    # Processing information
    status = Column(
        Enum(ProcessingStatus), 
        default=ProcessingStatus.PENDING, 
        nullable=False,
        index=True
    )
    progress = Column(Integer, default=0)  # 0-100
    status_message = Column(Text, nullable=True)
    
    # Metadata
    user_id = Column(String(128), nullable=False, index=True)
    metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow, 
        nullable=False
    )
    processed_at = Column(DateTime, nullable=True)
    
    # Processing results
    total_chunks = Column(Integer, default=0)
    total_characters = Column(Integer, default=0)
    processing_time_ms = Column(Integer, nullable=True)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_user_status", "user_id", "status"),
        Index("idx_created_at", "created_at"),
        Index("idx_checksum", "checksum"),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"


class DocumentChunk(Base):
    """Document chunk model."""
    __tablename__ = "document_chunks"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to document
    document_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("documents.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Chunk information
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    char_count = Column(Integer, nullable=False)
    word_count = Column(Integer, nullable=False)
    
    # Vector information
    vector_id = Column(String(128), nullable=True)  # ID in vector database
    
    # Metadata
    chunk_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    # Indexes
    __table_args__ = (
        Index("idx_document_chunk", "document_id", "chunk_index"),
        Index("idx_vector_id", "vector_id"),
    )
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index})>"