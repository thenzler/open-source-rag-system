"""
Query logging and analytics models.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import Column, String, Text, DateTime, Integer, Float, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.core.database import Base


class QueryLog(Base):
    """Query log model for analytics and monitoring."""
    
    __tablename__ = "query_logs"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Query information
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50), default="semantic")  # semantic, keyword, advanced
    
    # User information
    user_id = Column(String(100), nullable=False)
    session_id = Column(String(100))
    
    # Request details
    top_k = Column(Integer)
    min_score = Column(Float)
    filters = Column(JSON, default=dict)
    
    # Response details
    result_count = Column(Integer, default=0)
    max_score = Column(Float)
    avg_score = Column(Float)
    response_time_ms = Column(Integer)
    
    # Model information
    embedding_model = Column(String(100))
    llm_model = Column(String(100))
    
    # Success/failure
    success = Column(String(10), default="true")  # true, false
    error_message = Column(Text)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Request info
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(500))
    
    # Indexes
    __table_args__ = (
        Index('idx_query_logs_user_id', 'user_id'),
        Index('idx_query_logs_created_at', 'created_at'),
        Index('idx_query_logs_query_type', 'query_type'),
        Index('idx_query_logs_success', 'success'),
        Index('idx_query_logs_session_id', 'session_id'),
    )
    
    def __repr__(self):
        return f"<QueryLog(id={self.id}, user_id={self.user_id}, query_type={self.query_type})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query log to dictionary."""
        return {
            "id": str(self.id),
            "query_text": self.query_text,
            "query_type": self.query_type,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "filters": self.filters,
            "result_count": self.result_count,
            "max_score": self.max_score,
            "avg_score": self.avg_score,
            "response_time_ms": self.response_time_ms,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent
        }


class UserSession(Base):
    """User session tracking for analytics."""
    
    __tablename__ = "user_sessions"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(100), unique=True, nullable=False)
    
    # User information
    user_id = Column(String(100), nullable=False)
    
    # Session details
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Activity tracking
    query_count = Column(Integer, default=0)
    document_uploads = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True))
    
    # Session metadata
    metadata = Column(JSON, default=dict)
    
    # Indexes
    __table_args__ = (
        Index('idx_user_sessions_session_id', 'session_id'),
        Index('idx_user_sessions_user_id', 'user_id'),
        Index('idx_user_sessions_created_at', 'created_at'),
        Index('idx_user_sessions_last_activity', 'last_activity'),
    )
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, session_id={self.session_id}, user_id={self.user_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user session to dictionary."""
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "query_count": self.query_count,
            "document_uploads": self.document_uploads,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "metadata": self.metadata
        }
