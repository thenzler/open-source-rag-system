"""
Database Models for RAG System
SQLAlchemy models for document management, user data, and system tracking.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey, Index, LargeBinary
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func

Base = declarative_base()


class ProcessingStatus(Enum):
    """Document processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UserRole(Enum):
    """User role enumeration."""
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class User(Base):
    """User account model."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Roles and permissions
    roles = Column(ARRAY(String), default=["user"], nullable=False)
    permissions = Column(JSONB, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    email_verified_at = Column(DateTime(timezone=True))
    
    # User preferences
    preferences = Column(JSONB, default={})
    
    # API usage tracking
    api_quota_limit = Column(Integer, default=1000)  # Requests per day
    api_quota_used = Column(Integer, default=0)
    api_quota_reset_date = Column(DateTime(timezone=True), default=func.now())
    
    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    query_logs = relationship("QueryLog", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"


class Document(Base):
    """Document metadata model."""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)  # Store original name separately
    file_path = Column(Text, nullable=False)
    
    # File information
    mime_type = Column(String(100))
    file_size = Column(Integer)  # Size in bytes
    checksum = Column(String(64))  # SHA-256 hash
    
    # Processing status
    status = Column(String(50), default=ProcessingStatus.PENDING.value, nullable=False, index=True)
    processing_progress = Column(Integer, default=0)  # 0-100
    processing_message = Column(Text)
    processing_started_at = Column(DateTime(timezone=True))
    processing_completed_at = Column(DateTime(timezone=True))
    processing_error = Column(Text)
    
    # Timestamps
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # User and organization
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"), index=True)
    
    # Document metadata and tags
    metadata = Column(JSONB, default={})
    tags = Column(ARRAY(String), default=[])
    category = Column(String(100), index=True)
    language = Column(String(10), default="en")
    
    # Content statistics
    total_pages = Column(Integer)
    total_chunks = Column(Integer, default=0)
    total_characters = Column(Integer, default=0)
    total_words = Column(Integer, default=0)
    
    # Privacy and access control
    is_public = Column(Boolean, default=False)
    access_level = Column(String(50), default="private")  # private, organization, public
    encryption_key_id = Column(String(255))  # Reference to encryption key
    
    # Content analysis
    content_hash = Column(String(64))  # Hash of processed content for deduplication
    similarity_threshold = Column(Float, default=0.8)  # For near-duplicate detection
    
    # Relationships
    user = relationship("User", back_populates="documents")
    organization = relationship("Organization", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    query_results = relationship("QueryResult", back_populates="document")
    
    # Indexes
    __table_args__ = (
        Index('idx_documents_user_status', 'user_id', 'status'),
        Index('idx_documents_category_status', 'category', 'status'),
        Index('idx_documents_uploaded_at', 'uploaded_at'),
        Index('idx_documents_checksum', 'checksum'),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"


class DocumentChunk(Base):
    """Document chunk model for storing processed text segments."""
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Chunk information
    chunk_index = Column(Integer, nullable=False)  # Order within document
    content = Column(Text, nullable=False)
    content_hash = Column(String(64))  # For deduplication
    
    # Position information
    start_char = Column(Integer)  # Character position in original document
    end_char = Column(Integer)
    page_number = Column(Integer)  # For PDFs
    section_title = Column(String(500))  # Section/chapter title if available
    
    # Chunk statistics
    char_count = Column(Integer, nullable=False)
    word_count = Column(Integer, nullable=False)
    sentence_count = Column(Integer)
    
    # Vector storage reference
    vector_id = Column(String(255), index=True)  # Reference to vector in Qdrant
    embedding_model = Column(String(255))  # Which model was used
    embedding_created_at = Column(DateTime(timezone=True))
    
    # Chunk metadata
    chunk_metadata = Column(JSONB, default={})
    chunk_type = Column(String(50), default="text")  # text, table, image_caption, etc.
    
    # Quality scoring
    quality_score = Column(Float)  # 0-1, content quality assessment
    relevance_score = Column(Float)  # 0-1, relevance to document topic
    
    # Processing information
    processing_method = Column(String(100))  # Which chunking method was used
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    query_results = relationship("QueryResult", back_populates="chunk")
    
    # Indexes
    __table_args__ = (
        Index('idx_chunks_document_index', 'document_id', 'chunk_index'),
        Index('idx_chunks_vector_id', 'vector_id'),
        Index('idx_chunks_page_number', 'page_number'),
        Index('idx_chunks_content_hash', 'content_hash'),
    )
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index})>"


class QueryLog(Base):
    """Query execution log model."""
    __tablename__ = "query_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    
    # Query information
    query_text = Column(Text, nullable=False)
    query_hash = Column(String(64), index=True)  # For caching and analytics
    query_type = Column(String(50), default="semantic")  # semantic, keyword, hybrid
    
    # Query parameters
    top_k = Column(Integer, default=5)
    min_score = Column(Float, default=0.0)
    filters = Column(JSONB, default={})
    
    # Results
    response_text = Column(Text)
    source_documents = Column(JSONB, default=[])  # List of source document IDs and chunks
    result_count = Column(Integer, default=0)
    confidence_score = Column(Float)  # Overall confidence in response
    
    # Performance metrics
    processing_time_ms = Column(Integer)
    embedding_time_ms = Column(Integer)
    search_time_ms = Column(Integer)
    llm_time_ms = Column(Integer)
    
    # Context
    session_id = Column(String(255), index=True)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    
    # Feedback and evaluation
    user_rating = Column(Integer)  # 1-5 star rating
    user_feedback = Column(Text)
    was_helpful = Column(Boolean)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="query_logs")
    query_results = relationship("QueryResult", back_populates="query_log", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_query_logs_user_created', 'user_id', 'created_at'),
        Index('idx_query_logs_query_hash', 'query_hash'),
        Index('idx_query_logs_session', 'session_id'),
        Index('idx_query_logs_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<QueryLog(id={self.id}, user_id={self.user_id}, query_text={self.query_text[:50]}...)>"


class QueryResult(Base):
    """Individual query result model linking queries to document chunks."""
    __tablename__ = "query_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_log_id = Column(UUID(as_uuid=True), ForeignKey("query_logs.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Relevance scoring
    relevance_score = Column(Float, nullable=False)  # Vector similarity score
    rerank_score = Column(Float)  # Re-ranking score if applied
    final_score = Column(Float, nullable=False)  # Final combined score
    
    # Position in results
    rank = Column(Integer, nullable=False)  # 1-based ranking in result set
    
    # Used in response
    used_in_response = Column(Boolean, default=False)
    
    # Relationships
    query_log = relationship("QueryLog", back_populates="query_results")
    document = relationship("Document", back_populates="query_results")
    chunk = relationship("DocumentChunk", back_populates="query_results")
    
    # Indexes
    __table_args__ = (
        Index('idx_query_results_query_log', 'query_log_id'),
        Index('idx_query_results_document', 'document_id'),
        Index('idx_query_results_chunk', 'chunk_id'),
        Index('idx_query_results_score', 'final_score'),
    )
    
    def __repr__(self):
        return f"<QueryResult(id={self.id}, query_log_id={self.query_log_id}, score={self.final_score})>"


class Organization(Base):
    """Organization/tenant model for multi-tenancy."""
    __tablename__ = "organizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    
    # Organization details
    description = Column(Text)
    website = Column(String(255))
    logo_url = Column(String(500))
    
    # Subscription and limits
    plan_type = Column(String(50), default="free")  # free, pro, enterprise
    document_limit = Column(Integer, default=100)
    storage_limit_gb = Column(Integer, default=1)
    query_limit_monthly = Column(Integer, default=1000)
    
    # Usage tracking
    document_count = Column(Integer, default=0)
    storage_used_bytes = Column(Integer, default=0)
    query_count_current_month = Column(Integer, default=0)
    
    # Settings
    settings = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    documents = relationship("Document", back_populates="organization")
    members = relationship("OrganizationMember", back_populates="organization", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Organization(id={self.id}, name={self.name}, plan={self.plan_type})>"


class OrganizationMember(Base):
    """Organization membership model."""
    __tablename__ = "organization_members"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Membership details
    role = Column(String(50), default="member")  # member, admin, owner
    permissions = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    joined_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    organization = relationship("Organization", back_populates="members")
    user = relationship("User")
    
    # Unique constraint
    __table_args__ = (
        Index('idx_org_members_unique', 'organization_id', 'user_id', unique=True),
        Index('idx_org_members_user', 'user_id'),
    )
    
    def __repr__(self):
        return f"<OrganizationMember(org_id={self.organization_id}, user_id={self.user_id}, role={self.role})>"


class AuditLog(Base):
    """Audit log model for security and compliance."""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event details
    event_type = Column(String(100), nullable=False, index=True)
    action = Column(String(100), nullable=False)
    result = Column(String(50), nullable=False)  # SUCCESS, FAILURE, ERROR
    
    # Actor information
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    username = Column(String(255))  # Stored separately in case user is deleted
    
    # Target resource
    resource_type = Column(String(100))  # document, user, organization, etc.
    resource_id = Column(UUID(as_uuid=True), index=True)
    resource_name = Column(String(255))
    
    # Request context
    ip_address = Column(String(45))
    user_agent = Column(Text)
    session_id = Column(String(255))
    request_id = Column(String(255))
    
    # Event details
    details = Column(JSONB, default={})
    error_message = Column(Text)
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_logs_event_type', 'event_type'),
        Index('idx_audit_logs_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_audit_logs_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_logs_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, event_type={self.event_type}, action={self.action})>"


class SystemMetric(Base):
    """System metrics and monitoring data."""
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Metric identification
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    
    # Metric value
    value = Column(Float, nullable=False)
    unit = Column(String(20))  # bytes, seconds, requests, etc.
    
    # Labels/dimensions
    labels = Column(JSONB, default={})
    
    # Service/component
    service_name = Column(String(100), index=True)
    instance_id = Column(String(255))
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_system_metrics_name_timestamp', 'metric_name', 'timestamp'),
        Index('idx_system_metrics_service_timestamp', 'service_name', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<SystemMetric(name={self.metric_name}, value={self.value}, timestamp={self.timestamp})>"


class ApiKey(Base):
    """API key management model."""
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Key details
    name = Column(String(255), nullable=False)  # User-friendly name
    key_hash = Column(String(255), nullable=False, unique=True)  # Hashed API key
    key_prefix = Column(String(20), nullable=False)  # First few chars for identification
    
    # Permissions and limits
    scopes = Column(ARRAY(String), default=[])  # List of permitted scopes
    rate_limit_per_minute = Column(Integer, default=60)
    rate_limit_per_hour = Column(Integer, default=1000)
    rate_limit_per_day = Column(Integer, default=10000)
    
    # Usage tracking
    last_used_at = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)
    
    # Status
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_api_keys_user', 'user_id'),
        Index('idx_api_keys_hash', 'key_hash'),
        Index('idx_api_keys_prefix', 'key_prefix'),
    )
    
    def __repr__(self):
        return f"<ApiKey(id={self.id}, name={self.name}, user_id={self.user_id})>"


class DocumentTemplate(Base):
    """Template for document processing configurations."""
    __tablename__ = "document_templates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Template configuration
    file_patterns = Column(ARRAY(String), default=[])  # File patterns this template applies to
    mime_types = Column(ARRAY(String), default=[])
    
    # Processing settings
    chunking_strategy = Column(String(100), default="recursive")
    chunk_size = Column(Integer, default=512)
    chunk_overlap = Column(Integer, default=50)
    
    # Extraction settings
    extraction_settings = Column(JSONB, default={})
    
    # Metadata extraction
    metadata_extractors = Column(ARRAY(String), default=[])
    
    # Quality filters
    min_chunk_length = Column(Integer, default=100)
    max_chunk_length = Column(Integer, default=2000)
    quality_threshold = Column(Float, default=0.5)
    
    # Template metadata
    is_default = Column(Boolean, default=False)
    is_system = Column(Boolean, default=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<DocumentTemplate(id={self.id}, name={self.name})>"


# Create all indexes after table definitions
def create_additional_indexes(engine):
    """Create additional database indexes for performance."""
    with engine.begin() as conn:
        # Full-text search indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_filename_fts 
            ON documents USING gin(to_tsvector('english', filename));
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_content_fts 
            ON document_chunks USING gin(to_tsvector('english', content));
        """)
        
        # Partial indexes for performance
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_active_processing 
            ON documents (user_id, status) 
            WHERE status IN ('pending', 'processing');
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_logs_recent 
            ON query_logs (user_id, created_at) 
            WHERE created_at > NOW() - INTERVAL '30 days';
        """)
        
        # Composite indexes for analytics
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_logs_analytics 
            ON query_logs (created_at, query_type, result_count);
        """)
