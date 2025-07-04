"""
Pydantic Schemas for API Request/Response Validation
Data models for FastAPI endpoints with validation and serialization.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, EmailStr, HttpUrl
from enum import Enum


# Base Classes
class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime
    updated_at: Optional[datetime] = None


class ConfigMixin:
    """Common Pydantic configuration."""
    orm_mode = True
    use_enum_values = True
    validate_assignment = True


# Enums
class ProcessingStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UserRoleEnum(str, Enum):
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class QueryTypeEnum(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


# User Schemas
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_-]+$')
    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=255)


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., min_length=8, max_length=128)
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v


class UserUpdate(BaseModel):
    full_name: Optional[str] = Field(None, max_length=255)
    email: Optional[EmailStr] = None
    preferences: Optional[Dict[str, Any]] = None


class UserResponse(UserBase, TimestampMixin):
    id: UUID
    is_active: bool
    is_verified: bool
    roles: List[str]
    last_login: Optional[datetime]
    email_verified_at: Optional[datetime]
    preferences: Dict[str, Any] = {}
    
    class Config(ConfigMixin):
        pass


class UserProfile(UserResponse):
    """Extended user profile with statistics."""
    document_count: int = 0
    query_count: int = 0
    storage_used_bytes: int = 0
    api_quota_used: int = 0
    api_quota_limit: int = 1000


# Authentication Schemas
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: UserResponse


class TokenPayload(BaseModel):
    user_id: UUID
    username: str
    roles: List[str]
    exp: int
    iat: int


# Document Schemas
class DocumentBase(BaseModel):
    filename: str = Field(..., max_length=255)
    category: Optional[str] = Field(None, max_length=100)
    tags: List[str] = Field(default_factory=list, max_items=20)
    is_public: bool = False
    access_level: str = Field(default="private", regex=r'^(private|organization|public)$')


class DocumentCreate(DocumentBase):
    metadata: Optional[Dict[str, Any]] = None


class DocumentUpdate(BaseModel):
    filename: Optional[str] = Field(None, max_length=255)
    category: Optional[str] = Field(None, max_length=100)
    tags: Optional[List[str]] = Field(None, max_items=20)
    is_public: Optional[bool] = None
    access_level: Optional[str] = Field(None, regex=r'^(private|organization|public)$')
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(DocumentBase, TimestampMixin):
    id: UUID
    original_filename: str
    file_path: str
    mime_type: Optional[str]
    file_size: Optional[int]
    checksum: Optional[str]
    status: ProcessingStatusEnum
    processing_progress: int = 0
    processing_message: Optional[str]
    processing_started_at: Optional[datetime]
    processing_completed_at: Optional[datetime]
    uploaded_at: datetime
    user_id: UUID
    organization_id: Optional[UUID]
    metadata: Dict[str, Any] = {}
    total_pages: Optional[int]
    total_chunks: int = 0
    total_characters: int = 0
    total_words: int = 0
    language: str = "en"
    
    class Config(ConfigMixin):
        pass


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    skip: int
    limit: int
    has_next: bool = False
    has_previous: bool = False


class DocumentStatusResponse(BaseModel):
    document_id: UUID
    status: ProcessingStatusEnum
    progress: int
    message: Optional[str]
    processed_chunks: int = 0
    processing_started: Optional[datetime]
    processing_completed: Optional[datetime]
    errors: List[str] = []


class DocumentChunkResponse(BaseModel):
    id: UUID
    chunk_index: int
    content: str
    char_count: int
    word_count: int
    page_number: Optional[int]
    section_title: Optional[str]
    chunk_metadata: Dict[str, Any] = {}
    quality_score: Optional[float]
    
    class Config(ConfigMixin):
        pass


# Query Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = None
    document_ids: Optional[List[UUID]] = Field(None, max_items=100)
    include_metadata: bool = True
    max_char_length: Optional[int] = Field(None, ge=100, le=5000)
    
    @validator('query')
    def validate_query(cls, v):
        # Basic sanitization
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
    
    @validator('filters')
    def validate_filters(cls, v):
        if v is None:
            return v
        
        # Limit filter complexity
        if len(v) > 10:
            raise ValueError('Too many filter conditions (max 10)')
        
        # Validate filter values
        for key, value in v.items():
            if len(str(key)) > 100:
                raise ValueError(f'Filter key too long: {key}')
            if isinstance(value, str) and len(value) > 200:
                raise ValueError(f'Filter value too long for key: {key}')
            elif isinstance(value, list) and len(value) > 50:
                raise ValueError(f'Too many filter values for key: {key}')
        
        return v


class AdvancedQueryRequest(QueryRequest):
    retrieval_strategy: str = Field(default="semantic", regex=r'^(semantic|keyword|hybrid)$')
    rerank: bool = False
    rerank_model: Optional[str] = Field(None, regex=r'^(cross-encoder|colbert)$')
    expand_query: bool = False
    semantic_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_boost: float = Field(default=0.3, ge=0.0, le=1.0)
    final_k: Optional[int] = Field(None, ge=1, le=50)


class SourceDocument(BaseModel):
    document_id: UUID
    filename: str
    chunk_id: UUID
    chunk_index: int
    relevance_score: float
    page_number: Optional[int]
    text_snippet: str
    start_char: Optional[int]
    end_char: Optional[int]
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[SourceDocument]
    total_sources: int
    confidence_score: Optional[float]
    processing_time_ms: int
    retrieval_strategy: Optional[str]
    
    # Advanced query fields
    expanded_query: Optional[str]
    reranking_applied: bool = False
    retrieval_metrics: Optional[Dict[str, Any]]


class SimilarDocumentResponse(BaseModel):
    source_document: Dict[str, Any]
    similar_documents: List[Dict[str, Any]]


# Analytics Schemas
class QueryAnalyticsRequest(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    granularity: str = Field(default="day", regex=r'^(hour|day|week|month)$')
    user_id: Optional[UUID] = None
    document_ids: Optional[List[UUID]] = None


class QueryMetrics(BaseModel):
    date: str
    query_count: int
    average_response_time_ms: float
    unique_users: int
    top_queries: List[str]
    success_rate: float


class QueryAnalyticsResponse(BaseModel):
    period: Dict[str, Any]
    metrics: List[QueryMetrics]
    summary: Dict[str, Any]


class DocumentStats(BaseModel):
    total: int
    processed: int
    processing: int
    failed: int
    total_size_bytes: int


class ChunkStats(BaseModel):
    total: int
    average_per_document: float
    average_length_chars: int


class QueryStats(BaseModel):
    total_today: int
    total_this_week: int
    average_response_time_ms: float
    most_common_topics: List[str]


class StorageStats(BaseModel):
    documents_size_bytes: int
    vector_index_size_bytes: int
    database_size_bytes: int
    available_space_bytes: int


class StatsResponse(BaseModel):
    documents: DocumentStats
    chunks: ChunkStats
    queries: QueryStats
    storage: StorageStats
    system_health: Dict[str, Any] = {}


# Organization Schemas
class OrganizationBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    website: Optional[HttpUrl] = None


class OrganizationCreate(OrganizationBase):
    slug: str = Field(..., min_length=3, max_length=100, regex=r'^[a-z0-9-]+$')


class OrganizationUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    website: Optional[HttpUrl] = None
    settings: Optional[Dict[str, Any]] = None


class OrganizationResponse(OrganizationBase, TimestampMixin):
    id: UUID
    slug: str
    plan_type: str
    document_limit: int
    storage_limit_gb: int
    query_limit_monthly: int
    document_count: int
    storage_used_bytes: int
    query_count_current_month: int
    is_active: bool
    settings: Dict[str, Any] = {}
    
    class Config(ConfigMixin):
        pass


class OrganizationMemberResponse(BaseModel):
    id: UUID
    organization_id: UUID
    user_id: UUID
    user: UserResponse
    role: str
    permissions: Dict[str, Any] = {}
    is_active: bool
    joined_at: datetime
    
    class Config(ConfigMixin):
        pass


# API Key Schemas
class ApiKeyBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    scopes: List[str] = Field(default_factory=list, max_items=20)
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10000)
    rate_limit_per_hour: int = Field(default=1000, ge=1, le=100000)
    rate_limit_per_day: int = Field(default=10000, ge=1, le=1000000)
    expires_at: Optional[datetime] = None


class ApiKeyCreate(ApiKeyBase):
    pass


class ApiKeyResponse(ApiKeyBase, TimestampMixin):
    id: UUID
    key_prefix: str
    last_used_at: Optional[datetime]
    usage_count: int
    is_active: bool
    
    class Config(ConfigMixin):
        pass


class ApiKeyCreateResponse(ApiKeyResponse):
    """Response when creating API key - includes the actual key."""
    api_key: str  # Only returned during creation


# System Schemas
class HealthCheck(BaseModel):
    service: str
    status: str
    response_time_ms: Optional[float]
    details: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str  # healthy, unhealthy, degraded
    timestamp: datetime
    services: Dict[str, HealthCheck] = {}
    version: Optional[str]
    uptime_seconds: Optional[int]


class ConfigurationResponse(BaseModel):
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    similarity_threshold: float
    max_query_length: int
    rate_limits: Dict[str, int]
    features: Dict[str, bool]
    supported_formats: List[str]


# Error Schemas
class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ValidationError(BaseModel):
    field: str
    message: str
    invalid_value: Any


class ValidationErrorResponse(BaseModel):
    error: str
    details: List[ValidationError]


# Batch Operation Schemas
class BatchDocumentUpload(BaseModel):
    files: List[str] = Field(..., min_items=1, max_items=100)
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class BatchOperationStatus(BaseModel):
    operation_id: UUID
    status: str
    total_items: int
    completed_items: int
    failed_items: int
    errors: List[str] = []
    started_at: datetime
    completed_at: Optional[datetime]


# Template Schemas
class DocumentTemplateBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    file_patterns: List[str] = Field(default_factory=list, max_items=20)
    mime_types: List[str] = Field(default_factory=list, max_items=10)


class DocumentTemplateCreate(DocumentTemplateBase):
    chunking_strategy: str = Field(default="recursive")
    chunk_size: int = Field(default=512, ge=100, le=4000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    extraction_settings: Dict[str, Any] = {}
    metadata_extractors: List[str] = Field(default_factory=list)
    min_chunk_length: int = Field(default=100, ge=50, le=1000)
    max_chunk_length: int = Field(default=2000, ge=500, le=10000)
    quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class DocumentTemplateResponse(DocumentTemplateBase, TimestampMixin):
    id: UUID
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    extraction_settings: Dict[str, Any]
    metadata_extractors: List[str]
    min_chunk_length: int
    max_chunk_length: int
    quality_threshold: float
    is_default: bool
    is_system: bool
    created_by: Optional[UUID]
    
    class Config(ConfigMixin):
        pass


# Websocket Schemas
class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DocumentProcessingUpdate(WebSocketMessage):
    type: str = "document_processing"
    data: DocumentStatusResponse


class QueryUpdate(WebSocketMessage):
    type: str = "query_result"
    data: QueryResponse


# Export all schemas
__all__ = [
    # User schemas
    "UserBase", "UserCreate", "UserUpdate", "UserResponse", "UserProfile",
    
    # Authentication schemas
    "LoginRequest", "LoginResponse", "TokenPayload",
    
    # Document schemas
    "DocumentBase", "DocumentCreate", "DocumentUpdate", "DocumentResponse",
    "DocumentListResponse", "DocumentStatusResponse", "DocumentChunkResponse",
    
    # Query schemas
    "QueryRequest", "AdvancedQueryRequest", "SourceDocument", "QueryResponse",
    "SimilarDocumentResponse",
    
    # Analytics schemas
    "QueryAnalyticsRequest", "QueryMetrics", "QueryAnalyticsResponse",
    "DocumentStats", "ChunkStats", "QueryStats", "StorageStats", "StatsResponse",
    
    # Organization schemas
    "OrganizationBase", "OrganizationCreate", "OrganizationUpdate", 
    "OrganizationResponse", "OrganizationMemberResponse",
    
    # API Key schemas
    "ApiKeyBase", "ApiKeyCreate", "ApiKeyResponse", "ApiKeyCreateResponse",
    
    # System schemas
    "HealthCheck", "HealthResponse", "ConfigurationResponse",
    
    # Error schemas
    "ErrorDetail", "ErrorResponse", "ValidationError", "ValidationErrorResponse",
    
    # Batch operation schemas
    "BatchDocumentUpload", "BatchOperationStatus",
    
    # Template schemas
    "DocumentTemplateBase", "DocumentTemplateCreate", "DocumentTemplateResponse",
    
    # WebSocket schemas
    "WebSocketMessage", "DocumentProcessingUpdate", "QueryUpdate",
    
    # Enums
    "ProcessingStatusEnum", "UserRoleEnum", "QueryTypeEnum"
]
