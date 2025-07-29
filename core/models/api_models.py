"""
API Models for RAG System
Contains all Pydantic models used in the API
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# Document Models
class DocumentResponse(BaseModel):
    """Response model for document information"""

    id: int
    filename: str
    size: int
    content_type: str
    status: str


class DocumentUpdate(BaseModel):
    """Model for updating document metadata"""

    description: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[str] = None


class DocumentSearchResponse(BaseModel):
    """Response model for document search results"""

    query: str
    search_content: bool
    results: List[Dict[str, Any]]


class DocumentChunk(BaseModel):
    """Model for document chunk information"""

    document_id: int
    content: str
    similarity_score: float
    metadata: Optional[Dict[str, Any]] = {}
    total_found: int
    returned: int


# Query Models
class QueryRequest(BaseModel):
    """Request model for basic queries"""

    query: str
    top_k: Optional[int] = 5
    use_llm: Optional[bool] = None  # None = use default, True/False = override


class QueryResponse(BaseModel):
    """Response model for basic query results"""

    query: str
    results: List[dict]
    total_results: int
    message: Optional[str] = None


class LLMQueryResponse(BaseModel):
    """Response model for LLM-enhanced queries"""

    query: str
    answer: str
    method: str  # "llm_generated" or "vector_search"
    sources: List[dict]
    total_sources: int
    processing_time: Optional[float] = None


class SmartQueryRequest(BaseModel):
    """Request model for smart queries with advanced options"""

    query: str
    top_k: int = 5
    use_llm_fallback: bool = True
    strict_mode: bool = False  # If True, only return document-based answers


class SmartQueryResponse(BaseModel):
    """Response model for smart query results with detailed metadata"""

    query: str
    answer: str
    answer_type: str
    confidence: str
    confidence_score: float
    sources: List[Dict[str, Any]]
    reasoning: str
    chunk_count: int
    is_document_based: bool
    is_llm_generated: bool
    processing_time: float


# Chat Models
class ChatRequest(BaseModel):
    """Request model for chat interactions"""

    query: str
    chat_history: Optional[List[dict]] = []
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    context_limit: Optional[int] = 5


class ChatResponse(BaseModel):
    """Response model for chat interactions"""

    response: str
    query: Optional[str] = None
    context: Optional[List[dict]] = []
    confidence: Optional[float] = 0.0
    processing_time: Optional[float] = None


# Authentication Models
class LoginRequest(BaseModel):
    """Request model for user login"""

    username: str
    password: str


class LoginResponse(BaseModel):
    """Response model for successful login"""

    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: dict


class UserCreate(BaseModel):
    """Model for creating new users"""

    username: str
    email: str
    password: str
    role: Optional[str] = "user"


class UserResponse(BaseModel):
    """Response model for user information"""

    user_id: str
    username: str
    email: str
    role: str
    created_at: str
    last_login: Optional[str] = None
    is_active: bool
