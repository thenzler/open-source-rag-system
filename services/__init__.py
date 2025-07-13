# Services module for RAG system optimization
# This module contains optimized components for the RAG system

# Build __all__ list dynamically based on available modules
__all__ = []

# Vector search (may fail if FAISS not installed)
try:
    from .vector_search import FAISSVectorSearch, OptimizedVectorStore
    __all__.extend(['FAISSVectorSearch', 'OptimizedVectorStore'])
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    FAISSVectorSearch = OptimizedVectorStore = None

# Async processing (independent module)
try:
    from .async_processor import AsyncDocumentProcessor, ProcessingStatus, ProcessingJob
    __all__.extend(['AsyncDocumentProcessor', 'ProcessingStatus', 'ProcessingJob'])
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    AsyncDocumentProcessor = ProcessingStatus = ProcessingJob = None

# Authentication (independent module)
try:
    from .auth import AuthManager, User, UserRole, JWTManager
    __all__.extend(['AuthManager', 'User', 'UserRole', 'JWTManager'])
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    AuthManager = User = UserRole = JWTManager = None

# Input validation (independent module)
try:
    from .validation import InputValidator, ValidationResult, ValidationError, SanitizationLevel
    __all__.extend(['InputValidator', 'ValidationResult', 'ValidationError', 'SanitizationLevel'])
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    InputValidator = ValidationResult = ValidationError = SanitizationLevel = None

# Document management (independent module)
try:
    from .document_manager import DocumentManager, DocumentMetadata, DocumentChunk, DocumentStatus
    __all__.extend(['DocumentManager', 'DocumentMetadata', 'DocumentChunk', 'DocumentStatus'])
    DOCUMENT_MANAGER_AVAILABLE = True
except ImportError:
    DOCUMENT_MANAGER_AVAILABLE = False
    DocumentManager = DocumentMetadata = DocumentChunk = DocumentStatus = None