"""
Custom exceptions for the RAG System.
"""


class RAGSystemError(Exception):
    """Base exception for RAG System."""
    pass


class DocumentNotFoundError(RAGSystemError):
    """Raised when a document is not found."""
    pass


class ProcessingError(RAGSystemError):
    """Raised when document processing fails."""
    pass


class ValidationError(RAGSystemError):
    """Raised when validation fails."""
    pass


class VectorStoreError(RAGSystemError):
    """Raised when vector store operations fail."""
    pass


class LLMServiceError(RAGSystemError):
    """Raised when LLM service calls fail."""
    pass


class AuthenticationError(RAGSystemError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(RAGSystemError):
    """Raised when authorization fails."""
    pass
