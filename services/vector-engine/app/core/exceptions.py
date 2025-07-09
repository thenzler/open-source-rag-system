"""
Custom exceptions for Vector Engine Service.
"""
from typing import Optional, Any


class VectorEngineException(Exception):
    """Base exception for vector engine."""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details


class VectorSearchError(VectorEngineException):
    """Exception raised when vector search fails."""
    pass


class EmbeddingError(VectorEngineException):
    """Exception raised when embedding generation fails."""
    pass


class CollectionError(VectorEngineException):
    """Exception raised when collection operations fail."""
    pass


class StorageError(VectorEngineException):
    """Exception raised when storage operations fail."""
    pass


class ValidationError(VectorEngineException):
    """Exception raised when validation fails."""
    pass


class ConnectionError(VectorEngineException):
    """Exception raised when connection to Qdrant fails."""
    pass


class QuotaExceededError(VectorEngineException):
    """Exception raised when quota is exceeded."""
    pass


class ModelLoadError(VectorEngineException):
    """Exception raised when model loading fails."""
    pass