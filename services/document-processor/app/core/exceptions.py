"""
Custom exceptions for Document Processor Service.
"""
from typing import Optional, Any


class DocumentProcessorException(Exception):
    """Base exception for document processor."""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details


class ProcessingError(DocumentProcessorException):
    """Exception raised when document processing fails."""
    pass


class UnsupportedFormatError(DocumentProcessorException):
    """Exception raised when file format is not supported."""
    pass


class DocumentNotFoundError(DocumentProcessorException):
    """Exception raised when document is not found."""
    pass


class StorageError(DocumentProcessorException):
    """Exception raised when storage operations fail."""
    pass


class VectorServiceError(DocumentProcessorException):
    """Exception raised when vector service operations fail."""
    pass


class EmbeddingError(DocumentProcessorException):
    """Exception raised when embedding generation fails."""
    pass


class OCRError(DocumentProcessorException):
    """Exception raised when OCR operations fail."""
    pass


class ValidationError(DocumentProcessorException):
    """Exception raised when validation fails."""
    pass


class QuotaExceededError(DocumentProcessorException):
    """Exception raised when quota is exceeded."""
    pass