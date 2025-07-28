# Services module for RAG system MVP
# This module contains only essential services for the MVP

__all__ = []

# Input validation (critical for security)
try:
    from .validation import InputValidator, ValidationResult, ValidationError, SanitizationLevel
    __all__.extend(['InputValidator', 'ValidationResult', 'ValidationError', 'SanitizationLevel'])
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    InputValidator = ValidationResult = ValidationError = SanitizationLevel = None

# Document management (essential for document handling)
try:
    from .document_manager import DocumentManager, DocumentMetadata, DocumentStatus
    __all__.extend(['DocumentManager', 'DocumentMetadata', 'DocumentStatus'])
    DOCUMENT_MANAGER_AVAILABLE = True
except ImportError:
    DOCUMENT_MANAGER_AVAILABLE = False
    DocumentManager = DocumentMetadata = DocumentStatus = None

# Feature availability flags
FEATURES = {
    'validation': VALIDATION_AVAILABLE,
    'document_manager': DOCUMENT_MANAGER_AVAILABLE,
    
    # Archived features (marked as unavailable)
    'auth': False,
    'async_processing': False,
    'vector_search': False,
    'llm_manager': False,
    'hybrid_storage': False,
    'persistent_storage': False,
    'memory_safe_storage': False,
    'vector_store_db': False,
}

def get_available_features():
    """Get list of available features"""
    return [feature for feature, available in FEATURES.items() if available]

def is_feature_available(feature_name: str) -> bool:
    """Check if a specific feature is available"""
    return FEATURES.get(feature_name, False)