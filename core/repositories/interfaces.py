"""
Repository interfaces for all data entities
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseRepository, QueryResult, SearchOptions
from .models import Document, DocumentChunk, Embedding, QueryLog, User


class IDocumentRepository(BaseRepository[Document, int]):
    """Document repository interface"""

    @abstractmethod
    async def get_by_filename(self, filename: str) -> Optional[Document]:
        """Get document by filename"""
        pass

    @abstractmethod
    async def get_by_hash(self, file_hash: str) -> Optional[Document]:
        """Get document by file hash for deduplication"""
        pass

    @abstractmethod
    async def update_status(self, document_id: int, status: str) -> bool:
        """Update document processing status"""
        pass

    @abstractmethod
    async def get_by_uploader(
        self, uploader: str, options: Optional[SearchOptions] = None
    ) -> QueryResult[Document]:
        """Get documents by uploader"""
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get document statistics"""
        pass


class IChunkRepository(BaseRepository[DocumentChunk, int]):
    """Document chunk repository interface"""

    @abstractmethod
    async def get_by_document_id(
        self, document_id: int, options: Optional[SearchOptions] = None
    ) -> QueryResult[DocumentChunk]:
        """Get all chunks for a document"""
        pass

    @abstractmethod
    async def delete_by_document_id(self, document_id: int) -> int:
        """Delete all chunks for a document, return count deleted"""
        pass

    @abstractmethod
    async def search_text(
        self, query: str, options: Optional[SearchOptions] = None
    ) -> QueryResult[DocumentChunk]:
        """Full-text search in chunks"""
        pass

    @abstractmethod
    async def get_chunk_statistics(
        self, document_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get chunk statistics"""
        pass


class IEmbeddingRepository(BaseRepository[Embedding, int]):
    """Embedding repository interface"""

    @abstractmethod
    async def get_by_chunk_id(self, chunk_id: int) -> Optional[Embedding]:
        """Get embedding for a specific chunk"""
        pass

    @abstractmethod
    async def get_by_document_id(self, document_id: int) -> List[Embedding]:
        """Get all embeddings for a document"""
        pass

    @abstractmethod
    async def delete_by_document_id(self, document_id: int) -> int:
        """Delete all embeddings for a document"""
        pass

    @abstractmethod
    async def delete_by_chunk_id(self, chunk_id: int) -> bool:
        """Delete embedding for a specific chunk"""
        pass

    @abstractmethod
    async def bulk_create(self, embeddings: List[Embedding]) -> List[Embedding]:
        """Create multiple embeddings efficiently"""
        pass

    @abstractmethod
    async def update_model_version(self, old_model: str, new_model: str) -> int:
        """Update embedding model version, return count updated"""
        pass


class IVectorSearchRepository(ABC):
    """Vector search repository interface"""

    @abstractmethod
    async def build_index(self, embeddings: List[Embedding]) -> bool:
        """Build or rebuild the search index"""
        pass

    @abstractmethod
    async def add_to_index(self, embeddings: List[Embedding]) -> bool:
        """Add new embeddings to the index"""
        pass

    @abstractmethod
    async def remove_from_index(self, embedding_ids: List[int]) -> bool:
        """Remove embeddings from the index"""
        pass

    @abstractmethod
    async def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[int, float]]:  # [(embedding_id, similarity_score), ...]
        """Find similar embeddings by vector"""
        pass

    @abstractmethod
    async def search_similar_text(
        self, query: str, limit: int = 10, threshold: float = 0.7
    ) -> "QueryResult":
        """Find similar documents by text query (convenience method)"""
        pass

    @abstractmethod
    async def get_index_statistics(self) -> Dict[str, Any]:
        """Get search index statistics"""
        pass

    @abstractmethod
    async def is_index_ready(self) -> bool:
        """Check if search index is ready"""
        pass


class IUserRepository(BaseRepository[User, int]):
    """User repository interface"""

    @abstractmethod
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        pass

    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        pass

    @abstractmethod
    async def update_last_login(self, user_id: int) -> bool:
        """Update user's last login timestamp"""
        pass

    @abstractmethod
    async def deactivate_user(self, user_id: int) -> bool:
        """Deactivate a user account"""
        pass

    @abstractmethod
    async def get_active_users(
        self, options: Optional[SearchOptions] = None
    ) -> QueryResult[User]:
        """Get all active users"""
        pass


class IQueryHistoryRepository(BaseRepository[QueryLog, int]):
    """Query history repository interface"""

    @abstractmethod
    async def get_by_user_id(
        self, user_id: int, options: Optional[SearchOptions] = None
    ) -> QueryResult[QueryLog]:
        """Get query history for a user"""
        pass

    @abstractmethod
    async def get_popular_queries(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most popular queries with counts"""
        pass

    @abstractmethod
    async def get_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get query analytics for the last N days"""
        pass

    @abstractmethod
    async def cleanup_old_logs(self, days_to_keep: int = 90) -> int:
        """Clean up old query logs, return count deleted"""
        pass


class ICacheRepository(ABC):
    """Cache repository interface"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        pass

    @abstractmethod
    async def set(
        self, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set cached value with optional TTL"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cached value"""
        pass

    @abstractmethod
    async def clear_all(self) -> bool:
        """Clear all cached values"""
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Clean up expired cache entries"""
        pass


# Aggregate interface for complete RAG system
class IRAGRepository(ABC):
    """Aggregate repository interface for the complete RAG system"""

    @property
    @abstractmethod
    def documents(self) -> IDocumentRepository:
        """Document repository"""
        pass

    @property
    @abstractmethod
    def chunks(self) -> IChunkRepository:
        """Chunk repository"""
        pass

    @property
    @abstractmethod
    def embeddings(self) -> IEmbeddingRepository:
        """Embedding repository"""
        pass

    @property
    @abstractmethod
    def vector_search(self) -> IVectorSearchRepository:
        """Vector search repository"""
        pass

    @property
    @abstractmethod
    def users(self) -> IUserRepository:
        """User repository"""
        pass

    @property
    @abstractmethod
    def query_history(self) -> IQueryHistoryRepository:
        """Query history repository"""
        pass

    @property
    @abstractmethod
    def cache(self) -> ICacheRepository:
        """Cache repository"""
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize all repositories"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all repositories"""
        pass
