"""
Repository Factory for Production RAG System
Creates and manages all repository instances
"""

import logging
from typing import Dict, Optional

from .audit_repository import SwissAuditRepository
from .interfaces import (IDocumentRepository, IRAGRepository,
                         IVectorSearchRepository)
from .sqlite_repository import SQLiteDocumentRepository
from .vector_repository import ProductionVectorRepository

try:
    from .postgresql_repository import PostgreSQLRepository

    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
try:
    from config.config import config
except ImportError:
    config = None

logger = logging.getLogger(__name__)


class ProductionRAGRepository(IRAGRepository):
    """Production implementation of the complete RAG repository"""

    def __init__(
        self,
        db_path: Optional[str] = None,
        audit_db_path: Optional[str] = None,
        vector_cache_size: int = 1000,
    ):
        # Use config paths if available
        if not db_path and config:
            db_path = str(config.BASE_DIR / "data" / "rag_database.db")
        elif not db_path:
            db_path = "data/rag_database.db"

        if not audit_db_path and config:
            audit_db_path = str(config.BASE_DIR / "data" / "audit.db")
        elif not audit_db_path:
            audit_db_path = "data/audit.db"

        # Initialize repositories
        self._documents = SQLiteDocumentRepository(db_path)
        self._vector_search = ProductionVectorRepository(cache_size=vector_cache_size)
        self._audit = SwissAuditRepository(audit_db_path)

        # Initialize chunks and embeddings from the same SQLite connection
        # (These would be implemented similarly to DocumentRepository)
        self._chunks = None  # TODO: SQLiteChunkRepository(db_path)
        self._embeddings = None  # TODO: SQLiteEmbeddingRepository(db_path)
        self._users = None  # TODO: SQLiteUserRepository(db_path) if needed
        self._query_history = (
            None  # TODO: SQLiteQueryHistoryRepository(db_path) if needed
        )
        self._cache = None  # TODO: ProductionCacheRepository() if needed

        self._initialized = False
        logger.info("Initialized Production RAG Repository")

    @property
    def documents(self) -> IDocumentRepository:
        """Document repository"""
        return self._documents

    @property
    def chunks(self):
        """Chunk repository"""
        if not self._chunks:
            logger.warning("Chunk repository not implemented yet")
        return self._chunks

    @property
    def embeddings(self):
        """Embedding repository"""
        if not self._embeddings:
            logger.warning("Embedding repository not implemented yet")
        return self._embeddings

    @property
    def vector_search(self) -> IVectorSearchRepository:
        """Vector search repository"""
        return self._vector_search

    @property
    def users(self):
        """User repository"""
        if not self._users:
            logger.warning("User repository not implemented yet")
        return self._users

    @property
    def query_history(self):
        """Query history repository"""
        if not self._query_history:
            logger.warning("Query history repository not implemented yet")
        return self._query_history

    @property
    def cache(self):
        """Cache repository"""
        if not self._cache:
            logger.warning("Cache repository not implemented yet")
        return self._cache

    @property
    def audit(self) -> SwissAuditRepository:
        """Swiss audit repository"""
        return self._audit

    async def initialize(self) -> bool:
        """Initialize all repositories"""
        try:
            # Document repository is initialized on creation (SQLite)
            # Audit repository is initialized on creation (SQLite)

            # Ensure vector search repository is fully loaded
            await self._vector_search._ensure_embeddings_loaded()

            # TODO: Initialize other repositories when implemented

            self._initialized = True
            logger.info("All repositories initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize repositories: {e}")
            return False

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all repositories"""
        health = {}

        try:
            # Check document repository
            count = await self.documents.count()
            health["documents"] = True
            logger.debug(f"Document repository health check passed ({count} documents)")
        except Exception as e:
            health["documents"] = False
            logger.error(f"Document repository health check failed: {e}")

        try:
            # Check vector search repository
            stats = await self.vector_search.get_index_statistics()
            health["vector_search"] = "total_vectors" in stats
            logger.debug(
                f"Vector search repository health check passed ({stats.get('total_vectors', 0)} vectors)"
            )
        except Exception as e:
            health["vector_search"] = False
            logger.error(f"Vector search repository health check failed: {e}")

        try:
            # Check audit repository (simple connection test)
            await self.audit.get_compliance_report(days_back=1)
            health["audit"] = True
            logger.debug("Audit repository health check passed")
        except Exception as e:
            health["audit"] = False
            logger.error(f"Audit repository health check failed: {e}")

        # TODO: Add health checks for other repositories when implemented

        overall_health = all(health.values())
        logger.info(
            f"Repository health check completed: {health} (Overall: {overall_health})"
        )

        return health

    def is_initialized(self) -> bool:
        """Check if repositories are initialized"""
        return self._initialized


class RepositoryFactory:
    """Factory for creating repository instances"""

    _instance: Optional[ProductionRAGRepository] = None

    @classmethod
    def create_production_repository(
        cls,
        db_path: Optional[str] = None,
        audit_db_path: Optional[str] = None,
        vector_cache_size: int = 1000,
        force_new: bool = False,
        use_postgresql: bool = False,
        postgres_url: Optional[str] = None,
    ) -> ProductionRAGRepository:
        """Create or get singleton production repository instance"""

        if cls._instance is None or force_new:
            cls._instance = ProductionRAGRepository(
                db_path=db_path,
                audit_db_path=audit_db_path,
                vector_cache_size=vector_cache_size,
            )
            logger.info("Created new production repository instance")

        return cls._instance

    @classmethod
    def get_instance(cls) -> Optional[ProductionRAGRepository]:
        """Get current repository instance"""
        return cls._instance

    @classmethod
    async def initialize_production_repositories(cls) -> bool:
        """Initialize production repositories"""
        if cls._instance is None:
            cls._instance = cls.create_production_repository()

        return await cls._instance.initialize()


# Convenience function for getting repository instance
def get_rag_repository() -> ProductionRAGRepository:
    """Get the current RAG repository instance"""
    instance = RepositoryFactory.get_instance()
    if instance is None:
        # Create default instance
        instance = RepositoryFactory.create_production_repository()
        logger.info("Created default repository instance")

    return instance


# Dependency injection helpers for FastAPI
def get_document_repository() -> IDocumentRepository:
    """FastAPI dependency for document repository"""
    return get_rag_repository().documents


def get_vector_search_repository() -> IVectorSearchRepository:
    """FastAPI dependency for vector search repository"""
    return get_rag_repository().vector_search


def get_audit_repository() -> SwissAuditRepository:
    """FastAPI dependency for audit repository"""
    return get_rag_repository().audit
