"""
Base repository interfaces and abstract classes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

# Generic type for entity IDs
EntityId = TypeVar("EntityId")
Entity = TypeVar("Entity")


@dataclass
class QueryResult(Generic[Entity]):
    """Generic query result wrapper"""

    items: List[Entity]
    total_count: int
    page: Optional[int] = None
    page_size: Optional[int] = None
    has_more: bool = False


@dataclass
class SearchOptions:
    """Search and filter options"""

    page: int = 1
    page_size: int = 20
    sort_by: Optional[str] = None
    sort_order: str = "asc"  # "asc" or "desc"
    filters: Optional[Dict[str, Any]] = None


class BaseRepository(ABC, Generic[Entity, EntityId]):
    """Abstract base repository with common CRUD operations"""

    @abstractmethod
    async def create(self, entity: Entity) -> Entity:
        """Create a new entity"""
        pass

    @abstractmethod
    async def get_by_id(self, entity_id: EntityId) -> Optional[Entity]:
        """Get entity by ID"""
        pass

    @abstractmethod
    async def update(
        self, entity_id: EntityId, updates: Dict[str, Any]
    ) -> Optional[Entity]:
        """Update entity by ID"""
        pass

    @abstractmethod
    async def delete(self, entity_id: EntityId) -> bool:
        """Delete entity by ID"""
        pass

    @abstractmethod
    async def list_all(
        self, options: Optional[SearchOptions] = None
    ) -> QueryResult[Entity]:
        """List all entities with optional pagination/filtering"""
        pass

    @abstractmethod
    async def exists(self, entity_id: EntityId) -> bool:
        """Check if entity exists"""
        pass

    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities with optional filters"""
        pass


class TransactionalRepository(ABC):
    """Repository with transaction support"""

    @abstractmethod
    async def begin_transaction(self):
        """Begin a transaction"""
        pass

    @abstractmethod
    async def commit_transaction(self):
        """Commit current transaction"""
        pass

    @abstractmethod
    async def rollback_transaction(self):
        """Rollback current transaction"""
        pass
