"""
Database Factory for Repository Selection
Handles switching between SQLite and PostgreSQL
"""
import logging
import os
from typing import Optional, Dict, Any
from pathlib import Path

from .interfaces import IDocumentRepository
from .sqlite_repository import SQLiteDocumentRepository

logger = logging.getLogger(__name__)

# Try to import PostgreSQL dependencies
try:
    from .postgresql_repository import PostgreSQLRepository
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
    logger.info("PostgreSQL dependencies not available, using SQLite only")

class DatabaseFactory:
    """Factory for creating database repository instances"""
    
    @staticmethod
    def create_document_repository(
        db_type: str = "sqlite",
        connection_string: Optional[str] = None,
        **kwargs
    ) -> IDocumentRepository:
        """Create document repository based on database type"""
        
        if db_type.lower() == "postgresql":
            if not POSTGRESQL_AVAILABLE:
                logger.warning("PostgreSQL requested but dependencies not available, falling back to SQLite")
                return DatabaseFactory._create_sqlite_repository(connection_string, **kwargs)
            
            if not connection_string:
                raise ValueError("PostgreSQL connection string is required")
            
            return DatabaseFactory._create_postgresql_repository(connection_string, **kwargs)
        
        elif db_type.lower() == "sqlite":
            return DatabaseFactory._create_sqlite_repository(connection_string, **kwargs)
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    @staticmethod
    def _create_sqlite_repository(db_path: Optional[str] = None, **kwargs) -> SQLiteDocumentRepository:
        """Create SQLite repository"""
        if not db_path:
            db_path = "data/rag_database.db"
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating SQLite repository: {db_path}")
        return SQLiteDocumentRepository(db_path)
    
    @staticmethod
    def _create_postgresql_repository(connection_string: str, **kwargs) -> 'PostgreSQLRepository':
        """Create PostgreSQL repository"""
        if not POSTGRESQL_AVAILABLE:
            raise RuntimeError("PostgreSQL dependencies not available")
        
        pool_size = kwargs.get('pool_size', 10)
        
        logger.info(f"Creating PostgreSQL repository (pool size: {pool_size})")
        return PostgreSQLRepository(connection_string, pool_size=pool_size)
    
    @staticmethod
    def from_config(config_obj) -> IDocumentRepository:
        """Create repository from configuration object"""
        try:
            # Check if PostgreSQL is configured
            postgres_url = getattr(config_obj, 'DATABASE_URL', None)
            use_postgres = getattr(config_obj, 'USE_POSTGRESQL', False)
            
            if use_postgres and postgres_url and POSTGRESQL_AVAILABLE:
                logger.info("Using PostgreSQL from configuration")
                return DatabaseFactory.create_document_repository(
                    db_type="postgresql",
                    connection_string=postgres_url,
                    pool_size=getattr(config_obj, 'DB_POOL_SIZE', 10)
                )
            
            # Fall back to SQLite
            sqlite_path = getattr(config_obj, 'DATABASE_PATH', 'data/rag_database.db')
            logger.info("Using SQLite from configuration")
            return DatabaseFactory.create_document_repository(
                db_type="sqlite",
                connection_string=sqlite_path
            )
            
        except Exception as e:
            logger.error(f"Failed to create repository from config: {e}")
            # Ultimate fallback
            return DatabaseFactory.create_document_repository(db_type="sqlite")
    
    @staticmethod
    def from_environment() -> IDocumentRepository:
        """Create repository from environment variables"""
        try:
            # Check environment variables
            postgres_url = os.getenv('DATABASE_URL')
            use_postgres = os.getenv('USE_POSTGRESQL', 'false').lower() == 'true'
            
            if use_postgres and postgres_url and POSTGRESQL_AVAILABLE:
                logger.info("Using PostgreSQL from environment")
                return DatabaseFactory.create_document_repository(
                    db_type="postgresql",
                    connection_string=postgres_url,
                    pool_size=int(os.getenv('DB_POOL_SIZE', '10'))
                )
            
            # Fall back to SQLite
            sqlite_path = os.getenv('DATABASE_PATH', 'data/rag_database.db')
            logger.info("Using SQLite from environment")
            return DatabaseFactory.create_document_repository(
                db_type="sqlite",
                connection_string=sqlite_path
            )
            
        except Exception as e:
            logger.error(f"Failed to create repository from environment: {e}")
            # Ultimate fallback
            return DatabaseFactory.create_document_repository(db_type="sqlite")

class DatabaseMigrator:
    """Handle database migrations between SQLite and PostgreSQL"""
    
    def __init__(self, source_repo: IDocumentRepository, target_repo: IDocumentRepository):
        self.source_repo = source_repo
        self.target_repo = target_repo
    
    async def migrate_documents(self, batch_size: int = 100) -> bool:
        """Migrate documents from source to target repository"""
        try:
            logger.info("Starting document migration...")
            
            # Initialize target repository if needed
            if hasattr(self.target_repo, 'initialize'):
                await self.target_repo.initialize()
            
            # Get all documents from source
            offset = 0
            total_migrated = 0
            
            while True:
                result = await self.source_repo.list_all(limit=batch_size, offset=offset)
                documents = result.items
                
                if not documents:
                    break
                
                # Migrate batch
                for doc in documents:
                    try:
                        # Remove ID to let target repository assign new one
                        doc.id = None
                        await self.target_repo.create(doc)
                        total_migrated += 1
                    except Exception as e:
                        logger.error(f"Failed to migrate document {doc.filename}: {e}")
                
                offset += batch_size
                logger.info(f"Migrated {total_migrated} documents so far...")
                
                if len(documents) < batch_size:
                    break
            
            logger.info(f"Migration completed. Total documents migrated: {total_migrated}")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

# Utility functions
def get_database_info(repo: IDocumentRepository) -> Dict[str, Any]:
    """Get database information"""
    if isinstance(repo, SQLiteDocumentRepository):
        return {
            "type": "SQLite",
            "path": repo.db_path,
            "supports_concurrent": False,
            "supports_transactions": True
        }
    elif POSTGRESQL_AVAILABLE and isinstance(repo, PostgreSQLRepository):
        return {
            "type": "PostgreSQL", 
            "connection_string": repo.connection_string,
            "pool_size": repo.pool_size,
            "supports_concurrent": True,
            "supports_transactions": True
        }
    else:
        return {
            "type": "Unknown",
            "supports_concurrent": False,
            "supports_transactions": False
        }

def recommend_database_type(expected_users: int, expected_documents: int) -> str:
    """Recommend database type based on expected usage"""
    if expected_users > 10 or expected_documents > 10000:
        if POSTGRESQL_AVAILABLE:
            return "postgresql"
        else:
            logger.warning("PostgreSQL recommended but not available, suggesting SQLite with warning")
            return "sqlite"
    else:
        return "sqlite"