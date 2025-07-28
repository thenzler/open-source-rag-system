#!/usr/bin/env python3
"""
Database configuration with fallback to in-memory storage
Ensures system works with or without database
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class StorageConfig:
    """Storage configuration with intelligent fallback"""
    
    def __init__(self):
        # Try to get database URL from environment
        self.database_url = os.getenv('DATABASE_URL', '')
        self.use_database = bool(self.database_url)
        
        # Storage mode
        self.storage_mode = 'database' if self.use_database else 'memory'
        
        # Database settings
        self.db_pool_size = int(os.getenv('DB_POOL_SIZE', '20'))
        self.db_max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '40'))
        self.db_pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', '30'))
        
        # Vector settings
        self.vector_dimension = 384  # all-MiniLM-L6-v2
        self.max_chunk_size = 2000
        self.chunk_overlap = 300
        
        # Cache settings
        self.enable_cache = True
        self.cache_ttl_seconds = 3600  # 1 hour
        self.max_cache_size = 1000
        
        # Memory limits (for fallback mode)
        self.max_memory_documents = 1000
        self.max_memory_chunks = 10000
        self.memory_warning_threshold = 0.8  # Warn at 80% capacity
        
        logger.info(f"Storage mode: {self.storage_mode}")
        if self.use_database:
            logger.info(f"Database URL configured: {self._mask_db_url()}")
        else:
            logger.warning("No DATABASE_URL found - using in-memory storage (limited capacity)")
    
    def _mask_db_url(self) -> str:
        """Mask sensitive parts of database URL for logging"""
        if not self.database_url:
            return "Not configured"
        
        # Parse and mask password
        if '@' in self.database_url:
            parts = self.database_url.split('@')
            if '://' in parts[0]:
                protocol, creds = parts[0].split('://')
                if ':' in creds:
                    user = creds.split(':')[0]
                    return f"{protocol}://{user}:****@{parts[1]}"
        
        return "Configured"
    
    def is_database_available(self) -> bool:
        """Check if database is actually available"""
        if not self.use_database:
            return False
        
        try:
            # Try to import database dependencies
            import sqlalchemy
            import pgvector
            import psycopg2
            
            # Try to create engine (doesn't connect yet)
            from sqlalchemy import create_engine
            engine = create_engine(self.database_url, connect_args={"connect_timeout": 5})
            
            # Try to connect
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            
            return True
        except Exception as e:
            logger.warning(f"Database not available: {e}")
            return False
    
    def get_connection_params(self) -> dict:
        """Get database connection parameters"""
        return {
            'pool_size': self.db_pool_size,
            'max_overflow': self.db_max_overflow,
            'pool_timeout': self.db_pool_timeout,
            'pool_pre_ping': True,  # Check connections before use
            'echo': False  # Set to True for SQL debugging
        }

# Global configuration instance
storage_config = StorageConfig()

def get_storage_config() -> StorageConfig:
    """Get global storage configuration"""
    return storage_config