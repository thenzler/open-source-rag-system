"""
Connection Pool Management
Handles database connection pooling for better performance
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import database dependencies
try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    import aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class PostgreSQLConnectionPool:
    """PostgreSQL connection pool manager"""

    def __init__(
        self,
        connection_string: str,
        min_size: int = 2,
        max_size: int = 10,
        command_timeout: int = 60,
        max_queries: int = 50000,
        max_inactive_connection_lifetime: int = 300,
    ):
        """Initialize PostgreSQL connection pool"""
        if not ASYNCPG_AVAILABLE:
            raise RuntimeError("asyncpg is required for PostgreSQL connection pooling")

        self.connection_string = connection_string
        self.min_size = min_size
        self.max_size = max_size
        self.command_timeout = command_timeout
        self.max_queries = max_queries
        self.max_inactive_connection_lifetime = max_inactive_connection_lifetime

        self.pool: Optional[asyncpg.Pool] = None
        self._stats = {
            "created_at": time.time(),
            "connections_created": 0,
            "connections_closed": 0,
            "queries_executed": 0,
            "errors": 0,
        }

    async def initialize(self):
        """Initialize connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=self.command_timeout,
                max_queries=self.max_queries,
                max_inactive_connection_lifetime=self.max_inactive_connection_lifetime,
                init=self._init_connection,
            )

            logger.info(
                f"PostgreSQL connection pool initialized (min: {self.min_size}, max: {self.max_size})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
            raise

    async def _init_connection(self, conn):
        """Initialize individual connections"""
        # Set connection-specific settings
        await conn.execute("SET timezone TO 'UTC'")
        await conn.execute("SET statement_timeout TO '30s'")

        self._stats["connections_created"] += 1

    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            self._stats["connections_closed"] += self._stats["connections_created"]
            logger.info("PostgreSQL connection pool closed")

    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as connection:
            try:
                yield connection
                self._stats["queries_executed"] += 1
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Database query error: {e}")
                raise

    async def execute(self, query: str, *args):
        """Execute query with connection pooling"""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args):
        """Fetch multiple rows with connection pooling"""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args):
        """Fetch single row with connection pooling"""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        """Fetch single value with connection pooling"""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        current_time = time.time()
        uptime = current_time - self._stats["created_at"]

        pool_stats = {}
        if self.pool:
            pool_stats = {
                "size": self.pool.get_size(),
                "min_size": self.pool.get_min_size(),
                "max_size": self.pool.get_max_size(),
                "idle_connections": self.pool.get_idle_size(),
            }

        return {
            **self._stats,
            "uptime_seconds": uptime,
            "queries_per_second": self._stats["queries_executed"] / max(uptime, 1),
            "error_rate": self._stats["errors"]
            / max(self._stats["queries_executed"], 1),
            "pool": pool_stats,
        }


class RedisConnectionPool:
    """Redis connection pool manager"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_connections: int = 50,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
    ):
        """Initialize Redis connection pool"""
        if not REDIS_AVAILABLE:
            raise RuntimeError("aioredis is required for Redis connection pooling")

        self.redis_url = redis_url
        self.max_connections = max_connections
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval

        self.pool: Optional[aioredis.ConnectionPool] = None
        self.redis: Optional[aioredis.Redis] = None
        self._stats = {
            "created_at": time.time(),
            "commands_executed": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def initialize(self):
        """Initialize Redis connection pool"""
        try:
            self.pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=self.retry_on_timeout,
            )

            self.redis = aioredis.Redis(connection_pool=self.pool)

            # Test connection
            await self.redis.ping()

            logger.info(
                f"Redis connection pool initialized (max connections: {self.max_connections})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Redis connection pool: {e}")
            raise

    async def close(self):
        """Close Redis connection pool"""
        if self.redis:
            await self.redis.close()
        if self.pool:
            await self.pool.disconnect()
        logger.info("Redis connection pool closed")

    async def get(self, key: str):
        """Get value from Redis"""
        try:
            result = await self.redis.get(key)
            self._stats["commands_executed"] += 1

            if result is not None:
                self._stats["cache_hits"] += 1
            else:
                self._stats["cache_misses"] += 1

            return result
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Redis GET error: {e}")
            return None

    async def set(self, key: str, value: str, ex: Optional[int] = None):
        """Set value in Redis"""
        try:
            await self.redis.set(key, value, ex=ex)
            self._stats["commands_executed"] += 1
            return True
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Redis SET error: {e}")
            return False

    async def delete(self, key: str):
        """Delete key from Redis"""
        try:
            result = await self.redis.delete(key)
            self._stats["commands_executed"] += 1
            return result
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Redis DELETE error: {e}")
            return 0

    async def exists(self, key: str):
        """Check if key exists in Redis"""
        try:
            result = await self.redis.exists(key)
            self._stats["commands_executed"] += 1
            return bool(result)
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Redis EXISTS error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis pool statistics"""
        current_time = time.time()
        uptime = current_time - self._stats["created_at"]

        pool_stats = {}
        if self.pool:
            pool_stats = {
                "max_connections": self.max_connections,
                "created_connections": len(self.pool._created_connections),
                "available_connections": len(self.pool._available_connections),
                "in_use_connections": len(self.pool._in_use_connections),
            }

        cache_hit_rate = 0
        total_cache_ops = self._stats["cache_hits"] + self._stats["cache_misses"]
        if total_cache_ops > 0:
            cache_hit_rate = self._stats["cache_hits"] / total_cache_ops

        return {
            **self._stats,
            "uptime_seconds": uptime,
            "commands_per_second": self._stats["commands_executed"] / max(uptime, 1),
            "error_rate": self._stats["errors"]
            / max(self._stats["commands_executed"], 1),
            "cache_hit_rate": cache_hit_rate,
            "pool": pool_stats,
        }


class ConnectionPoolManager:
    """Manages multiple connection pools"""

    def __init__(self):
        self.postgres_pool: Optional[PostgreSQLConnectionPool] = None
        self.redis_pool: Optional[RedisConnectionPool] = None
        self._initialized = False

    async def initialize_postgres(
        self, connection_string: str, min_size: int = 2, max_size: int = 10, **kwargs
    ):
        """Initialize PostgreSQL connection pool"""
        self.postgres_pool = PostgreSQLConnectionPool(
            connection_string=connection_string,
            min_size=min_size,
            max_size=max_size,
            **kwargs,
        )
        await self.postgres_pool.initialize()

    async def initialize_redis(
        self,
        redis_url: str = "redis://localhost:6379",
        max_connections: int = 50,
        **kwargs,
    ):
        """Initialize Redis connection pool"""
        self.redis_pool = RedisConnectionPool(
            redis_url=redis_url, max_connections=max_connections, **kwargs
        )
        await self.redis_pool.initialize()

    async def close_all(self):
        """Close all connection pools"""
        if self.postgres_pool:
            await self.postgres_pool.close()
        if self.redis_pool:
            await self.redis_pool.close()

        self._initialized = False
        logger.info("All connection pools closed")

    def get_postgres_pool(self) -> Optional[PostgreSQLConnectionPool]:
        """Get PostgreSQL connection pool"""
        return self.postgres_pool

    def get_redis_pool(self) -> Optional[RedisConnectionPool]:
        """Get Redis connection pool"""
        return self.redis_pool

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all pools"""
        stats = {}

        if self.postgres_pool:
            stats["postgresql"] = self.postgres_pool.get_stats()

        if self.redis_pool:
            stats["redis"] = self.redis_pool.get_stats()

        return stats


# Global connection pool manager
_pool_manager: Optional[ConnectionPoolManager] = None


def get_pool_manager() -> ConnectionPoolManager:
    """Get global connection pool manager"""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
    return _pool_manager


async def initialize_connection_pools(config):
    """Initialize connection pools from configuration"""
    manager = get_pool_manager()

    # Initialize PostgreSQL pool if configured
    postgres_url = getattr(config, "DATABASE_URL", None)
    if postgres_url and ASYNCPG_AVAILABLE:
        try:
            await manager.initialize_postgres(
                connection_string=postgres_url,
                min_size=getattr(config, "DB_POOL_MIN_SIZE", 2),
                max_size=getattr(config, "DB_POOL_SIZE", 10),
            )
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")

    # Initialize Redis pool if configured
    redis_url = getattr(config, "REDIS_URL", None)
    if redis_url and REDIS_AVAILABLE:
        try:
            await manager.initialize_redis(
                redis_url=redis_url,
                max_connections=getattr(config, "REDIS_MAX_CONNECTIONS", 50),
            )
        except Exception as e:
            logger.error(f"Failed to initialize Redis pool: {e}")


async def close_connection_pools():
    """Close all connection pools"""
    global _pool_manager
    if _pool_manager:
        await _pool_manager.close_all()
        _pool_manager = None
