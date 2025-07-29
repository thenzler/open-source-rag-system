#!/usr/bin/env python3
"""
Redis Caching Service for RAG System
Provides intelligent caching for queries, documents, and embeddings
"""
import hashlib
import json
import logging
import pickle
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    aioredis = None
    REDIS_AVAILABLE = False

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class CacheKeyType(Enum):
    """Cache key types for organized storage"""
    QUERY_RESULT = "query"
    DOCUMENT_EMBEDDING = "doc_emb"
    SEARCH_RESULT = "search"
    LLM_RESPONSE = "llm"
    USER_SESSION = "session"
    SYSTEM_CONFIG = "config"
    TENANT_DATA = "tenant"

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    memory_usage: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class RedisCacheService:
    """Redis-based caching service with intelligent cache management"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        redis_db: int = 0,
        default_ttl: int = 3600,  # 1 hour
        max_connections: int = 20,
        enable_compression: bool = True,
        key_prefix: str = "rag_cache"
    ):
        self.redis_url = redis_url
        self.redis_db = redis_db
        self.default_ttl = default_ttl
        self.max_connections = max_connections
        self.enable_compression = enable_compression
        self.key_prefix = key_prefix
        
        self.redis_client: Optional[aioredis.Redis] = None
        self.connection_pool: Optional[aioredis.ConnectionPool] = None
        self.metrics = CacheMetrics()
        self.is_available = False
        
        # TTL configurations for different cache types
        self.ttl_config = {
            CacheKeyType.QUERY_RESULT: 1800,      # 30 minutes
            CacheKeyType.DOCUMENT_EMBEDDING: 86400,  # 24 hours
            CacheKeyType.SEARCH_RESULT: 1800,     # 30 minutes
            CacheKeyType.LLM_RESPONSE: 3600,      # 1 hour
            CacheKeyType.USER_SESSION: 7200,     # 2 hours
            CacheKeyType.SYSTEM_CONFIG: 300,     # 5 minutes
            CacheKeyType.TENANT_DATA: 3600,      # 1 hour
        }
    
    async def initialize(self) -> bool:
        """Initialize Redis connection"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, caching disabled")
            return False
        
        try:
            # Create connection pool
            self.connection_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                db=self.redis_db,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Create Redis client
            self.redis_client = aioredis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=False  # We handle encoding/decoding manually
            )
            
            # Test connection
            await self.redis_client.ping()
            self.is_available = True
            
            logger.info(f"Redis cache service initialized: {self.redis_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self.is_available = False
            return False
    
    async def close(self):
        """Close Redis connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.connection_pool:
            await self.connection_pool.disconnect()
        
        logger.info("Redis cache service closed")
    
    def _generate_key(
        self, 
        key_type: CacheKeyType, 
        identifier: str, 
        tenant_id: Optional[str] = None
    ) -> str:
        """Generate a standardized cache key"""
        parts = [self.key_prefix, key_type.value]
        
        if tenant_id:
            parts.append(f"tenant:{tenant_id}")
        
        parts.append(identifier)
        return ":".join(parts)
    
    def _hash_key(self, data: Union[str, Dict, List]) -> str:
        """Generate a hash for complex cache keys"""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True, default=str)
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            if self.enable_compression:
                import gzip
                serialized = pickle.dumps(value)
                return gzip.compress(serialized)
            else:
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            if self.enable_compression:
                import gzip
                decompressed = gzip.decompress(data)
                return pickle.loads(decompressed)
            else:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise
    
    async def get(
        self, 
        key_type: CacheKeyType, 
        identifier: str, 
        tenant_id: Optional[str] = None
    ) -> Optional[Any]:
        """Get value from cache"""
        if not self.is_available:
            return None
        
        try:
            cache_key = self._generate_key(key_type, identifier, tenant_id)
            data = await self.redis_client.get(cache_key)
            
            if data is None:
                self.metrics.misses += 1
                return None
            
            value = self._deserialize_value(data)
            self.metrics.hits += 1
            
            logger.debug(f"Cache hit: {cache_key}")
            return value
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.metrics.misses += 1
            return None
    
    async def set(
        self,
        key_type: CacheKeyType,
        identifier: str,
        value: Any,
        ttl: Optional[int] = None,
        tenant_id: Optional[str] = None
    ) -> bool:
        """Set value in cache"""
        if not self.is_available:
            return False
        
        try:
            cache_key = self._generate_key(key_type, identifier, tenant_id)
            serialized_value = self._serialize_value(value)
            
            # Use type-specific TTL if not provided
            if ttl is None:
                ttl = self.ttl_config.get(key_type, self.default_ttl)
            
            await self.redis_client.setex(cache_key, ttl, serialized_value)
            self.metrics.sets += 1
            
            logger.debug(f"Cache set: {cache_key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(
        self,
        key_type: CacheKeyType,
        identifier: str,
        tenant_id: Optional[str] = None
    ) -> bool:
        """Delete value from cache"""
        if not self.is_available:
            return False
        
        try:
            cache_key = self._generate_key(key_type, identifier, tenant_id)
            deleted = await self.redis_client.delete(cache_key)
            
            if deleted:
                self.metrics.deletes += 1
                logger.debug(f"Cache delete: {cache_key}")
            
            return bool(deleted)
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def invalidate_pattern(
        self,
        pattern: str,
        tenant_id: Optional[str] = None
    ) -> int:
        """Invalidate all keys matching a pattern"""
        if not self.is_available:
            return 0
        
        try:
            if tenant_id:
                search_pattern = f"{self.key_prefix}:*:tenant:{tenant_id}:*{pattern}*"
            else:
                search_pattern = f"{self.key_prefix}:*{pattern}*"
            
            keys = await self.redis_client.keys(search_pattern)
            if keys:
                deleted = await self.redis_client.delete(*keys)
                self.metrics.deletes += deleted
                logger.info(f"Invalidated {deleted} keys matching pattern: {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Pattern invalidation error: {e}")
            return 0
    
    async def get_query_cache(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Get cached query result"""
        cache_data = {
            'query': query,
            'filters': filters or {}
        }
        cache_key = self._hash_key(cache_data)
        
        return await self.get(CacheKeyType.QUERY_RESULT, cache_key, tenant_id)
    
    async def set_query_cache(
        self,
        query: str,
        result: Dict,
        tenant_id: Optional[str] = None,
        filters: Optional[Dict] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache query result"""
        cache_data = {
            'query': query,
            'filters': filters or {}
        }
        cache_key = self._hash_key(cache_data)
        
        # Add metadata to result
        cached_result = {
            'result': result,
            'cached_at': time.time(),
            'query_hash': cache_key
        }
        
        return await self.set(CacheKeyType.QUERY_RESULT, cache_key, cached_result, ttl, tenant_id)
    
    async def get_document_embedding(
        self,
        document_id: str,
        model_name: str,
        tenant_id: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Get cached document embedding"""
        cache_key = f"{document_id}:{model_name}"
        cached_data = await self.get(CacheKeyType.DOCUMENT_EMBEDDING, cache_key, tenant_id)
        
        if cached_data and 'embedding' in cached_data:
            return np.array(cached_data['embedding'])
        
        return None
    
    async def set_document_embedding(
        self,
        document_id: str,
        model_name: str,
        embedding: np.ndarray,
        tenant_id: Optional[str] = None
    ) -> bool:
        """Cache document embedding"""
        cache_key = f"{document_id}:{model_name}"
        cached_data = {
            'embedding': embedding.tolist(),
            'model_name': model_name,
            'created_at': time.time()
        }
        
        return await self.set(CacheKeyType.DOCUMENT_EMBEDDING, cache_key, cached_data, None, tenant_id)
    
    async def get_search_results(
        self,
        query_embedding: np.ndarray,
        tenant_id: Optional[str] = None,
        top_k: int = 5
    ) -> Optional[List[Dict]]:
        """Get cached search results for similar embeddings"""
        # Create a hash of the query embedding
        embedding_hash = hashlib.sha256(query_embedding.tobytes()).hexdigest()[:16]
        cache_key = f"{embedding_hash}:k{top_k}"
        
        return await self.get(CacheKeyType.SEARCH_RESULT, cache_key, tenant_id)
    
    async def set_search_results(
        self,
        query_embedding: np.ndarray,
        results: List[Dict],
        tenant_id: Optional[str] = None,
        top_k: int = 5
    ) -> bool:
        """Cache search results"""
        embedding_hash = hashlib.sha256(query_embedding.tobytes()).hexdigest()[:16]
        cache_key = f"{embedding_hash}:k{top_k}"
        
        cached_data = {
            'results': results,
            'embedding_hash': embedding_hash,
            'top_k': top_k,
            'cached_at': time.time()
        }
        
        return await self.set(CacheKeyType.SEARCH_RESULT, cache_key, cached_data, None, tenant_id)
    
    async def get_llm_response(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.7,
        tenant_id: Optional[str] = None
    ) -> Optional[str]:
        """Get cached LLM response"""
        cache_data = {
            'prompt': prompt,
            'model': model_name,
            'temperature': temperature
        }
        cache_key = self._hash_key(cache_data)
        
        cached = await self.get(CacheKeyType.LLM_RESPONSE, cache_key, tenant_id)
        return cached.get('response') if cached else None
    
    async def set_llm_response(
        self,
        prompt: str,
        response: str,
        model_name: str,
        temperature: float = 0.7,
        tenant_id: Optional[str] = None
    ) -> bool:
        """Cache LLM response"""
        cache_data = {
            'prompt': prompt,
            'model': model_name,
            'temperature': temperature
        }
        cache_key = self._hash_key(cache_data)
        
        cached_response = {
            'response': response,
            'model': model_name,
            'temperature': temperature,
            'cached_at': time.time()
        }
        
        return await self.set(CacheKeyType.LLM_RESPONSE, cache_key, cached_response, None, tenant_id)
    
    async def invalidate_tenant_cache(self, tenant_id: str) -> int:
        """Invalidate all cache entries for a tenant"""
        return await self.invalidate_pattern("", tenant_id)
    
    async def invalidate_document_cache(self, document_id: str, tenant_id: Optional[str] = None) -> int:
        """Invalidate all cache entries related to a document"""
        return await self.invalidate_pattern(f"*{document_id}*", tenant_id)
    
    async def get_cache_stats(self) -> Dict:
        """Get comprehensive cache statistics"""
        if not self.is_available:
            return {
                'status': 'unavailable',
                'redis_available': False
            }
        
        try:
            # Get Redis info
            redis_info = await self.redis_client.info()
            
            # Get memory usage
            memory_usage = redis_info.get('used_memory', 0)
            memory_human = redis_info.get('used_memory_human', '0B')
            
            # Get key counts by type
            key_counts = {}
            for key_type in CacheKeyType:
                pattern = f"{self.key_prefix}:{key_type.value}:*"
                keys = await self.redis_client.keys(pattern)
                key_counts[key_type.value] = len(keys)
            
            return {
                'status': 'available',
                'redis_available': True,
                'metrics': asdict(self.metrics),
                'memory_usage': memory_usage,
                'memory_usage_human': memory_human,
                'key_counts': key_counts,
                'redis_info': {
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'total_commands_processed': redis_info.get('total_commands_processed', 0),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'keyspace_misses': redis_info.get('keyspace_misses', 0),
                    'evicted_keys': redis_info.get('evicted_keys', 0),
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                'status': 'error',
                'redis_available': False,
                'error': str(e)
            }
    
    async def flush_cache(self, tenant_id: Optional[str] = None) -> bool:
        """Flush cache (optionally for specific tenant)"""
        if not self.is_available:
            return False
        
        try:
            if tenant_id:
                # Delete tenant-specific keys only
                deleted = await self.invalidate_tenant_cache(tenant_id)
                logger.info(f"Flushed {deleted} cache entries for tenant: {tenant_id}")
            else:
                # Flush entire database
                await self.redis_client.flushdb()
                logger.info("Flushed entire cache database")
            
            return True
            
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            return False

# Global cache service instance
_cache_service: Optional[RedisCacheService] = None

def get_cache_service() -> Optional[RedisCacheService]:
    """Get global cache service instance"""
    return _cache_service

async def initialize_cache_service(
    redis_url: str = "redis://localhost:6379",
    redis_db: int = 0,
    **kwargs
) -> Optional[RedisCacheService]:
    """Initialize global cache service"""
    global _cache_service
    
    _cache_service = RedisCacheService(
        redis_url=redis_url,
        redis_db=redis_db,
        **kwargs
    )
    
    success = await _cache_service.initialize()
    if not success:
        _cache_service = None
        logger.warning("Cache service initialization failed, running without cache")
    else:
        logger.info("Redis cache service initialized successfully")
    
    return _cache_service

async def shutdown_cache_service():
    """Shutdown global cache service"""
    global _cache_service
    if _cache_service:
        await _cache_service.close()
        _cache_service = None
        logger.info("Cache service shutdown completed")

# Decorators for easy caching
def cache_query_result(ttl: Optional[int] = None):
    """Decorator to cache query results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = get_cache_service()
            if not cache:
                return await func(*args, **kwargs)
            
            # Generate cache key from function arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached = await cache.get(CacheKeyType.QUERY_RESULT, cache_key)
            if cached:
                return cached
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(CacheKeyType.QUERY_RESULT, cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

def cache_embedding(ttl: Optional[int] = None):
    """Decorator to cache embeddings"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = get_cache_service()
            if not cache:
                return await func(*args, **kwargs)
            
            # Generate cache key from function arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached = await cache.get(CacheKeyType.DOCUMENT_EMBEDDING, cache_key)
            if cached:
                return cached
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(CacheKeyType.DOCUMENT_EMBEDDING, cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator