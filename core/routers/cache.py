#!/usr/bin/env python3
"""
Cache Management API Router
Endpoints for managing Redis cache operations
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

try:
    from ..services.redis_cache_service import (CacheKeyType,
                                                RedisCacheService,
                                                get_cache_service)

    REDIS_CACHE_AVAILABLE = True
except ImportError:
    # Fallback when Redis is not available
    REDIS_CACHE_AVAILABLE = False

    def get_cache_service():
        return None

    class CacheKeyType:
        QUERY_RESULT = "query"
        DOCUMENT_EMBEDDING = "doc_emb"
        SEARCH_RESULT = "search"
        LLM_RESPONSE = "llm"
        USER_SESSION = "session"
        SYSTEM_CONFIG = "config"
        TENANT_DATA = "tenant"

    class RedisCacheService:
        pass


from ..middleware.tenant_middleware import get_current_tenant
from ..utils.security import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cache", tags=["cache"])


def get_cache() -> Optional[RedisCacheService]:
    """Dependency to get cache service"""
    return get_cache_service()


@router.get("/stats", response_model=dict)
async def get_cache_statistics(cache: Optional[RedisCacheService] = Depends(get_cache)):
    """Get comprehensive cache statistics"""
    if not cache:
        return {
            "status": "unavailable",
            "redis_available": False,
            "message": "Redis cache service not available",
        }

    try:
        stats = await cache.get_cache_stats()
        return stats

    except Exception as e:
        logger.error(f"Failed to get cache statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=dict)
async def cache_health_check(cache: Optional[RedisCacheService] = Depends(get_cache)):
    """Health check for cache service"""
    if not cache:
        return {"status": "unavailable", "redis_available": False, "healthy": False}

    try:
        if cache.is_available and cache.redis_client:
            # Test Redis connection
            await cache.redis_client.ping()
            return {
                "status": "healthy",
                "redis_available": True,
                "healthy": True,
                "hit_rate": cache.metrics.hit_rate,
            }
        else:
            return {"status": "unavailable", "redis_available": False, "healthy": False}

    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": "unhealthy",
            "redis_available": False,
            "healthy": False,
            "error": str(e),
        }


@router.delete("/flush", response_model=dict)
async def flush_cache(
    tenant_id: Optional[str] = Query(
        None, description="Flush cache for specific tenant only"
    ),
    current_tenant: Optional[str] = Depends(get_current_tenant),
    cache: Optional[RedisCacheService] = Depends(get_cache),
):
    """Flush cache entries"""
    if not cache:
        raise HTTPException(status_code=503, detail="Cache service not available")

    try:
        # Use current tenant if not specified
        target_tenant = tenant_id or current_tenant

        success = await cache.flush_cache(target_tenant)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to flush cache")

        return {
            "message": (
                f"Cache flushed successfully for tenant: {target_tenant}"
                if target_tenant
                else "Entire cache flushed successfully"
            ),
            "tenant_id": target_tenant,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Failed to flush cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/invalidate/pattern", response_model=dict)
async def invalidate_pattern(
    pattern: str = Query(..., description="Pattern to match for cache invalidation"),
    tenant_id: Optional[str] = Query(
        None, description="Tenant ID for scoped invalidation"
    ),
    current_tenant: Optional[str] = Depends(get_current_tenant),
    cache: Optional[RedisCacheService] = Depends(get_cache),
):
    """Invalidate cache entries matching a pattern"""
    if not cache:
        raise HTTPException(status_code=503, detail="Cache service not available")

    try:
        # Use current tenant if not specified
        target_tenant = tenant_id or current_tenant

        deleted_count = await cache.invalidate_pattern(pattern, target_tenant)

        return {
            "message": f"Invalidated {deleted_count} cache entries",
            "pattern": pattern,
            "tenant_id": target_tenant,
            "deleted_count": deleted_count,
        }

    except Exception as e:
        logger.error(f"Failed to invalidate pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/invalidate/document/{document_id}", response_model=dict)
async def invalidate_document_cache(
    document_id: str,
    tenant_id: Optional[str] = Depends(get_current_tenant),
    cache: Optional[RedisCacheService] = Depends(get_cache),
):
    """Invalidate all cache entries related to a specific document"""
    if not cache:
        raise HTTPException(status_code=503, detail="Cache service not available")

    try:
        deleted_count = await cache.invalidate_document_cache(document_id, tenant_id)

        return {
            "message": f"Invalidated {deleted_count} cache entries for document",
            "document_id": document_id,
            "tenant_id": tenant_id,
            "deleted_count": deleted_count,
        }

    except Exception as e:
        logger.error(f"Failed to invalidate document cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/invalidate/tenant/{tenant_id}", response_model=dict)
async def invalidate_tenant_cache(
    tenant_id: str,
    current_tenant: Optional[str] = Depends(get_current_tenant),
    cache: Optional[RedisCacheService] = Depends(get_cache),
):
    """Invalidate all cache entries for a specific tenant"""
    if not cache:
        raise HTTPException(status_code=503, detail="Cache service not available")

    # Security check: only allow invalidating current tenant's cache
    if current_tenant and tenant_id != current_tenant:
        raise HTTPException(
            status_code=403, detail="Can only invalidate current tenant's cache"
        )

    try:
        deleted_count = await cache.invalidate_tenant_cache(tenant_id)

        return {
            "message": f"Invalidated {deleted_count} cache entries for tenant",
            "tenant_id": tenant_id,
            "deleted_count": deleted_count,
        }

    except Exception as e:
        logger.error(f"Failed to invalidate tenant cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/{query_hash}", response_model=dict)
async def get_cached_query(
    query_hash: str,
    tenant_id: Optional[str] = Depends(get_current_tenant),
    cache: Optional[RedisCacheService] = Depends(get_cache),
):
    """Get a cached query result by hash"""
    if not cache:
        raise HTTPException(status_code=503, detail="Cache service not available")

    try:
        cached_result = await cache.get(
            CacheKeyType.QUERY_RESULT, query_hash, tenant_id
        )

        if not cached_result:
            raise HTTPException(status_code=404, detail="Cached query not found")

        return {
            "query_hash": query_hash,
            "cached_result": cached_result,
            "tenant_id": tenant_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cached query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/cache", response_model=dict)
async def cache_query_result(
    query: str,
    result: dict,
    filters: Optional[dict] = None,
    ttl: Optional[int] = None,
    tenant_id: Optional[str] = Depends(get_current_tenant),
    cache: Optional[RedisCacheService] = Depends(get_cache),
):
    """Manually cache a query result"""
    if not cache:
        raise HTTPException(status_code=503, detail="Cache service not available")

    try:
        success = await cache.set_query_cache(
            query=query, result=result, tenant_id=tenant_id, filters=filters, ttl=ttl
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to cache query result")

        # Generate the cache key for reference
        cache_data = {"query": query, "filters": filters or {}}
        cache_key = cache._hash_key(cache_data)

        return {
            "message": "Query result cached successfully",
            "cache_key": cache_key,
            "query": query,
            "tenant_id": tenant_id,
            "ttl": ttl,
        }

    except Exception as e:
        logger.error(f"Failed to cache query result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=dict)
async def get_cache_metrics(cache: Optional[RedisCacheService] = Depends(get_cache)):
    """Get cache performance metrics"""
    if not cache:
        return {"available": False, "metrics": None}

    try:
        return {
            "available": cache.is_available,
            "metrics": {
                "hits": cache.metrics.hits,
                "misses": cache.metrics.misses,
                "sets": cache.metrics.sets,
                "deletes": cache.metrics.deletes,
                "hit_rate": cache.metrics.hit_rate,
                "total_requests": cache.metrics.hits + cache.metrics.misses,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get cache metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test", response_model=dict)
async def test_cache_operations(
    cache: Optional[RedisCacheService] = Depends(get_cache),
):
    """Test basic cache operations"""
    if not cache:
        raise HTTPException(status_code=503, detail="Cache service not available")

    try:
        test_key = "cache_test"
        test_value = {"test": True, "timestamp": "now"}

        # Test set
        set_success = await cache.set(
            CacheKeyType.SYSTEM_CONFIG, test_key, test_value, 60
        )
        if not set_success:
            return {"success": False, "error": "Failed to set test value"}

        # Test get
        retrieved_value = await cache.get(CacheKeyType.SYSTEM_CONFIG, test_key)
        if retrieved_value != test_value:
            return {"success": False, "error": "Retrieved value doesn't match"}

        # Test delete
        delete_success = await cache.delete(CacheKeyType.SYSTEM_CONFIG, test_key)
        if not delete_success:
            return {"success": False, "error": "Failed to delete test value"}

        # Verify deletion
        deleted_value = await cache.get(CacheKeyType.SYSTEM_CONFIG, test_key)
        if deleted_value is not None:
            return {"success": False, "error": "Value not properly deleted"}

        return {
            "success": True,
            "message": "All cache operations completed successfully",
            "operations_tested": ["set", "get", "delete"],
        }

    except Exception as e:
        logger.error(f"Cache test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config", response_model=dict)
async def get_cache_configuration(
    cache: Optional[RedisCacheService] = Depends(get_cache),
):
    """Get cache service configuration"""
    if not cache:
        return {"available": False, "config": None}

    return {
        "available": cache.is_available,
        "config": {
            "redis_url": (
                cache.redis_url.replace(
                    cache.redis_url.split("@")[-1].split("/")[0], "***"
                )
                if "@" in cache.redis_url
                else cache.redis_url
            ),
            "redis_db": cache.redis_db,
            "default_ttl": cache.default_ttl,
            "max_connections": cache.max_connections,
            "enable_compression": cache.enable_compression,
            "key_prefix": cache.key_prefix,
            "ttl_config": {key.value: ttl for key, ttl in cache.ttl_config.items()},
        },
    }
