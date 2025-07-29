#!/usr/bin/env python3
"""
Tests for Redis Cache Service
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

try:
    from core.services.redis_cache_service import (CacheKeyType, CacheMetrics,
                                                   RedisCacheService,
                                                   cache_embedding,
                                                   cache_query_result,
                                                   get_cache_service,
                                                   initialize_cache_service)

    REDIS_CACHE_AVAILABLE = True
except ImportError:
    REDIS_CACHE_AVAILABLE = False
    pytest.skip("Redis cache service not available", allow_module_level=True)


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    mock_client = AsyncMock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = None
    mock_client.setex.return_value = True
    mock_client.delete.return_value = 1
    mock_client.keys.return_value = []
    mock_client.info.return_value = {
        "used_memory": 1024,
        "used_memory_human": "1KB",
        "connected_clients": 1,
        "total_commands_processed": 100,
        "keyspace_hits": 50,
        "keyspace_misses": 25,
        "evicted_keys": 0,
    }
    return mock_client


@pytest.fixture
async def cache_service(mock_redis):
    """Create cache service with mocked Redis"""
    with patch("core.services.redis_cache_service.aioredis") as mock_aioredis:
        mock_pool = AsyncMock()
        mock_aioredis.ConnectionPool.from_url.return_value = mock_pool
        mock_aioredis.Redis.return_value = mock_redis

        service = RedisCacheService(
            redis_url="redis://localhost:6379", redis_db=0, default_ttl=3600
        )

        success = await service.initialize()
        assert success is True

        yield service

        await service.close()


@pytest.mark.asyncio
async def test_cache_initialization(mock_redis):
    """Test cache service initialization"""
    with patch("core.services.redis_cache_service.aioredis") as mock_aioredis:
        mock_pool = AsyncMock()
        mock_aioredis.ConnectionPool.from_url.return_value = mock_pool
        mock_aioredis.Redis.return_value = mock_redis

        service = RedisCacheService()
        success = await service.initialize()

        assert success is True
        assert service.is_available is True
        assert service.redis_client == mock_redis

        await service.close()


@pytest.mark.asyncio
async def test_cache_initialization_failure():
    """Test cache service initialization failure"""
    with patch("core.services.redis_cache_service.aioredis") as mock_aioredis:
        mock_aioredis.ConnectionPool.from_url.side_effect = Exception(
            "Connection failed"
        )

        service = RedisCacheService()
        success = await service.initialize()

        assert success is False
        assert service.is_available is False


@pytest.mark.asyncio
async def test_cache_unavailable():
    """Test cache operations when Redis is unavailable"""
    with patch("core.services.redis_cache_service.REDIS_AVAILABLE", False):
        service = RedisCacheService()
        success = await service.initialize()

        assert success is False

        # Operations should return gracefully
        result = await service.get(CacheKeyType.QUERY_RESULT, "test_key")
        assert result is None

        success = await service.set(CacheKeyType.QUERY_RESULT, "test_key", "test_value")
        assert success is False


@pytest.mark.asyncio
async def test_basic_cache_operations(cache_service, mock_redis):
    """Test basic cache get/set/delete operations"""
    # Mock serialization
    test_value = {"test": "value"}
    serialized_data = b"serialized_data"

    with patch.object(
        cache_service, "_serialize_value", return_value=serialized_data
    ), patch.object(cache_service, "_deserialize_value", return_value=test_value):

        # Test set
        success = await cache_service.set(
            CacheKeyType.QUERY_RESULT, "test_key", test_value
        )
        assert success is True
        assert cache_service.metrics.sets == 1

        # Test get (cache hit)
        mock_redis.get.return_value = serialized_data
        result = await cache_service.get(CacheKeyType.QUERY_RESULT, "test_key")
        assert result == test_value
        assert cache_service.metrics.hits == 1

        # Test get (cache miss)
        mock_redis.get.return_value = None
        result = await cache_service.get(CacheKeyType.QUERY_RESULT, "missing_key")
        assert result is None
        assert cache_service.metrics.misses == 1

        # Test delete
        success = await cache_service.delete(CacheKeyType.QUERY_RESULT, "test_key")
        assert success is True
        assert cache_service.metrics.deletes == 1


@pytest.mark.asyncio
async def test_key_generation(cache_service):
    """Test cache key generation"""
    # Test basic key
    key = cache_service._generate_key(CacheKeyType.QUERY_RESULT, "test_id")
    assert key == "rag_cache:query:test_id"

    # Test key with tenant
    key = cache_service._generate_key(CacheKeyType.QUERY_RESULT, "test_id", "tenant1")
    assert key == "rag_cache:query:tenant:tenant1:test_id"

    # Test hash key
    data = {"query": "test", "filters": {"type": "doc"}}
    hash_key = cache_service._hash_key(data)
    assert len(hash_key) == 16  # SHA256 truncated to 16 chars
    assert isinstance(hash_key, str)


@pytest.mark.asyncio
async def test_query_cache(cache_service, mock_redis):
    """Test query result caching"""
    query = "What is the capital of France?"
    result = {"answer": "Paris", "sources": ["doc1.pdf"]}
    filters = {"type": "geography"}

    serialized_data = b"serialized_data"

    with patch.object(
        cache_service, "_serialize_value", return_value=serialized_data
    ), patch.object(
        cache_service,
        "_deserialize_value",
        return_value={
            "result": result,
            "cached_at": 1234567890,
            "query_hash": "abc123",
        },
    ):

        # Test set query cache
        success = await cache_service.set_query_cache(query, result, filters=filters)
        assert success is True

        # Test get query cache (hit)
        mock_redis.get.return_value = serialized_data
        cached_result = await cache_service.get_query_cache(query, filters=filters)
        assert cached_result["result"] == result

        # Test get query cache (miss)
        mock_redis.get.return_value = None
        cached_result = await cache_service.get_query_cache(
            "different query", filters=filters
        )
        assert cached_result is None


@pytest.mark.asyncio
async def test_document_embedding_cache(cache_service, mock_redis):
    """Test document embedding caching"""
    document_id = "doc123"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = np.array([0.1, 0.2, 0.3, 0.4])

    serialized_data = b"serialized_data"
    cached_data = {
        "embedding": embedding.tolist(),
        "model_name": model_name,
        "created_at": 1234567890,
    }

    with patch.object(
        cache_service, "_serialize_value", return_value=serialized_data
    ), patch.object(cache_service, "_deserialize_value", return_value=cached_data):

        # Test set embedding cache
        success = await cache_service.set_document_embedding(
            document_id, model_name, embedding
        )
        assert success is True

        # Test get embedding cache (hit)
        mock_redis.get.return_value = serialized_data
        cached_embedding = await cache_service.get_document_embedding(
            document_id, model_name
        )
        assert isinstance(cached_embedding, np.ndarray)
        np.testing.assert_array_equal(cached_embedding, embedding)

        # Test get embedding cache (miss)
        mock_redis.get.return_value = None
        cached_embedding = await cache_service.get_document_embedding(
            "different_doc", model_name
        )
        assert cached_embedding is None


@pytest.mark.asyncio
async def test_search_results_cache(cache_service, mock_redis):
    """Test search results caching"""
    query_embedding = np.array([0.1, 0.2, 0.3])
    results = [
        {"document_id": "doc1", "score": 0.9},
        {"document_id": "doc2", "score": 0.8},
    ]

    serialized_data = b"serialized_data"
    cached_data = {
        "results": results,
        "embedding_hash": "hash123",
        "top_k": 5,
        "cached_at": 1234567890,
    }

    with patch.object(
        cache_service, "_serialize_value", return_value=serialized_data
    ), patch.object(cache_service, "_deserialize_value", return_value=cached_data):

        # Test set search cache
        success = await cache_service.set_search_results(query_embedding, results)
        assert success is True

        # Test get search cache (hit)
        mock_redis.get.return_value = serialized_data
        cached_results = await cache_service.get_search_results(query_embedding)
        assert cached_results == results


@pytest.mark.asyncio
async def test_llm_response_cache(cache_service, mock_redis):
    """Test LLM response caching"""
    prompt = "Explain quantum computing"
    response = "Quantum computing is a computing paradigm..."
    model_name = "llama2"
    temperature = 0.7

    serialized_data = b"serialized_data"
    cached_data = {
        "response": response,
        "model": model_name,
        "temperature": temperature,
        "cached_at": 1234567890,
    }

    with patch.object(
        cache_service, "_serialize_value", return_value=serialized_data
    ), patch.object(cache_service, "_deserialize_value", return_value=cached_data):

        # Test set LLM cache
        success = await cache_service.set_llm_response(
            prompt, response, model_name, temperature
        )
        assert success is True

        # Test get LLM cache (hit)
        mock_redis.get.return_value = serialized_data
        cached_response = await cache_service.get_llm_response(
            prompt, model_name, temperature
        )
        assert cached_response == response

        # Test get LLM cache (miss with different temperature)
        mock_redis.get.return_value = None
        cached_response = await cache_service.get_llm_response(prompt, model_name, 0.5)
        assert cached_response is None


@pytest.mark.asyncio
async def test_invalidation_patterns(cache_service, mock_redis):
    """Test cache invalidation patterns"""
    # Mock keys for pattern matching
    mock_redis.keys.return_value = [
        b"rag_cache:query:tenant:test_tenant:key1",
        b"rag_cache:doc_emb:tenant:test_tenant:key2",
        b"rag_cache:search:tenant:test_tenant:key3",
    ]
    mock_redis.delete.return_value = 3

    # Test pattern invalidation
    deleted_count = await cache_service.invalidate_pattern(
        "test_pattern", "test_tenant"
    )
    assert deleted_count == 3
    assert cache_service.metrics.deletes == 3

    # Test tenant invalidation
    deleted_count = await cache_service.invalidate_tenant_cache("test_tenant")
    assert deleted_count == 3

    # Test document invalidation
    deleted_count = await cache_service.invalidate_document_cache(
        "doc123", "test_tenant"
    )
    assert deleted_count == 3


@pytest.mark.asyncio
async def test_cache_stats(cache_service, mock_redis):
    """Test cache statistics"""
    # Mock Redis info and keys
    mock_redis.info.return_value = {
        "used_memory": 2048,
        "used_memory_human": "2KB",
        "connected_clients": 2,
        "total_commands_processed": 200,
        "keyspace_hits": 100,
        "keyspace_misses": 50,
        "evicted_keys": 5,
    }

    # Mock different key types
    mock_redis.keys.side_effect = [
        [b"key1", b"key2"],  # QUERY_RESULT
        [b"key3"],  # DOCUMENT_EMBEDDING
        [],  # SEARCH_RESULT
        [b"key4", b"key5", b"key6"],  # LLM_RESPONSE
        [],  # USER_SESSION
        [b"key7"],  # SYSTEM_CONFIG
        [],  # TENANT_DATA
    ]

    stats = await cache_service.get_cache_stats()

    assert stats["status"] == "available"
    assert stats["redis_available"] is True
    assert stats["memory_usage"] == 2048
    assert stats["memory_usage_human"] == "2KB"
    assert stats["key_counts"]["query"] == 2
    assert stats["key_counts"]["doc_emb"] == 1
    assert stats["key_counts"]["llm"] == 3
    assert stats["redis_info"]["connected_clients"] == 2


@pytest.mark.asyncio
async def test_cache_flush(cache_service, mock_redis):
    """Test cache flushing"""
    # Test full flush
    success = await cache_service.flush_cache()
    assert success is True
    mock_redis.flushdb.assert_called_once()

    # Reset mock
    mock_redis.reset_mock()

    # Test tenant-specific flush
    mock_redis.keys.return_value = [b"key1", b"key2"]
    mock_redis.delete.return_value = 2

    success = await cache_service.flush_cache("test_tenant")
    assert success is True
    mock_redis.keys.assert_called_once()
    mock_redis.delete.assert_called_once()


@pytest.mark.asyncio
async def test_serialization_compression(cache_service):
    """Test data serialization with compression"""
    test_data = {"large_data": "x" * 1000, "numbers": [1, 2, 3, 4, 5]}

    # Test with compression enabled
    cache_service.enable_compression = True
    serialized = cache_service._serialize_value(test_data)
    deserialized = cache_service._deserialize_value(serialized)
    assert deserialized == test_data

    # Test with compression disabled
    cache_service.enable_compression = False
    serialized = cache_service._serialize_value(test_data)
    deserialized = cache_service._deserialize_value(serialized)
    assert deserialized == test_data


def test_cache_metrics():
    """Test cache metrics calculations"""
    metrics = CacheMetrics()
    assert metrics.hit_rate == 0.0

    metrics.hits = 75
    metrics.misses = 25
    assert metrics.hit_rate == 0.75

    metrics.hits = 0
    metrics.misses = 10
    assert metrics.hit_rate == 0.0


@pytest.mark.asyncio
async def test_ttl_configuration(cache_service, mock_redis):
    """Test TTL configuration for different cache types"""
    test_value = {"test": "data"}
    serialized_data = b"serialized_data"

    with patch.object(cache_service, "_serialize_value", return_value=serialized_data):
        # Test query result TTL
        await cache_service.set(CacheKeyType.QUERY_RESULT, "test", test_value)
        mock_redis.setex.assert_called_with(
            "rag_cache:query:test",
            cache_service.ttl_config[CacheKeyType.QUERY_RESULT],
            serialized_data,
        )

        # Test document embedding TTL
        await cache_service.set(CacheKeyType.DOCUMENT_EMBEDDING, "test", test_value)
        mock_redis.setex.assert_called_with(
            "rag_cache:doc_emb:test",
            cache_service.ttl_config[CacheKeyType.DOCUMENT_EMBEDDING],
            serialized_data,
        )

        # Test custom TTL
        await cache_service.set(CacheKeyType.SYSTEM_CONFIG, "test", test_value, ttl=600)
        mock_redis.setex.assert_called_with(
            "rag_cache:config:test", 600, serialized_data
        )


@pytest.mark.asyncio
async def test_cache_decorators():
    """Test cache decorators"""

    @cache_query_result(ttl=300)
    async def test_function(arg1, arg2):
        return {"result": f"{arg1}_{arg2}"}

    # Mock cache service
    with patch("core.services.redis_cache_service.get_cache_service") as mock_get_cache:
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None  # Cache miss
        mock_cache.set.return_value = True
        mock_get_cache.return_value = mock_cache

        # First call - should execute function and cache result
        result = await test_function("hello", "world")
        assert result == {"result": "hello_world"}
        mock_cache.set.assert_called_once()

        # Reset mock
        mock_cache.reset_mock()
        mock_cache.get.return_value = {"result": "cached_hello_world"}  # Cache hit

        # Second call - should return cached result
        result = await test_function("hello", "world")
        assert result == {"result": "cached_hello_world"}
        mock_cache.set.assert_not_called()


@pytest.mark.asyncio
async def test_global_cache_service():
    """Test global cache service management"""
    with patch("core.services.redis_cache_service.aioredis") as mock_aioredis:
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_pool = AsyncMock()
        mock_aioredis.ConnectionPool.from_url.return_value = mock_pool
        mock_aioredis.Redis.return_value = mock_redis

        # Test initialization
        cache_service = await initialize_cache_service()
        assert cache_service is not None
        assert get_cache_service() == cache_service

        # Test getting service
        service = get_cache_service()
        assert service == cache_service

        # Test shutdown
        from core.services.redis_cache_service import shutdown_cache_service

        await shutdown_cache_service()
        assert get_cache_service() is None


@pytest.mark.asyncio
async def test_error_handling(cache_service, mock_redis):
    """Test error handling in cache operations"""
    # Test serialization error
    with patch.object(
        cache_service, "_serialize_value", side_effect=Exception("Serialization failed")
    ):
        success = await cache_service.set(CacheKeyType.QUERY_RESULT, "test", "value")
        assert success is False

    # Test Redis connection error
    mock_redis.get.side_effect = Exception("Redis connection failed")
    result = await cache_service.get(CacheKeyType.QUERY_RESULT, "test")
    assert result is None
    assert cache_service.metrics.misses == 1

    # Test delete error
    mock_redis.delete.side_effect = Exception("Redis delete failed")
    success = await cache_service.delete(CacheKeyType.QUERY_RESULT, "test")
    assert success is False


if __name__ == "__main__":
    pytest.main([__file__])
