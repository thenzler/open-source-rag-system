"""
Performance optimization tools for the RAG system.
Includes caching, query optimization, and performance monitoring.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import pickle
from contextlib import asynccontextmanager

import redis
import psutil
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Summary
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_database

logger = logging.getLogger(__name__)
settings = get_settings()

# Prometheus metrics
QUERY_COUNTER = Counter('rag_queries_total', 'Total number of queries', ['endpoint', 'status'])
QUERY_DURATION = Histogram('rag_query_duration_seconds', 'Query processing time', ['endpoint'])
CACHE_HIT_RATE = Gauge('rag_cache_hit_rate', 'Cache hit rate percentage')
MEMORY_USAGE = Gauge('rag_memory_usage_bytes', 'Memory usage in bytes', ['component'])
CONCURRENT_QUERIES = Gauge('rag_concurrent_queries', 'Number of concurrent queries')
VECTOR_SEARCH_LATENCY = Summary('rag_vector_search_latency_seconds', 'Vector search latency')
EMBEDDING_GENERATION_TIME = Summary('rag_embedding_generation_seconds', 'Embedding generation time')


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    endpoint: str
    duration: float
    memory_usage: float
    cpu_usage: float
    cache_hit: bool
    error: Optional[str] = None


@dataclass
class QueryOptimization:
    """Query optimization configuration."""
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    enable_batching: bool = True
    batch_size: int = 10
    enable_prefetch: bool = True
    prefetch_size: int = 5
    enable_compression: bool = True
    max_query_length: int = 1000


class PerformanceCache:
    """High-performance caching layer for query results."""
    
    def __init__(self):
        self.redis_client = None
        self.local_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        self.max_local_cache_size = 1000
        
    async def initialize(self):
        """Initialize the cache."""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True
            )
            logger.info("Performance cache initialized")
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            # Try local cache first
            if key in self.local_cache:
                self.cache_stats['hits'] += 1
                return self.local_cache[key]
            
            # Try Redis cache
            if self.redis_client:
                value = await self._get_from_redis(key)
                if value is not None:
                    self.cache_stats['hits'] += 1
                    # Store in local cache
                    await self._set_local_cache(key, value)
                    return value
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache."""
        try:
            # Set in local cache
            await self._set_local_cache(key, value)
            
            # Set in Redis cache
            if self.redis_client:
                await self._set_in_redis(key, value, ttl)
                
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
    
    async def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        try:
            # Clear local cache
            keys_to_remove = [k for k in self.local_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.local_cache[key]
            
            # Clear Redis cache
            if self.redis_client:
                await self._invalidate_redis(pattern)
                
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get failed: {e}")
            return None
    
    async def _set_in_redis(self, key: str, value: Any, ttl: int):
        """Set value in Redis."""
        try:
            serialized = json.dumps(value, default=str)
            self.redis_client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Redis set failed: {e}")
    
    async def _set_local_cache(self, key: str, value: Any):
        """Set value in local cache."""
        if len(self.local_cache) >= self.max_local_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.local_cache))
            del self.local_cache[oldest_key]
            self.cache_stats['evictions'] += 1
        
        self.local_cache[key] = value
    
    async def _invalidate_redis(self, pattern: str):
        """Invalidate Redis entries matching pattern."""
        try:
            keys = self.redis_client.keys(f"*{pattern}*")
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis invalidation failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'evictions': self.cache_stats['evictions'],
            'local_cache_size': len(self.local_cache)
        }


class QueryOptimizer:
    """Query optimization engine."""
    
    def __init__(self):
        self.cache = PerformanceCache()
        self.query_patterns = {}
        self.optimization_config = QueryOptimization()
        
    async def initialize(self):
        """Initialize the optimizer."""
        await self.cache.initialize()
        logger.info("Query optimizer initialized")
    
    async def optimize_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a query for better performance."""
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(query, context)
            
            # Check cache
            if self.optimization_config.enable_caching:
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    return cached_result
            
            # Apply optimizations
            optimized_query = await self._apply_optimizations(query, context)
            
            # Cache result
            if self.optimization_config.enable_caching:
                await self.cache.set(
                    cache_key,
                    optimized_query,
                    self.optimization_config.cache_ttl
                )
            
            return optimized_query
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return {"query": query, "optimizations": []}
    
    async def _apply_optimizations(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply various optimizations to the query."""
        optimizations = []
        optimized_query = query
        
        # 1. Query length optimization
        if len(query) > self.optimization_config.max_query_length:
            optimized_query = query[:self.optimization_config.max_query_length]
            optimizations.append("truncated_query")
        
        # 2. Query pattern matching
        pattern_match = await self._match_query_pattern(optimized_query)
        if pattern_match:
            optimizations.append(f"pattern_match_{pattern_match}")
        
        # 3. Context-based optimization
        if context.get('user_preferences'):
            optimizations.append("context_optimization")
        
        # 4. Stop word removal for better performance
        optimized_query = await self._remove_stop_words(optimized_query)
        optimizations.append("stop_word_removal")
        
        return {
            "original_query": query,
            "optimized_query": optimized_query,
            "optimizations": optimizations,
            "estimated_improvement": len(optimizations) * 0.1  # Mock improvement
        }
    
    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """Generate cache key for query."""
        key_data = {
            "query": query,
            "context": context
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _match_query_pattern(self, query: str) -> Optional[str]:
        """Match query against known patterns."""
        # Simple pattern matching (could be enhanced with ML)
        patterns = {
            "definition": ["what is", "define", "meaning of"],
            "comparison": ["compare", "vs", "versus", "difference"],
            "process": ["how to", "steps", "process", "procedure"],
            "list": ["list", "types", "examples", "categories"]
        }
        
        query_lower = query.lower()
        for pattern_name, keywords in patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return pattern_name
        
        return None
    
    async def _remove_stop_words(self, query: str) -> str:
        """Remove stop words from query."""
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how',
            'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so'
        }
        
        words = query.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self):
        self.metrics_buffer = []
        self.max_buffer_size = 1000
        self.active_queries = 0
        
    async def record_metric(self, metric: PerformanceMetrics):
        """Record a performance metric."""
        try:
            # Add to buffer
            self.metrics_buffer.append(metric)
            
            # Maintain buffer size
            if len(self.metrics_buffer) > self.max_buffer_size:
                self.metrics_buffer.pop(0)
            
            # Update Prometheus metrics
            QUERY_COUNTER.labels(
                endpoint=metric.endpoint,
                status="success" if not metric.error else "error"
            ).inc()
            
            QUERY_DURATION.labels(endpoint=metric.endpoint).observe(metric.duration)
            MEMORY_USAGE.labels(component="api").set(metric.memory_usage)
            
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        try:
            if not self.metrics_buffer:
                return {"error": "No metrics available"}
            
            # Calculate statistics
            durations = [m.duration for m in self.metrics_buffer]
            memory_usage = [m.memory_usage for m in self.metrics_buffer]
            
            stats = {
                "total_queries": len(self.metrics_buffer),
                "average_duration": np.mean(durations),
                "median_duration": np.median(durations),
                "p95_duration": np.percentile(durations, 95),
                "p99_duration": np.percentile(durations, 99),
                "average_memory": np.mean(memory_usage),
                "max_memory": np.max(memory_usage),
                "active_queries": self.active_queries,
                "error_rate": len([m for m in self.metrics_buffer if m.error]) / len(self.metrics_buffer) * 100
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {"error": str(e)}
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used
            memory_total = memory.total
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used = disk.used
            disk_total = disk.total
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count
                },
                "memory": {
                    "percent": memory_percent,
                    "used": memory_used,
                    "total": memory_total
                },
                "disk": {
                    "percent": disk_percent,
                    "used": disk_used,
                    "total": disk_total
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e)}


# Global instances
cache = PerformanceCache()
optimizer = QueryOptimizer()
monitor = PerformanceMonitor()


def performance_tracking(endpoint: str):
    """Decorator for performance tracking."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            monitor.active_queries += 1
            CONCURRENT_QUERIES.set(monitor.active_queries)
            
            try:
                result = await func(*args, **kwargs)
                error = None
                
                return result
                
            except Exception as e:
                error = str(e)
                raise
                
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                duration = end_time - start_time
                memory_usage = end_memory - start_memory
                
                metric = PerformanceMetrics(
                    timestamp=datetime.now(),
                    endpoint=endpoint,
                    duration=duration,
                    memory_usage=memory_usage,
                    cpu_usage=psutil.cpu_percent(),
                    cache_hit=False,  # Would be set by cache logic
                    error=error
                )
                
                await monitor.record_metric(metric)
                monitor.active_queries -= 1
                CONCURRENT_QUERIES.set(monitor.active_queries)
        
        return wrapper
    return decorator


@asynccontextmanager
async def performance_context(operation: str):
    """Context manager for performance tracking."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        duration = end_time - start_time
        memory_usage = end_memory - start_memory
        
        metric = PerformanceMetrics(
            timestamp=datetime.now(),
            endpoint=operation,
            duration=duration,
            memory_usage=memory_usage,
            cpu_usage=psutil.cpu_percent(),
            cache_hit=False
        )
        
        await monitor.record_metric(metric)


async def initialize_performance_system():
    """Initialize the performance system."""
    await cache.initialize()
    await optimizer.initialize()
    logger.info("Performance system initialized")


async def cleanup_performance_system():
    """Cleanup the performance system."""
    # Cleanup logic here
    logger.info("Performance system cleanup completed")


# Performance optimization utilities
async def optimize_database_queries(db: AsyncSession):
    """Optimize database queries."""
    try:
        # Analyze query performance
        analysis_query = """
        SELECT 
            query,
            calls,
            total_time,
            mean_time,
            rows
        FROM pg_stat_statements
        ORDER BY total_time DESC
        LIMIT 20;
        """
        
        result = await db.execute(text(analysis_query))
        slow_queries = result.fetchall()
        
        optimizations = []
        
        for query in slow_queries:
            if query.mean_time > 100:  # Queries taking more than 100ms
                optimizations.append({
                    "query": query.query[:100] + "...",
                    "mean_time": query.mean_time,
                    "calls": query.calls,
                    "recommendation": "Consider adding indexes or optimizing query structure"
                })
        
        return {
            "slow_queries": len(slow_queries),
            "optimizations": optimizations
        }
        
    except Exception as e:
        logger.error(f"Database optimization analysis failed: {e}")
        return {"error": str(e)}


async def optimize_vector_search():
    """Optimize vector search performance."""
    try:
        # Mock optimization recommendations
        recommendations = [
            "Consider using HNSW index for better performance",
            "Increase vector dimensions for better accuracy",
            "Use batch processing for multiple queries",
            "Consider quantization for memory efficiency"
        ]
        
        return {
            "current_performance": {
                "average_search_time": 0.045,
                "index_size": "2.5GB",
                "memory_usage": "1.8GB"
            },
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Vector search optimization failed: {e}")
        return {"error": str(e)}
