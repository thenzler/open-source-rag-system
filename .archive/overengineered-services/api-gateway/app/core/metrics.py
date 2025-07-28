"""
Prometheus Metrics and Monitoring for RAG System
"""

import time
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CollectorRegistry
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
from fastapi import FastAPI, Response
from fastapi.responses import Response as FastAPIResponse
import logging
from datetime import datetime
import asyncio
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create custom registry for the RAG system
REGISTRY = CollectorRegistry()

# ============================================================================
# PROMETHEUS METRICS DEFINITIONS
# ============================================================================

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=REGISTRY
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=REGISTRY
)

# Document processing metrics
documents_uploaded_total = Counter(
    'documents_uploaded_total',
    'Total documents uploaded',
    ['file_type', 'status'],
    registry=REGISTRY
)

documents_processed_total = Counter(
    'documents_processed_total',
    'Total documents processed',
    ['file_type', 'status'],
    registry=REGISTRY
)

document_processing_duration_seconds = Histogram(
    'document_processing_duration_seconds',
    'Document processing duration in seconds',
    ['file_type'],
    registry=REGISTRY
)

document_size_bytes = Histogram(
    'document_size_bytes',
    'Document size in bytes',
    ['file_type'],
    buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600],  # 1KB to 100MB
    registry=REGISTRY
)

documents_active_gauge = Gauge(
    'documents_active',
    'Number of active documents',
    ['status'],
    registry=REGISTRY
)

# Query metrics
queries_total = Counter(
    'queries_total',
    'Total queries performed',
    ['status'],
    registry=REGISTRY
)

query_duration_seconds = Histogram(
    'query_duration_seconds',
    'Query processing duration in seconds',
    registry=REGISTRY
)

query_results_count = Histogram(
    'query_results_count',
    'Number of results returned per query',
    buckets=[0, 1, 5, 10, 20, 50, 100],
    registry=REGISTRY
)

query_vector_search_duration_seconds = Histogram(
    'query_vector_search_duration_seconds',
    'Vector search duration in seconds',
    registry=REGISTRY
)

# Vector database metrics
vector_embeddings_total = Counter(
    'vector_embeddings_total',
    'Total embeddings generated',
    ['model'],
    registry=REGISTRY
)

vector_embedding_duration_seconds = Histogram(
    'vector_embedding_duration_seconds',
    'Embedding generation duration in seconds',
    ['model'],
    registry=REGISTRY
)

vector_storage_operations_total = Counter(
    'vector_storage_operations_total',
    'Total vector storage operations',
    ['operation', 'status'],
    registry=REGISTRY
)

vector_db_size_gauge = Gauge(
    'vector_db_size',
    'Vector database size (number of vectors)',
    registry=REGISTRY
)

# LLM metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'operation', 'status'],
    registry=REGISTRY
)

llm_request_duration_seconds = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration in seconds',
    ['model', 'operation'],
    registry=REGISTRY
)

llm_tokens_processed_total = Counter(
    'llm_tokens_processed_total',
    'Total tokens processed by LLM',
    ['model', 'type'],  # type: input/output
    registry=REGISTRY
)

# System metrics
system_health_gauge = Gauge(
    'system_health',
    'System health status (1=healthy, 0=unhealthy)',
    ['service'],
    registry=REGISTRY
)

active_users_gauge = Gauge(
    'active_users',
    'Number of active users',
    ['time_window'],  # 1h, 24h, 7d
    registry=REGISTRY
)

storage_usage_bytes = Gauge(
    'storage_usage_bytes',
    'Storage usage in bytes',
    ['type'],  # documents, vectors, cache
    registry=REGISTRY
)

# Error metrics
errors_total = Counter(
    'errors_total',
    'Total errors',
    ['service', 'error_type'],
    registry=REGISTRY
)

# Cache metrics
cache_operations_total = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'status'],  # operation: hit/miss/set/delete
    registry=REGISTRY
)

cache_size_gauge = Gauge(
    'cache_size',
    'Cache size (number of entries)',
    registry=REGISTRY
)

# Application info
app_info = Info(
    'app_info',
    'Application information',
    registry=REGISTRY
)

# ============================================================================
# METRICS COLLECTION UTILITIES
# ============================================================================

class MetricsCollector:
    """Centralized metrics collection and reporting."""
    
    def __init__(self):
        self.start_time = time.time()
        self._setup_app_info()
    
    def _setup_app_info(self):
        """Set up application information metrics."""
        app_info.info({
            'version': '1.0.0',
            'environment': 'development',
            'build_date': datetime.now().isoformat(),
            'python_version': '3.11'
        })
    
    # HTTP Request Metrics
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        http_requests_total.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
    
    # Document Metrics
    def record_document_upload(self, file_type: str, status: str = 'success'):
        """Record document upload metrics."""
        documents_uploaded_total.labels(file_type=file_type, status=status).inc()
    
    def record_document_processing(self, file_type: str, status: str, duration: float, size_bytes: int):
        """Record document processing metrics."""
        documents_processed_total.labels(file_type=file_type, status=status).inc()
        document_processing_duration_seconds.labels(file_type=file_type).observe(duration)
        document_size_bytes.labels(file_type=file_type).observe(size_bytes)
    
    def update_active_documents(self, status_counts: Dict[str, int]):
        """Update active documents gauge."""
        for status, count in status_counts.items():
            documents_active_gauge.labels(status=status).set(count)
    
    # Query Metrics
    def record_query(self, status: str, duration: float, results_count: int):
        """Record query metrics."""
        queries_total.labels(status=status).inc()
        query_duration_seconds.observe(duration)
        query_results_count.observe(results_count)
    
    def record_vector_search(self, duration: float):
        """Record vector search metrics."""
        query_vector_search_duration_seconds.observe(duration)
    
    # Vector Database Metrics
    def record_embedding_generation(self, model: str, duration: float, count: int = 1):
        """Record embedding generation metrics."""
        vector_embeddings_total.labels(model=model).inc(count)
        vector_embedding_duration_seconds.labels(model=model).observe(duration)
    
    def record_vector_storage(self, operation: str, status: str, count: int = 1):
        """Record vector storage operations."""
        vector_storage_operations_total.labels(operation=operation, status=status).inc(count)
    
    def update_vector_db_size(self, size: int):
        """Update vector database size."""
        vector_db_size_gauge.set(size)
    
    # LLM Metrics
    def record_llm_request(self, model: str, operation: str, status: str, duration: float, 
                          input_tokens: int = 0, output_tokens: int = 0):
        """Record LLM request metrics."""
        llm_requests_total.labels(model=model, operation=operation, status=status).inc()
        llm_request_duration_seconds.labels(model=model, operation=operation).observe(duration)
        
        if input_tokens > 0:
            llm_tokens_processed_total.labels(model=model, type='input').inc(input_tokens)
        if output_tokens > 0:
            llm_tokens_processed_total.labels(model=model, type='output').inc(output_tokens)
    
    # System Metrics
    def update_system_health(self, service: str, healthy: bool):
        """Update system health metrics."""
        system_health_gauge.labels(service=service).set(1 if healthy else 0)
    
    def update_active_users(self, time_window: str, count: int):
        """Update active users metrics."""
        active_users_gauge.labels(time_window=time_window).set(count)
    
    def update_storage_usage(self, storage_type: str, bytes_used: int):
        """Update storage usage metrics."""
        storage_usage_bytes.labels(type=storage_type).set(bytes_used)
    
    # Error Metrics
    def record_error(self, service: str, error_type: str):
        """Record error metrics."""
        errors_total.labels(service=service, error_type=error_type).inc()
    
    # Cache Metrics
    def record_cache_operation(self, operation: str, status: str):
        """Record cache operation metrics."""
        cache_operations_total.labels(operation=operation, status=status).inc()
    
    def update_cache_size(self, size: int):
        """Update cache size metrics."""
        cache_size_gauge.set(size)
    
    # Utility Methods
    def get_uptime(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self.start_time


# Global metrics collector instance
metrics_collector = MetricsCollector()

# ============================================================================
# DECORATORS FOR AUTOMATIC METRICS COLLECTION
# ============================================================================

def monitor_http_requests(func):
    """Decorator to monitor HTTP requests."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        method = kwargs.get('method', 'GET')
        endpoint = kwargs.get('endpoint', 'unknown')
        status_code = 200
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            status_code = getattr(e, 'status_code', 500)
            raise
        finally:
            duration = time.time() - start_time
            metrics_collector.record_http_request(method, endpoint, status_code, duration)
    
    return wrapper


def monitor_document_processing(func):
    """Decorator to monitor document processing."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        file_type = kwargs.get('file_type', 'unknown')
        status = 'success'
        size_bytes = kwargs.get('size_bytes', 0)
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            status = 'failed'
            metrics_collector.record_error('document_processor', type(e).__name__)
            raise
        finally:
            duration = time.time() - start_time
            metrics_collector.record_document_processing(file_type, status, duration, size_bytes)
    
    return wrapper


def monitor_queries(func):
    """Decorator to monitor queries."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        status = 'success'
        results_count = 0
        
        try:
            result = await func(*args, **kwargs)
            results_count = result.get('total_results', 0) if isinstance(result, dict) else 0
            return result
        except Exception as e:
            status = 'failed'
            metrics_collector.record_error('query_service', type(e).__name__)
            raise
        finally:
            duration = time.time() - start_time
            metrics_collector.record_query(status, duration, results_count)
    
    return wrapper


def monitor_vector_operations(operation_type: str):
    """Decorator to monitor vector operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'failed'
                metrics_collector.record_error('vector_engine', type(e).__name__)
                raise
            finally:
                duration = time.time() - start_time
                if operation_type == 'embedding':
                    model = kwargs.get('model', 'unknown')
                    metrics_collector.record_embedding_generation(model, duration)
                elif operation_type == 'storage':
                    metrics_collector.record_vector_storage('store', status)
        
        return wrapper
    return decorator


def monitor_llm_requests(operation: str):
    """Decorator to monitor LLM requests."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            model = kwargs.get('model', 'unknown')
            status = 'success'
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'failed'
                metrics_collector.record_error('llm_service', type(e).__name__)
                raise
            finally:
                duration = time.time() - start_time
                metrics_collector.record_llm_request(model, operation, status, duration)
        
        return wrapper
    return decorator


# ============================================================================
# FASTAPI MIDDLEWARE FOR AUTOMATIC METRICS
# ============================================================================

class MetricsMiddleware:
    """Middleware to automatically collect HTTP metrics."""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        method = scope["method"]
        path = scope["path"]
        status_code = 200
        
        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            status_code = getattr(e, 'status_code', 500)
            raise
        finally:
            duration = time.time() - start_time
            metrics_collector.record_http_request(method, path, status_code, duration)


# ============================================================================
# METRICS ENDPOINTS
# ============================================================================

def setup_metrics_endpoints(app: FastAPI):
    """Set up metrics endpoints."""
    
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint."""
        return FastAPIResponse(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST
        )
    
    @app.get("/health/metrics")
    async def get_health_metrics():
        """Health metrics endpoint."""
        return {
            "uptime_seconds": metrics_collector.get_uptime(),
            "timestamp": datetime.now().isoformat(),
            "metrics_enabled": True
        }


# ============================================================================
# BACKGROUND METRICS COLLECTION
# ============================================================================

class SystemMetricsCollector:
    """Collects system-level metrics in the background."""
    
    def __init__(self):
        self.running = False
        self.collection_interval = 60  # seconds
    
    async def start(self):
        """Start background metrics collection."""
        self.running = True
        asyncio.create_task(self._collect_metrics_loop())
    
    async def stop(self):
        """Stop background metrics collection."""
        self.running = False
    
    async def _collect_metrics_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        # This would integrate with the analytics service
        # to collect real-time metrics
        pass
    
    async def _collect_health_metrics(self):
        """Collect health metrics."""
        # Check service health and update metrics
        services = ['api_gateway', 'document_processor', 'vector_engine', 'llm_service']
        
        for service in services:
            # Mock health check - in reality, this would check actual service health
            healthy = True  # Replace with actual health check
            metrics_collector.update_system_health(service, healthy)
    
    async def _collect_storage_metrics(self):
        """Collect storage metrics."""
        # Mock storage metrics - in reality, this would check actual storage
        metrics_collector.update_storage_usage('documents', 1024 * 1024 * 100)  # 100MB
        metrics_collector.update_storage_usage('vectors', 1024 * 1024 * 50)     # 50MB
        metrics_collector.update_storage_usage('cache', 1024 * 1024 * 10)       # 10MB
    
    async def _collect_user_metrics(self):
        """Collect user activity metrics."""
        # Mock user metrics - in reality, this would query the database
        metrics_collector.update_active_users('1h', 5)
        metrics_collector.update_active_users('24h', 25)
        metrics_collector.update_active_users('7d', 100)


# Global system metrics collector
system_metrics_collector = SystemMetricsCollector()

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_metrics():
    """Initialize metrics collection."""
    logger.info("Initializing Prometheus metrics...")
    
    # Set initial values
    metrics_collector.update_system_health('api_gateway', True)
    metrics_collector.update_vector_db_size(0)
    metrics_collector.update_cache_size(0)
    
    logger.info("Prometheus metrics initialized successfully")


def setup_metrics_for_app(app: FastAPI):
    """Set up metrics for FastAPI application."""
    
    # Initialize metrics
    initialize_metrics()
    
    # Add middleware
    app.add_middleware(MetricsMiddleware)
    
    # Add metrics endpoints
    setup_metrics_endpoints(app)
    
    # Start background collection
    @app.on_event("startup")
    async def startup_metrics():
        await system_metrics_collector.start()
    
    @app.on_event("shutdown")
    async def shutdown_metrics():
        await system_metrics_collector.stop()
    
    logger.info("Metrics setup complete for FastAPI application")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of current metrics."""
    return {
        "uptime_seconds": metrics_collector.get_uptime(),
        "total_requests": http_requests_total._value.sum(),
        "total_documents": documents_uploaded_total._value.sum(),
        "total_queries": queries_total._value.sum(),
        "total_embeddings": vector_embeddings_total._value.sum(),
        "total_errors": errors_total._value.sum(),
        "timestamp": datetime.now().isoformat()
    }


def reset_metrics():
    """Reset all metrics (for testing)."""
    logger.warning("Resetting all metrics")
    
    # Clear all metrics
    for collector in REGISTRY._collector_to_names.keys():
        if hasattr(collector, '_metrics'):
            collector._metrics.clear()
    
    # Reinitialize
    initialize_metrics()


# Export key components
__all__ = [
    'metrics_collector',
    'system_metrics_collector',
    'setup_metrics_for_app',
    'monitor_http_requests',
    'monitor_document_processing',
    'monitor_queries',
    'monitor_vector_operations',
    'monitor_llm_requests',
    'get_metrics_summary',
    'reset_metrics'
]
