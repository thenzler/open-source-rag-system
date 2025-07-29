"""
Metrics Collection Middleware
Automatically collects HTTP request metrics and integrates with Prometheus
"""
import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..services.metrics_service import get_metrics_service

logger = logging.getLogger(__name__)

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting HTTP metrics"""
    
    def __init__(self, app, collect_detailed_metrics: bool = True):
        super().__init__(app)
        self.collect_detailed_metrics = collect_detailed_metrics
        self.metrics_service = get_metrics_service()
        
        logger.info(f"Metrics middleware initialized (detailed: {collect_detailed_metrics})")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics"""
        
        # Skip metrics for metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)
        
        start_time = time.time()
        
        # Extract request information
        method = request.method
        path = request.url.path
        
        # Normalize endpoint paths to avoid high cardinality
        endpoint = self._normalize_endpoint(path)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            if self.metrics_service.enabled:
                self.metrics_service.record_http_request(
                    method=method,
                    endpoint=endpoint,
                    status_code=response.status_code,
                    duration=duration
                )
            
            # Add metrics headers if detailed collection is enabled
            if self.collect_detailed_metrics:
                response.headers["X-Response-Time"] = f"{duration:.3f}s"
                response.headers["X-Request-ID"] = getattr(request.state, 'request_id', 'unknown')
            
            return response
            
        except Exception as e:
            # Calculate duration even for errors
            duration = time.time() - start_time
            
            # Record error metrics
            if self.metrics_service.enabled:
                self.metrics_service.record_http_request(
                    method=method,
                    endpoint=endpoint,
                    status_code=500,
                    duration=duration
                )
            
            logger.error(f"Request failed: {method} {path} - {e}")
            raise
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path to reduce cardinality"""
        
        # Remove query parameters
        if '?' in path:
            path = path.split('?')[0]
        
        # Normalize common patterns
        normalized_patterns = [
            # Document IDs
            (r'/api/v\d+/documents/\d+', '/api/v*/documents/{id}'),
            (r'/api/v\d+/documents/[a-zA-Z0-9_-]+', '/api/v*/documents/{id}'),
            
            # Query endpoints with IDs
            (r'/api/v\d+/queries/\d+', '/api/v*/queries/{id}'),
            
            # Tenant-specific endpoints
            (r'/tenant/[a-zA-Z0-9_-]+/', '/tenant/{slug}/'),
            
            # File downloads
            (r'/download/[a-zA-Z0-9_.-]+', '/download/{filename}'),
            
            # Static files
            (r'/static/.+', '/static/{file}'),
            
            # Admin endpoints with IDs
            (r'/admin/.+/\d+', '/admin/{resource}/{id}'),
        ]
        
        import re
        for pattern, replacement in normalized_patterns:
            if re.match(pattern, path):
                return replacement
        
        # For API endpoints, keep the structure but limit depth
        if path.startswith('/api/'):
            parts = path.split('/')
            if len(parts) > 4:  # /api/v1/resource/...
                return '/'.join(parts[:4]) + '/{...}'
        
        return path

class QueryMetricsCollector:
    """Specialized metrics collector for RAG queries"""
    
    def __init__(self):
        self.metrics_service = get_metrics_service()
    
    def record_query_start(self, tenant: str = "default"):
        """Record query start"""
        return {
            'start_time': time.time(),
            'tenant': tenant,
            'components': {}
        }
    
    def record_component_start(self, context: dict, component: str):
        """Record component processing start"""
        context['components'][component] = {'start_time': time.time()}
    
    def record_component_end(self, context: dict, component: str):
        """Record component processing end"""
        if component in context['components']:
            duration = time.time() - context['components'][component]['start_time']
            context['components'][component]['duration'] = duration
    
    def record_query_end(self, context: dict, status: str = "success", relevance_score: float = None):
        """Record query completion"""
        if not self.metrics_service.enabled:
            return
        
        # Calculate component durations
        component_durations = {}
        for component, data in context['components'].items():
            if 'duration' in data:
                component_durations[component] = data['duration']
        
        # Record query metrics
        self.metrics_service.record_query(
            tenant=context['tenant'],
            status=status,
            component_durations=component_durations
        )
        
        # Record relevance score if available
        if relevance_score is not None:
            self.metrics_service.record_query_relevance(relevance_score)

class DatabaseMetricsCollector:
    """Specialized metrics collector for database operations"""
    
    def __init__(self):
        self.metrics_service = get_metrics_service()
    
    def record_operation(self, operation: str, table: str, duration: float):
        """Record database operation"""
        if self.metrics_service.enabled:
            self.metrics_service.record_db_operation(operation, table, duration)
    
    def update_connection_count(self, count: int):
        """Update active connection count"""
        if self.metrics_service.enabled:
            self.metrics_service.set_db_connections(count)

class DocumentMetricsCollector:
    """Specialized metrics collector for document operations"""
    
    def __init__(self):
        self.metrics_service = get_metrics_service()
    
    def record_processing(self, doc_type: str, status: str, duration: float = 0):
        """Record document processing"""
        if self.metrics_service.enabled:
            self.metrics_service.record_document_processing(doc_type, status, duration)
    
    def update_document_count(self, tenant: str, count: int):
        """Update document count for tenant"""
        if self.metrics_service.enabled:
            self.metrics_service.set_document_count(tenant, count)

class LLMMetricsCollector:
    """Specialized metrics collector for LLM operations"""
    
    def __init__(self):
        self.metrics_service = get_metrics_service()
    
    def record_request(self, model: str, status: str, duration: float, 
                      input_tokens: int = 0, output_tokens: int = 0):
        """Record LLM request"""
        if self.metrics_service.enabled:
            self.metrics_service.record_llm_request(
                model=model,
                status=status,
                duration=duration,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

# Global collectors
_query_collector: QueryMetricsCollector = None
_db_collector: DatabaseMetricsCollector = None
_doc_collector: DocumentMetricsCollector = None
_llm_collector: LLMMetricsCollector = None

def get_query_metrics() -> QueryMetricsCollector:
    """Get query metrics collector"""
    global _query_collector
    if _query_collector is None:
        _query_collector = QueryMetricsCollector()
    return _query_collector

def get_db_metrics() -> DatabaseMetricsCollector:
    """Get database metrics collector"""
    global _db_collector
    if _db_collector is None:
        _db_collector = DatabaseMetricsCollector()
    return _db_collector

def get_doc_metrics() -> DocumentMetricsCollector:
    """Get document metrics collector"""
    global _doc_collector
    if _doc_collector is None:
        _doc_collector = DocumentMetricsCollector()
    return _doc_collector

def get_llm_metrics() -> LLMMetricsCollector:
    """Get LLM metrics collector"""
    global _llm_collector
    if _llm_collector is None:
        _llm_collector = LLMMetricsCollector()
    return _llm_collector