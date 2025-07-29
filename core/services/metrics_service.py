"""
Prometheus Metrics Service
Provides comprehensive metrics collection for RAG system monitoring
"""
import logging
import time
import psutil
from typing import Dict, Any, Optional
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

# Try to import Prometheus dependencies
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info, CollectorRegistry, 
        generate_latest, CONTENT_TYPE_LATEST, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available, metrics collection disabled")

class MetricsService:
    """Handles Prometheus metrics collection and exposure"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics service"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available, metrics service disabled")
            self.enabled = False
            return
        
        self.enabled = True
        self.registry = registry or CollectorRegistry()
        self._lock = Lock()
        
        # Initialize metrics
        self._init_application_metrics()
        self._init_rag_metrics()
        self._init_system_metrics()
        
        logger.info("Prometheus metrics service initialized")
    
    def _init_application_metrics(self):
        """Initialize general application metrics"""
        if not self.enabled:
            return
        
        # HTTP Request metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Database metrics
        self.db_operations_total = Counter(
            'database_operations_total',
            'Total database operations',
            ['operation', 'table'],
            registry=self.registry
        )
        
        self.db_operation_duration = Histogram(
            'database_operation_duration_seconds',
            'Database operation duration',
            ['operation', 'table'],
            registry=self.registry
        )
        
        self.db_connections_active = Gauge(
            'database_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_operations_total = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation', 'result'],
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'cache_hit_rate',
            'Cache hit rate',
            registry=self.registry
        )
    
    def _init_rag_metrics(self):
        """Initialize RAG-specific metrics"""
        if not self.enabled:
            return
        
        # Document processing metrics
        self.documents_processed_total = Counter(
            'documents_processed_total',
            'Total documents processed',
            ['status', 'type'],
            registry=self.registry
        )
        
        self.document_processing_duration = Histogram(
            'document_processing_duration_seconds',
            'Document processing duration',
            ['type'],
            registry=self.registry
        )
        
        self.documents_total = Gauge(
            'documents_total',
            'Total documents in system',
            ['tenant'],
            registry=self.registry
        )
        
        # Query processing metrics
        self.queries_total = Counter(
            'queries_total',
            'Total queries processed',
            ['tenant', 'status'],
            registry=self.registry
        )
        
        self.query_duration = Histogram(
            'query_duration_seconds',
            'Query processing duration',
            ['component'],
            registry=self.registry,
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.query_relevance_score = Histogram(
            'query_relevance_score',
            'Query relevance scores',
            registry=self.registry,
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Vector search metrics
        self.vector_search_operations = Counter(
            'vector_search_operations_total',
            'Total vector search operations',
            ['status'],
            registry=self.registry
        )
        
        self.vector_search_duration = Histogram(
            'vector_search_duration_seconds',
            'Vector search duration',
            registry=self.registry
        )
        
        self.vector_index_size = Gauge(
            'vector_index_size',
            'Vector index size',
            registry=self.registry
        )
        
        # LLM metrics
        self.llm_requests_total = Counter(
            'llm_requests_total',
            'Total LLM requests',
            ['model', 'status'],
            registry=self.registry
        )
        
        self.llm_request_duration = Histogram(
            'llm_request_duration_seconds',
            'LLM request duration',
            ['model'],
            registry=self.registry,
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0]
        )
        
        self.llm_token_count = Counter(
            'llm_tokens_total',
            'Total LLM tokens',
            ['model', 'type'],
            registry=self.registry
        )
    
    def _init_system_metrics(self):
        """Initialize system resource metrics"""
        if not self.enabled:
            return
        
        # System resource metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'system_disk_usage_bytes',
            'System disk usage in bytes',
            ['mountpoint', 'type'],
            registry=self.registry
        )
        
        # Application info
        self.app_info = Info(
            'application_info',
            'Application information',
            registry=self.registry
        )
        
        self.app_uptime = Gauge(
            'application_uptime_seconds',
            'Application uptime in seconds',
            registry=self.registry
        )
        
        # Set application info
        self.app_info.info({
            'version': '2.0',
            'name': 'RAG System',
            'python_version': str(psutil.sys.version_info[:3])
        })
        
        self.start_time = time.time()
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        if not self.enabled:
            return
        
        with self._lock:
            self.http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=str(status_code)
            ).inc()
            
            self.http_request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
    
    def record_db_operation(self, operation: str, table: str, duration: float):
        """Record database operation metrics"""
        if not self.enabled:
            return
        
        with self._lock:
            self.db_operations_total.labels(
                operation=operation,
                table=table
            ).inc()
            
            self.db_operation_duration.labels(
                operation=operation,
                table=table
            ).observe(duration)
    
    def set_db_connections(self, count: int):
        """Set active database connections count"""
        if not self.enabled:
            return
        
        self.db_connections_active.set(count)
    
    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache operation metrics"""
        if not self.enabled:
            return
        
        result = 'hit' if hit else 'miss'
        self.cache_operations_total.labels(
            operation=operation,
            result=result
        ).inc()
    
    def set_cache_hit_rate(self, rate: float):
        """Set cache hit rate"""
        if not self.enabled:
            return
        
        self.cache_hit_rate.set(rate)
    
    def record_document_processing(self, doc_type: str, status: str, duration: float):
        """Record document processing metrics"""
        if not self.enabled:
            return
        
        with self._lock:
            self.documents_processed_total.labels(
                status=status,
                type=doc_type
            ).inc()
            
            if duration > 0:
                self.document_processing_duration.labels(
                    type=doc_type
                ).observe(duration)
    
    def set_document_count(self, tenant: str, count: int):
        """Set total document count for tenant"""
        if not self.enabled:
            return
        
        self.documents_total.labels(tenant=tenant).set(count)
    
    def record_query(self, tenant: str, status: str, component_durations: Dict[str, float]):
        """Record query processing metrics"""
        if not self.enabled:
            return
        
        with self._lock:
            self.queries_total.labels(
                tenant=tenant,
                status=status
            ).inc()
            
            # Record component durations
            for component, duration in component_durations.items():
                self.query_duration.labels(component=component).observe(duration)
    
    def record_query_relevance(self, score: float):
        """Record query relevance score"""
        if not self.enabled:
            return
        
        self.query_relevance_score.observe(score)
    
    def record_vector_search(self, status: str, duration: float):
        """Record vector search metrics"""
        if not self.enabled:
            return
        
        with self._lock:
            self.vector_search_operations.labels(status=status).inc()
            if duration > 0:
                self.vector_search_duration.observe(duration)
    
    def set_vector_index_size(self, size: int):
        """Set vector index size"""
        if not self.enabled:
            return
        
        self.vector_index_size.set(size)
    
    def record_llm_request(self, model: str, status: str, duration: float, 
                          input_tokens: int = 0, output_tokens: int = 0):
        """Record LLM request metrics"""
        if not self.enabled:
            return
        
        with self._lock:
            self.llm_requests_total.labels(
                model=model,
                status=status
            ).inc()
            
            if duration > 0:
                self.llm_request_duration.labels(model=model).observe(duration)
            
            if input_tokens > 0:
                self.llm_token_count.labels(
                    model=model,
                    type='input'
                ).inc(input_tokens)
            
            if output_tokens > 0:
                self.llm_token_count.labels(
                    model=model,
                    type='output'
                ).inc(output_tokens)
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        if not self.enabled:
            return
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.labels(type='total').set(memory.total)
            self.memory_usage.labels(type='used').set(memory.used)
            self.memory_usage.labels(type='available').set(memory.available)
            
            # Disk usage
            for partition in psutil.disk_partitions():
                try:
                    disk_usage = psutil.disk_usage(partition.mountpoint)
                    self.disk_usage.labels(
                        mountpoint=partition.mountpoint,
                        type='total'
                    ).set(disk_usage.total)
                    self.disk_usage.labels(
                        mountpoint=partition.mountpoint,
                        type='used'
                    ).set(disk_usage.used)
                    self.disk_usage.labels(
                        mountpoint=partition.mountpoint,
                        type='free'
                    ).set(disk_usage.free)
                except PermissionError:
                    # Skip partitions we can't access
                    continue
            
            # Application uptime
            uptime = time.time() - self.start_time
            self.app_uptime.set(uptime)
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        if not self.enabled:
            return "# Prometheus metrics not available\n"
        
        # Update system metrics before export
        self.update_system_metrics()
        
        return generate_latest(self.registry).decode('utf-8')
    
    def get_content_type(self) -> str:
        """Get Prometheus content type"""
        return CONTENT_TYPE_LATEST
    
    def start_metrics_server(self, port: int = 8001, addr: str = '0.0.0.0'):
        """Start standalone metrics server"""
        if not self.enabled:
            logger.warning("Cannot start metrics server - Prometheus not available")
            return
        
        try:
            start_http_server(port, addr, registry=self.registry)
            logger.info(f"Prometheus metrics server started on {addr}:{port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get a summary of key metrics for health checks"""
        if not self.enabled:
            return {"metrics_enabled": False}
        
        try:
            # Get basic system stats
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            return {
                "metrics_enabled": True,
                "uptime_seconds": time.time() - self.start_time,
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": round(memory.available / (1024**3), 2)
                },
                "registry_collectors": len(self.registry._collector_to_names)
            }
        except Exception as e:
            logger.error(f"Failed to get stats summary: {e}")
            return {"metrics_enabled": True, "error": str(e)}

# Global metrics service instance
_metrics_service: Optional[MetricsService] = None

def get_metrics_service() -> MetricsService:
    """Get global metrics service instance"""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service

def init_metrics_service(registry: Optional[CollectorRegistry] = None) -> MetricsService:
    """Initialize global metrics service"""
    global _metrics_service
    _metrics_service = MetricsService(registry)
    return _metrics_service