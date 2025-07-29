# RAG System Monitoring Setup

This directory contains the complete monitoring stack for the RAG System using Prometheus, Grafana, and AlertManager.

## Components

### 1. Prometheus (`prometheus.yml`)
- Scrapes metrics from the RAG System API at `/metrics`
- Collects system metrics via node_exporter
- Stores metrics with 30-day retention
- Available at: http://localhost:9090

### 2. Grafana (`grafana/`)
- Visualizes metrics through dashboards
- Pre-configured with RAG System Overview dashboard
- Default credentials: admin/admin123
- Available at: http://localhost:3000

### 3. AlertManager (`alertmanager.yml`)
- Handles alerts from Prometheus
- Configured for email and webhook notifications
- Available at: http://localhost:9093

### 4. Node Exporter
- Provides system-level metrics (CPU, memory, disk)
- Available at: http://localhost:9100

## Quick Start

### 1. Start Monitoring Stack
```bash
# Start all monitoring services
cd deployment/monitoring
docker-compose -f docker-compose.monitoring.yml up -d

# Check service status
docker-compose -f docker-compose.monitoring.yml ps
```

### 2. Start RAG System
```bash
# In the main directory
python run_core.py
```

### 3. Access Dashboards
- **Grafana Dashboard**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093

### 4. View Metrics
The RAG System exposes metrics at:
- **Main metrics**: http://localhost:8000/metrics
- **Health check**: http://localhost:8000/metrics/health
- **Statistics**: http://localhost:8000/metrics/stats

## Key Metrics

### HTTP Metrics
- `http_requests_total` - Total HTTP requests by method, endpoint, status
- `http_request_duration_seconds` - Request duration histogram

### RAG Metrics
- `queries_total` - Total queries processed
- `query_duration_seconds` - Query processing time by component
- `query_relevance_score` - Distribution of relevance scores
- `documents_total` - Total documents in system by tenant
- `documents_processed_total` - Document processing metrics

### LLM Metrics
- `llm_requests_total` - LLM requests by model and status
- `llm_request_duration_seconds` - LLM response times
- `llm_tokens_total` - Token usage statistics

### System Metrics
- `system_cpu_usage_percent` - CPU usage
- `system_memory_usage_bytes` - Memory usage
- `system_disk_usage_bytes` - Disk usage
- `application_uptime_seconds` - Application uptime

### Database Metrics
- `database_operations_total` - Database operations
- `database_operation_duration_seconds` - Database operation times
- `database_connections_active` - Active database connections

## Alerting

### Pre-configured Alerts
- **High Error Rate**: >10% HTTP 5xx responses
- **High Response Time**: >5s 95th percentile
- **High CPU Usage**: >80% for 5 minutes
- **High Memory Usage**: >85% for 5 minutes
- **LLM Timeouts**: >5 timeouts in 10 minutes
- **Service Down**: Service unreachable

### Alert Configuration
Edit `alertmanager.yml` to configure:
- Email notifications
- Webhook endpoints
- Slack integration (add webhook receiver)

## Dashboard Customization

### Adding Custom Panels
1. Access Grafana at http://localhost:3000
2. Navigate to the RAG System Overview dashboard
3. Click "Add Panel" to create custom visualizations
4. Use PromQL queries to select metrics

### Example Queries
```promql
# Request rate by endpoint
rate(http_requests_total[5m])

# 95th percentile response time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Memory usage percentage
(system_memory_usage_bytes{type="used"} / system_memory_usage_bytes{type="total"}) * 100
```

## Production Considerations

### Security
1. Change default Grafana password
2. Configure proper CORS origins
3. Use HTTPS for all services
4. Restrict network access to monitoring ports

### Scaling
1. Increase Prometheus retention for longer history
2. Configure remote storage for large deployments
3. Use Grafana organizations for multi-tenant monitoring

### Backup
```bash
# Backup Prometheus data
docker exec rag-prometheus tar -czf /prometheus-backup.tar.gz /prometheus

# Backup Grafana dashboards
docker exec rag-grafana tar -czf /grafana-backup.tar.gz /var/lib/grafana
```

## Troubleshooting

### Metrics Not Appearing
1. Check RAG System is running: `curl http://localhost:8000/metrics/health`
2. Verify Prometheus targets: http://localhost:9090/targets
3. Check Prometheus logs: `docker logs rag-prometheus`

### Grafana Issues
1. Check datasource connection in Grafana
2. Verify Prometheus is accessible from Grafana container
3. Check Grafana logs: `docker logs rag-grafana`

### AlertManager Not Sending Alerts
1. Check AlertManager configuration: http://localhost:9093/#/status
2. Verify SMTP settings in `alertmanager.yml`
3. Test webhook endpoints manually

## Advanced Configuration

### Custom Metrics
Add custom metrics in your RAG System code:
```python
from core.middleware.metrics_middleware import get_query_metrics

# Record custom metrics
metrics = get_query_metrics()
context = metrics.record_query_start("tenant1")
metrics.record_component_start(context, "document_search")
# ... processing ...
metrics.record_component_end(context, "document_search")
metrics.record_query_end(context, "success", 0.85)
```

### Integration with External Systems
- Configure Prometheus to scrape additional services
- Add custom Grafana datasources
- Set up AlertManager integrations (PagerDuty, OpsGenie)