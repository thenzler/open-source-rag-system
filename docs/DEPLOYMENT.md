# Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Open Source RAG System in various environments, from local development to production clusters.

## Prerequisites

### Hardware Requirements

#### Minimum Configuration
- **CPU**: 8 cores (Intel/AMD x64)
- **RAM**: 32 GB
- **Storage**: 500 GB SSD
- **Network**: 1 Gbps connection
- **GPU**: Optional (CPU inference supported)

#### Recommended Configuration
- **CPU**: 16+ cores (Intel/AMD x64)
- **RAM**: 64 GB+
- **Storage**: 2 TB NVMe SSD
- **Network**: 10 Gbps connection
- **GPU**: NVIDIA RTX 4090 or A100 (for faster inference)

### Software Requirements

#### Base System
- **OS**: Ubuntu 22.04 LTS, CentOS 8+, or Docker-compatible system
- **Docker**: 24.0+ with Docker Compose
- **Git**: 2.30+
- **Python**: 3.11+ (if running outside containers)

#### Optional for Production
- **Kubernetes**: 1.25+ (for orchestration)
- **Nginx**: Load balancing and reverse proxy
- **Prometheus/Grafana**: Monitoring and observability

## Quick Start (Docker Compose)

### 1. Clone Repository

```bash
git clone https://github.com/thenzler/open-source-rag-system.git
cd open-source-rag-system
```

### 2. Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
nano .env
```

**Required Environment Variables**:
```bash
# Database Configuration
POSTGRES_USER=raguser
POSTGRES_PASSWORD=secure_password_here
POSTGRES_DB=ragdb
DATABASE_URL=postgresql://raguser:secure_password_here@postgres:5432/ragdb

# Security
SECRET_KEY=your_super_secret_key_here_change_this
JWT_SECRET_KEY=another_secret_key_for_jwt_tokens

# Vector Database
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=optional_api_key

# LLM Configuration
LLM_SERVICE_URL=http://ollama:11434
LLM_MODEL_NAME=llama3.1:8b

# Storage
UPLOAD_DIRECTORY=/app/storage/uploads
MAX_FILE_SIZE_MB=100

# Performance
WORKER_PROCESSES=4
CHUNK_SIZE=512
CHUNK_OVERLAP=50
EMBEDDING_BATCH_SIZE=32
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Initialize System

```bash
# Create database tables
docker-compose exec api-gateway python -m scripts.init_db

# Download and start LLM model
docker-compose exec ollama ollama pull llama3.1:8b

# Check system health
curl http://localhost:8000/api/v1/health
```

### 5. Test Upload and Query

```bash
# Upload a test document
curl -X POST \"http://localhost:8000/api/v1/documents\" \
  -H \"Content-Type: multipart/form-data\" \
  -F \"file=@test_document.pdf\"

# Wait for processing, then query
curl -X POST \"http://localhost:8000/api/v1/query\" \
  -H \"Content-Type: application/json\" \
  -d '{\"query\": \"What is the main topic?\", \"top_k\": 3}'
```

## Service Architecture

### Docker Compose Services

```yaml
# docker-compose.yml structure
services:
  # Core API service
  api-gateway:
    build: ./services/api-gateway
    ports:
      - \"8000:8000\"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - QDRANT_URL=${QDRANT_URL}
    depends_on:
      - postgres
      - qdrant
      - redis

  # Document processing service  
  document-processor:
    build: ./services/document-processor
    volumes:
      - ./storage:/app/storage
    environment:
      - CELERY_BROKER=redis://redis:6379/0
      - DATABASE_URL=${DATABASE_URL}

  # Vector search service
  vector-engine:
    build: ./services/vector-engine
    environment:
      - QDRANT_URL=${QDRANT_URL}
      - EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

  # Local LLM service
  ollama:
    image: ollama/ollama:latest
    ports:
      - \"11434:11434\"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Databases
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - \"6333:6333\"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__HTTP_PORT: 6333

  redis:
    image: redis:7-alpine
    ports:
      - \"6379:6379\"
    volumes:
      - redis_data:/data

  # Optional: Web interface
  web-interface:
    build: ./services/web-interface
    ports:
      - \"3000:3000\"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
```

## Production Deployment

### 1. Kubernetes Deployment

#### Prerequisites

```bash
# Install kubectl
curl -LO \"https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl\"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

#### Namespace Setup

```bash
# Create namespace
kubectl create namespace rag-system

# Create secrets
kubectl create secret generic rag-secrets \
  --from-literal=postgres-password=secure_password \
  --from-literal=jwt-secret=jwt_secret_key \
  --namespace=rag-system
```

#### Database Deployment

```yaml
# postgres-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: rag-system
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: ragdb
        - name: POSTGRES_USER
          value: raguser
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: postgres-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: [\"ReadWriteOnce\"]
      resources:
        requests:
          storage: 100Gi
      storageClassName: fast-ssd
```

#### Vector Database Deployment

```yaml
# qdrant-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: rag-system
spec:
  serviceName: qdrant
  replicas: 3  # For clustering
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports:
        - containerPort: 6333
        - containerPort: 6334
        env:
        - name: QDRANT__CLUSTER__ENABLED
          value: \"true\"
        - name: QDRANT__CLUSTER__P2P__PORT
          value: \"6335\"
        volumeMounts:
        - name: qdrant-storage
          mountPath: /qdrant/storage
  volumeClaimTemplates:
  - metadata:
      name: qdrant-storage
    spec:
      accessModes: [\"ReadWriteOnce\"]
      resources:
        requests:
          storage: 500Gi
      storageClassName: fast-ssd
```

#### API Gateway Deployment

```yaml
# api-gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: api-gateway
        image: ragystem/api-gateway:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: postgresql://raguser:$(POSTGRES_PASSWORD)@postgres:5432/ragdb
        - name: QDRANT_URL
          value: http://qdrant:6333
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: jwt-secret
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Load Balancer and Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-ingress
  namespace: rag-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: \"true\"
    cert-manager.io/cluster-issuer: \"letsencrypt-prod\"
spec:
  tls:
  - hosts:
    - api.yourragdomain.com
    secretName: rag-tls
  rules:
  - host: api.yourragdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-gateway
            port:
              number: 8000
```

### 2. Deploy to Kubernetes

```bash
# Apply all configurations
kubectl apply -f infrastructure/kubernetes/

# Check deployment status
kubectl get pods -n rag-system
kubectl get services -n rag-system

# Check logs
kubectl logs -f deployment/api-gateway -n rag-system
```

## Scaling Configuration

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-gateway
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Pod Autoscaler

```yaml
# vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: api-gateway-vpa
  namespace: rag-system
spec:
  targetRef:
    apiVersion: \"apps/v1\"
    kind: Deployment
    name: api-gateway
  updatePolicy:
    updateMode: \"Auto\"
  resourcePolicy:
    containerPolicies:
    - containerName: api-gateway
      maxAllowed:
        cpu: 4
        memory: 8Gi
      minAllowed:
        cpu: 100m
        memory: 256Mi
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: rag-system
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'rag-api'
      static_configs:
      - targets: ['api-gateway:8000']
      metrics_path: '/metrics'
    - job_name: 'postgres'
      static_configs:
      - targets: ['postgres:9187']
    - job_name: 'qdrant'
      static_configs:
      - targets: ['qdrant:6333']
      metrics_path: '/metrics'
```

### Grafana Dashboards

```bash
# Install Grafana
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

helm install grafana grafana/grafana \
  --namespace rag-system \
  --set persistence.enabled=true \
  --set persistence.size=10Gi \
  --set adminPassword=admin
```

## SSL/TLS Configuration

### Certificate Management

```yaml
# cert-manager-issuer.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourragdomain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup-db.sh

BACKUP_DIR=\"/backups/postgres\"
TIMESTAMP=$(date +\"%Y%m%d_%H%M%S\")
BACKUP_FILE=\"${BACKUP_DIR}/ragdb_backup_${TIMESTAMP}.sql\"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
kubectl exec -n rag-system postgres-0 -- pg_dump -U raguser ragdb > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name \"*.gz\" -mtime +7 -delete

echo \"Backup completed: ${BACKUP_FILE}.gz\"
```

### Vector Database Backup

```bash
#!/bin/bash
# backup-vectors.sh

BACKUP_DIR=\"/backups/qdrant\"
TIMESTAMP=$(date +\"%Y%m%d_%H%M%S\")

# Create snapshot
kubectl exec -n rag-system qdrant-0 -- curl -X POST \"http://localhost:6333/snapshots\"

# Copy snapshot
kubectl cp rag-system/qdrant-0:/qdrant/snapshots $BACKUP_DIR/$TIMESTAMP/

echo \"Vector backup completed: $BACKUP_DIR/$TIMESTAMP\"
```

## Security Hardening

### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rag-network-policy
  namespace: rag-system
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: rag-system
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: rag-system
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

### Pod Security Standards

```yaml
# pod-security.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

## Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check database connectivity
kubectl exec -n rag-system api-gateway-xxx -- python -c \"
import psycopg2
try:
    conn = psycopg2.connect('postgresql://raguser:password@postgres:5432/ragdb')
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
\"
```

#### Vector Database Issues
```bash
# Check Qdrant health
kubectl exec -n rag-system qdrant-0 -- curl http://localhost:6333/health

# Check collections
kubectl exec -n rag-system qdrant-0 -- curl http://localhost:6333/collections
```

#### Storage Issues
```bash
# Check persistent volume claims
kubectl get pvc -n rag-system

# Check storage usage
kubectl exec -n rag-system postgres-0 -- df -h
kubectl exec -n rag-system qdrant-0 -- df -h
```

### Log Analysis

```bash
# Centralized logging with ELK stack
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch --namespace rag-system
helm install kibana elastic/kibana --namespace rag-system
helm install filebeat elastic/filebeat --namespace rag-system
```

## Performance Tuning

### Database Optimization

```sql
-- PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '25% of RAM';
ALTER SYSTEM SET effective_cache_size = '75% of RAM';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;

-- Reload configuration
SELECT pg_reload_conf();
```

### Vector Database Optimization

```yaml
# Qdrant configuration optimization
service:
  max_request_size_mb: 32
  max_workers: 0  # Auto-detect

storage:
  performance:
    max_search_threads: 0  # Auto-detect
    max_optimization_threads: 2

hnsw_config:
  m: 16
  ef_construct: 200
  full_scan_threshold: 10000
  max_indexing_threads: 0
```

This deployment guide provides comprehensive instructions for setting up the RAG system in both development and production environments, with proper scaling, monitoring, and security considerations.
