# Technology Stack

## Overview

This document outlines the complete technology stack for the Open Source RAG System, including rationale for each choice, alternatives considered, and integration details.

## Core Technologies

### Programming Languages

#### Python 3.11+
**Primary Language for Backend Services**

**Rationale**:
- Extensive ML/AI ecosystem (transformers, langchain, etc.)
- Rich document processing libraries
- Mature web frameworks (FastAPI)
- Strong typing support with Pydantic
- Active community and maintenance

**Key Libraries**:
- **FastAPI**: High-performance web framework with automatic OpenAPI
- **Pydantic**: Data validation and settings management
- **SQLAlchemy**: Database ORM with async support
- **Celery**: Distributed task queue for background processing
- **sentence-transformers**: State-of-the-art embedding models

#### TypeScript/React
**Frontend Web Interface**

**Rationale**:
- Type safety for large applications
- Rich ecosystem of components
- Modern development experience
- Strong tooling support

## Document Processing Stack

### PDF Processing
```python
# Primary libraries with fallback chain
1. pymupdf (fitz) - Primary for speed and accuracy
2. pdfplumber - Backup for complex layouts
3. PyPDF2 - Legacy fallback
```

**Implementation Strategy**:
```python
async def extract_pdf_text(file_path: str) -> ExtractedContent:
    try:
        # Try pymupdf first - fastest and most accurate
        return await extract_with_pymupdf(file_path)
    except Exception as e:
        logger.warning(f"pymupdf failed: {e}, trying pdfplumber")
        try:
            return await extract_with_pdfplumber(file_path)
        except Exception as e2:
            logger.warning(f"pdfplumber failed: {e2}, trying PyPDF2")
            return await extract_with_pypdf2(file_path)
```

### Word Document Processing
```python
# Libraries and approach
- python-docx: Primary for .docx files
- python-docx2txt: Fallback for text extraction
- mammoth: For complex formatting preservation
```

### Excel Processing
```python
# Libraries for different use cases
- openpyxl: Primary for .xlsx files with rich metadata
- pandas: Data analysis and CSV conversion
- xlrd: Legacy .xls support
```

### XML Processing
```python
# Flexible XML parsing stack
- lxml: High-performance C-based parser
- BeautifulSoup: Robust HTML/XML parsing
- xml.etree.ElementTree: Built-in fallback
```

## Database Technologies

### PostgreSQL 15+
**Primary Database for Metadata**

**Rationale**:
- ACID compliance for data integrity
- JSON/JSONB support for flexible schemas
- Full-text search capabilities
- Excellent performance and scaling
- Rich extension ecosystem

**Key Features Used**:
- UUID primary keys for distributed systems
- JSONB columns for flexible metadata
- Full-text search indexes
- Connection pooling with asyncpg

**Configuration**:
```sql
-- Performance optimizations
shared_buffers = '256MB'
effective_cache_size = '1GB'
maintenance_work_mem = '64MB'
checkpoint_completion_target = 0.9
wal_buffers = '16MB'
default_statistics_target = 100
```

### Qdrant Vector Database
**High-Performance Vector Search**

**Rationale**:
- Native Rust implementation for speed
- Advanced HNSW indexing
- Horizontal scaling support
- Rich filtering capabilities
- Active development and community

**Configuration**:
```yaml
# qdrant.yaml
service:
  http_port: 6333
  grpc_port: 6334

storage:
  storage_path: ./storage
  snapshots_path: ./snapshots

# Performance settings
hnsw_config:
  m: 16  # Number of bi-directional links for every new element during construction
  ef_construct: 200  # Size of the dynamic candidate list
  full_scan_threshold: 10000
  max_indexing_threads: 0  # Auto-detect

optimizers_config:
  deleted_threshold: 0.2
  vacuum_min_vector_number: 1000
  default_segment_number: 0  # Auto-detect
```

### Redis (Optional)
**Caching and Session Storage**

**Use Cases**:
- Query result caching
- Session management
- Rate limiting counters
- Background job queues

## Machine Learning Stack

### Embedding Models

#### sentence-transformers/all-mpnet-base-v2
**Primary Embedding Model**

**Specifications**:
- Model size: ~420MB
- Vector dimensions: 768
- Max sequence length: 384 tokens
- Language: English (primarily)

**Performance**:
- Encoding speed: ~1000 sentences/second (CPU)
- Quality: SOTA on various benchmarks

#### Alternative Models
```python
# Model selection based on use case
EMBEDDING_MODELS = {
    "general": "sentence-transformers/all-mpnet-base-v2",
    "multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "code": "microsoft/codebert-base",
    "domain_specific": "sentence-transformers/all-distilroberta-v1"
}
```

### Local LLM Integration

#### Ollama (Recommended)
**Easy Local Model Deployment**

**Supported Models**:
- Llama 3.1 (8B, 70B)
- Mistral 7B
- CodeLlama
- Phi-3
- Custom fine-tuned models

**Integration**:
```python
from ollama import Client

class OllamaLLM:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.client = Client(host='http://localhost:11434')
        self.model_name = model_name
    
    async def generate_response(self, prompt: str, context: str) -> str:
        response = await self.client.generate(
            model=self.model_name,
            prompt=f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:",
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 500
            }
        )
        return response['response']
```

#### vLLM (High Performance)
**Production-Grade Inference Server**

**Features**:
- PagedAttention for memory efficiency
- Continuous batching
- Tensor parallelism
- Quantization support

#### Text Generation Inference (Alternative)
**Hugging Face's Inference Server**

**Benefits**:
- Integration with Hugging Face Hub
- Streaming responses
- Batching and caching

## API Framework

### FastAPI
**Modern Python Web Framework**

**Key Features**:
- Automatic OpenAPI/Swagger documentation
- Type hints integration with Pydantic
- Async/await support
- Built-in validation
- High performance (comparable to Node.js)

**Extensions**:
```python
# Core dependencies
fastapi[all]  # Includes extras like Jinja2, python-multipart
uvicorn[standard]  # ASGI server
gunicorn  # Process manager for production
```

### Authentication & Authorization

#### JWT Tokens
**Stateless Authentication**

```python
from jose import JWTError, jwt
from passlib.context import CryptContext

class AuthService:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = os.getenv("SECRET_KEY")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
```

## Container Orchestration

### Docker
**Application Containerization**

**Multi-stage Build Strategy**:
```dockerfile
# Base stage with Python dependencies
FROM python:3.11-slim as base
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Production stage
FROM base as production
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
**Local Development Environment**

```yaml
version: '3.8'
services:
  api-gateway:
    build: ./services/api-gateway
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/ragdb
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - postgres
      - qdrant
      - redis

  document-processor:
    build: ./services/document-processor
    volumes:
      - ./storage:/app/storage
    environment:
      - CELERY_BROKER=redis://redis:6379/0

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ragdb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
  qdrant_data:
```

### Kubernetes (Production)
**Scalable Orchestration**

```yaml
# Example deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
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
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
```

## Monitoring and Observability

### Logging
**Structured Logging with JSON**

```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

### Metrics
**Prometheus + Grafana**

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

# Business metrics
documents_processed = Counter('documents_processed_total', 'Total documents processed')
query_duration = Histogram('query_duration_seconds', 'Time spent processing queries')
active_connections = Gauge('active_connections', 'Active database connections')
```

### Health Checks
**Comprehensive Service Monitoring**

```python
from fastapi import FastAPI
from sqlalchemy import text

app = FastAPI()

@app.get("/health")
async def health_check():
    checks = {
        "database": await check_database(),
        "qdrant": await check_qdrant(),
        "storage": await check_storage(),
        "llm": await check_llm_service()
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return {"status": "healthy" if all_healthy else "unhealthy", "checks": checks}
```

## Development Tools

### Code Quality
```python
# Development dependencies
black  # Code formatting
isort  # Import sorting
flake8  # Linting
mypy  # Type checking
pytest  # Testing framework
pytest-asyncio  # Async testing
pytest-cov  # Coverage reporting
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.950
    hooks:
      - id: mypy
```

## Performance Considerations

### Database Optimization
```sql
-- Indexes for performance
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_upload_time ON documents(upload_timestamp);
CREATE INDEX idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_chunks_vector_id ON document_chunks(vector_id);

-- Full-text search
CREATE INDEX idx_chunks_content_fts ON document_chunks USING gin(to_tsvector('english', content));
```

### Caching Strategy
```python
# Redis caching for expensive operations
from functools import wraps
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached(expiration: int = 3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try cache first
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Compute and cache result
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

## Security Technologies

### Input Validation
```python
from pydantic import BaseModel, validator
import bleach

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict = {}
    
    @validator('query')
    def sanitize_query(cls, v):
        # Remove potentially dangerous characters
        return bleach.clean(v, strip=True)
    
    @validator('top_k')
    def validate_top_k(cls, v):
        if not 1 <= v <= 100:
            raise ValueError('top_k must be between 1 and 100')
        return v
```

### Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/query")
@limiter.limit("10/minute")
async def query_documents(request: Request, query_data: QueryRequest):
    # Query processing logic
    pass
```

This technology stack provides a solid foundation for building a scalable, maintainable, and high-performance RAG system while maintaining the principle of local deployment and data privacy.
