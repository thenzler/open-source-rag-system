# Testing Strategy

## Overview

The Open Source RAG System employs a comprehensive testing strategy covering unit tests, integration tests, performance tests, and end-to-end validation. This ensures reliability, performance, and maintainability across all components.

## Testing Philosophy

### Core Principles
1. **Test Pyramid**: More unit tests, fewer integration tests, minimal E2E tests
2. **Fast Feedback**: Tests should run quickly to enable rapid development
3. **Reliable**: Tests should be deterministic and not flaky
4. **Maintainable**: Tests should be easy to understand and modify
5. **Comprehensive**: Critical paths and edge cases must be covered

### Quality Gates
- **Code Coverage**: Minimum 80% line coverage, 90% for critical components
- **Performance**: All API endpoints must respond within SLA times
- **Security**: Security tests must pass with zero high-severity issues
- **Documentation**: All public APIs must have test documentation

## Test Categories

### 1. Unit Tests

#### Scope
- Individual functions and methods
- Business logic validation
- Data transformation and processing
- Error handling and edge cases

#### Technologies
- **Framework**: pytest with asyncio support
- **Mocking**: pytest-mock, AsyncMock
- **Fixtures**: pytest fixtures for test data
- **Coverage**: pytest-cov

#### Example Structure
```python
# tests/unit/test_document_processor.py
import pytest
from unittest.mock import AsyncMock, patch
from app.services.document_processor import DocumentProcessor

@pytest.fixture
def sample_pdf_path():
    return "tests/fixtures/sample.pdf"

@pytest.fixture  
def mock_embedding_service():
    return AsyncMock()

class TestDocumentProcessor:
    
    @pytest.mark.asyncio
    async def test_process_pdf_success(self, sample_pdf_path, mock_embedding_service):
        processor = DocumentProcessor()
        processor.vector_service = mock_embedding_service
        
        result = await processor._process_pdf(sample_pdf_path)
        
        assert result['text'] is not None
        assert len(result['text']) > 0
        assert result['metadata']['format'] == 'pdf'
        assert 'pages' in result['metadata']
    
    @pytest.mark.asyncio
    async def test_process_pdf_empty_file(self):
        processor = DocumentProcessor()
        
        with pytest.raises(ProcessingError):
            await processor._process_pdf("nonexistent.pdf")
    
    def test_chunk_text_with_overlap(self):
        processor = DocumentProcessor()
        text = "This is a test document. " * 100  # Long text
        
        chunks = processor._chunk_text(text, {})
        
        assert len(chunks) > 1
        assert all(chunk['char_count'] <= processor.chunk_size for chunk in chunks)
        assert chunks[0]['metadata']['chunk_index'] == 0
```

#### Coverage Requirements
```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --cov=app
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
    --asyncio-mode=auto
```

### 2. Integration Tests

#### Scope
- Service-to-service communication
- Database operations
- External API integrations
- File system operations

#### Test Database
```python
# conftest.py
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from app.core.database import Base

@pytest.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(
        "postgresql+asyncpg://test:test@localhost:5432/test_ragdb",
        echo=True
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def db_session(test_engine):
    async with AsyncSession(test_engine) as session:
        yield session
        await session.rollback()
```

#### Example Integration Tests
```python
# tests/integration/test_document_workflow.py
import pytest
from app.services.document_service import DocumentService
from app.services.vector_service import VectorService

class TestDocumentWorkflow:
    
    @pytest.mark.asyncio
    async def test_full_document_processing_workflow(self, db_session, test_file):
        # Setup services
        doc_service = DocumentService()
        vector_service = VectorService()
        
        # Upload document
        document = await doc_service.upload_document(
            file=test_file,
            metadata='{"category": "test"}',
            user_id="test_user",
            db=db_session
        )
        
        assert document.status == ProcessingStatus.PENDING
        
        # Process document
        result = await doc_service.process_document_async(document.id, db_session)
        
        # Verify processing completed
        updated_doc = await doc_service.get_document(document.id, "test_user", db_session)
        assert updated_doc.status == ProcessingStatus.COMPLETED
        
        # Verify vectors were created
        vectors = await vector_service.get_document_vectors(document.id)
        assert len(vectors) > 0
        
        # Test search functionality
        search_results = await vector_service.search_similar(
            query="test query",
            document_ids=[document.id]
        )
        assert len(search_results) > 0
```

### 3. API Tests

#### Scope
- HTTP endpoint validation
- Request/response format testing
- Authentication and authorization
- Rate limiting and error handling

#### Framework
```python
# tests/api/test_documents_api.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test_token"}

class TestDocumentsAPI:
    
    @pytest.mark.asyncio
    async def test_upload_document_success(self, client, auth_headers):
        files = {"file": ("test.pdf", b"test content", "application/pdf")}
        
        response = await client.post(
            "/api/v1/documents",
            files=files,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "test.pdf"
        assert data["status"] == "pending"
    
    @pytest.mark.asyncio
    async def test_upload_document_file_too_large(self, client, auth_headers):
        large_content = b"x" * (100 * 1024 * 1024 + 1)  # Over 100MB
        files = {"file": ("large.pdf", large_content, "application/pdf")}
        
        response = await client.post(
            "/api/v1/documents",
            files=files,
            headers=auth_headers
        )
        
        assert response.status_code == 413
        assert "too large" in response.json()["error"]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_query_documents(self, client, auth_headers):
        query_data = {
            "query": "test query",
            "top_k": 5,
            "filters": {"category": ["test"]}
        }
        
        response = await client.post(
            "/api/v1/query",
            json=query_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)
```

### 4. Performance Tests

#### Load Testing with Locust
```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between
import json

class RAGSystemUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Login to get auth token
        response = self.client.post("/api/v1/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(3)
    def query_documents(self):
        query_data = {
            "query": "artificial intelligence machine learning",
            "top_k": 10
        }
        self.client.post(
            "/api/v1/query",
            json=query_data,
            headers=self.headers,
            name="Query Documents"
        )
    
    @task(1)
    def upload_document(self):
        files = {"file": ("test.txt", "Sample document content", "text/plain")}
        self.client.post(
            "/api/v1/documents",
            files=files,
            headers=self.headers,
            name="Upload Document"
        )
    
    @task(2)
    def list_documents(self):
        self.client.get(
            "/api/v1/documents?limit=20",
            headers=self.headers,
            name="List Documents"
        )
```

#### Performance Benchmarks
```python
# tests/performance/test_benchmarks.py
import pytest
import time
import asyncio
from app.services.vector_service import VectorService

class TestPerformanceBenchmarks:
    
    @pytest.mark.asyncio
    async def test_embedding_generation_performance(self):
        vector_service = VectorService()
        await vector_service.initialize()
        
        texts = ["Sample document text"] * 100
        
        start_time = time.time()
        embeddings = await vector_service._generate_embeddings(texts)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = len(texts) / processing_time
        
        # Assert performance requirements
        assert processing_time < 10.0  # Should process 100 texts in under 10 seconds
        assert throughput > 10  # Should process at least 10 texts per second
        assert len(embeddings) == len(texts)
    
    @pytest.mark.asyncio  
    async def test_search_performance(self):
        vector_service = VectorService()
        await vector_service.initialize()
        
        # Setup: Add test documents
        test_docs = [f"Document {i} content" for i in range(1000)]
        await vector_service.add_documents(
            texts=test_docs,
            metadatas=[{"doc_id": i} for i in range(1000)],
            document_id="test_doc"
        )
        
        # Test search performance
        query = "sample query text"
        
        start_time = time.time()
        results = await vector_service.search_similar(query, top_k=10)
        end_time = time.time()
        
        search_time = end_time - start_time
        
        # Assert performance requirements
        assert search_time < 1.0  # Search should complete in under 1 second
        assert len(results) <= 10
```

### 5. Security Tests

#### Authentication and Authorization
```python
# tests/security/test_auth.py
import pytest
from app.core.security import create_access_token, verify_token

class TestSecurity:
    
    def test_token_creation_and_verification(self):
        payload = {"user_id": "test_user", "username": "testuser"}
        token = create_access_token(payload)
        
        verified_payload = verify_token(token)
        
        assert verified_payload["user_id"] == payload["user_id"]
        assert verified_payload["username"] == payload["username"]
    
    def test_invalid_token_raises_error(self):
        invalid_token = "invalid.jwt.token"
        
        with pytest.raises(Exception):
            verify_token(invalid_token)
    
    @pytest.mark.asyncio
    async def test_unauthorized_access_blocked(self, client):
        response = await client.get("/api/v1/documents")
        assert response.status_code == 401
```

#### Input Validation
```python
# tests/security/test_input_validation.py
import pytest

class TestInputValidation:
    
    @pytest.mark.asyncio
    async def test_sql_injection_protection(self, client, auth_headers):
        malicious_query = "'; DROP TABLE documents; --"
        
        response = await client.post(
            "/api/v1/query",
            json={"query": malicious_query},
            headers=auth_headers
        )
        
        # Should not cause internal server error
        assert response.status_code in [200, 400]
    
    @pytest.mark.asyncio
    async def test_xss_protection(self, client, auth_headers):
        xss_content = "<script>alert('xss')</script>"
        files = {"file": ("test.txt", xss_content, "text/plain")}
        
        response = await client.post(
            "/api/v1/documents",
            files=files,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        # Verify content is properly escaped
        data = response.json()
        assert "<script>" not in str(data)
```

### 6. End-to-End Tests

#### Complete User Workflows
```python
# tests/e2e/test_user_workflows.py
import pytest

class TestUserWorkflows:
    
    @pytest.mark.asyncio
    async def test_complete_document_lifecycle(self, client):
        # 1. User registration/login
        auth_response = await client.post("/api/v1/auth/login", json={
            "username": "testuser",
            "password": "testpass"
        })
        token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 2. Upload document
        files = {"file": ("research.pdf", b"Research content", "application/pdf")}
        upload_response = await client.post(
            "/api/v1/documents",
            files=files,
            headers=headers
        )
        document_id = upload_response.json()["document_id"]
        
        # 3. Wait for processing (or mock completion)
        await asyncio.sleep(2)  # In real tests, poll status endpoint
        
        # 4. Query the document
        query_response = await client.post(
            "/api/v1/query",
            json={"query": "research findings", "top_k": 5},
            headers=headers
        )
        
        # 5. Verify results
        assert query_response.status_code == 200
        results = query_response.json()
        assert len(results["sources"]) > 0
        assert any(source["document_id"] == document_id for source in results["sources"])
        
        # 6. Delete document
        delete_response = await client.delete(
            f"/api/v1/documents/{document_id}",
            headers=headers
        )
        assert delete_response.status_code == 200
```

## Test Data Management

### Fixtures and Test Data
```python
# tests/fixtures/data_factory.py
import factory
from app.models.documents import Document

class DocumentFactory(factory.Factory):
    class Meta:
        model = Document
    
    id = factory.Faker('uuid4')
    filename = factory.Faker('file_name', extension='pdf')
    file_path = factory.LazyAttribute(lambda obj: f'/tmp/{obj.filename}')
    mime_type = 'application/pdf'
    file_size = factory.Faker('random_int', min=1024, max=10485760)
    user_id = factory.Faker('uuid4')
    status = 'pending'

# tests/conftest.py
@pytest.fixture
def sample_document():
    return DocumentFactory()

@pytest.fixture
def sample_documents():
    return DocumentFactory.create_batch(5)
```

### Test File Resources
```
tests/
├── fixtures/
│   ├── documents/
│   │   ├── sample.pdf
│   │   ├── sample.docx
│   │   ├── sample.xlsx
│   │   ├── sample.xml
│   │   └── sample.txt
│   └── data/
│       ├── test_queries.json
│       └── expected_results.json
```

## Continuous Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_ragdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
      
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=app --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
      env:
        DATABASE_URL: postgresql://postgres:test@localhost:5432/test_ragdb
        QDRANT_URL: http://localhost:6333
        REDIS_URL: redis://localhost:6379/0
    
    - name: Run API tests
      run: |
        pytest tests/api/ -v
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Performance Testing

### Load Testing Setup
```bash
# Run load tests
pip install locust
locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Automated performance testing
pytest tests/performance/ --benchmark-only
```

### Performance Metrics
- **API Response Time**: 95th percentile < 500ms
- **Query Throughput**: > 100 queries/second
- **Document Processing**: < 30 seconds for 10MB PDF
- **Memory Usage**: < 2GB for 10,000 documents
- **Storage Efficiency**: Vector compression ratio > 50%

## Test Environment Setup

### Local Development
```bash
# Setup test environment
make setup-test-env

# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-api
make test-performance
```

### Docker Test Environment
```yaml
# docker-compose.test.yml
version: '3.8'
services:
  test-runner:
    build:
      context: .
      dockerfile: Dockerfile.test
    environment:
      - DATABASE_URL=postgresql://test:test@postgres-test:5432/test_ragdb
      - QDRANT_URL=http://qdrant-test:6333
      - REDIS_URL=redis://redis-test:6379/0
    depends_on:
      - postgres-test
      - qdrant-test
      - redis-test
    command: pytest tests/ -v

  postgres-test:
    image: postgres:15
    environment:
      POSTGRES_DB: test_ragdb
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test

  qdrant-test:
    image: qdrant/qdrant:latest

  redis-test:
    image: redis:7-alpine
```

## Quality Assurance

### Code Quality Checks
```python
# setup.cfg
[flake8]
max-line-length = 100
exclude = .git,__pycache__,docs/source/conf.py,old,build,dist
ignore = E203,W503

[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True

[coverage:run]
source = app
omit = 
    */tests/*
    */venv/*
    */migrations/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

### Test Reporting
```python
# Generate comprehensive test report
pytest tests/ \
  --html=reports/test_report.html \
  --cov=app \
  --cov-report=html:reports/coverage \
  --junitxml=reports/junit.xml \
  --benchmark-json=reports/benchmark.json
```

This comprehensive testing strategy ensures the RAG system maintains high quality, performance, and reliability throughout development and deployment.
