"""
Test configuration and fixtures for the RAG System
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import httpx

from app.main import app
from app.core.config import get_settings, override_settings
from app.core.database import get_database, Base
from app.services.document_service import DocumentService
from app.services.query_service import QueryService
from app.services.analytics_service import AnalyticsService


# Test settings
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"
TEST_SETTINGS = {
    "environment": "testing",
    "debug": True,
    "database_url": TEST_DATABASE_URL,
    "redis_url": "redis://localhost:6379/1",
    "document_processor_url": "http://localhost:8001",
    "vector_engine_url": "http://localhost:8002",
    "llm_service_url": "http://localhost:11434",
    "upload_directory": "./test_uploads",
    "enable_caching": False,
    "enable_query_expansion": False,
    "enable_reranking": False,
    "max_file_size_mb": 10,
    "chunk_size": 100,
    "chunk_overlap": 20,
    "max_query_length": 500,
    "max_search_results": 10,
    "secret_key": "test-secret-key",
    "jwt_secret_key": "test-jwt-secret-key",
}


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def override_test_settings():
    """Override settings for testing."""
    override_settings(**TEST_SETTINGS)
    yield
    # Reset after tests


@pytest.fixture(scope="session")
async def test_engine(override_test_settings):
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    await engine.dispose()


@pytest.fixture
async def test_db(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture
def override_get_database(test_db):
    """Override the get_database dependency."""
    async def _get_test_db():
        yield test_db
    
    app.dependency_overrides[get_database] = _get_test_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def test_client(override_get_database):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_document_service():
    """Mock document service."""
    service = MagicMock(spec=DocumentService)
    service.initialize = AsyncMock()
    service.upload_document = AsyncMock()
    service.get_document = AsyncMock()
    service.list_documents = AsyncMock()
    service.delete_document = AsyncMock()
    service.process_document_async = AsyncMock()
    service.health_check = AsyncMock(return_value=True)
    return service


@pytest.fixture
def mock_query_service():
    """Mock query service."""
    service = MagicMock(spec=QueryService)
    service.initialize = AsyncMock()
    service.query_documents = AsyncMock()
    service.get_similar_documents = AsyncMock()
    service.health_check = AsyncMock(return_value=True)
    service.check_llm_health = AsyncMock(return_value=True)
    return service


@pytest.fixture
def mock_analytics_service():
    """Mock analytics service."""
    service = MagicMock(spec=AnalyticsService)
    service.initialize = AsyncMock()
    service.log_query = AsyncMock()
    service.get_system_stats = AsyncMock()
    service.get_time_series_data = AsyncMock()
    service.get_popular_queries = AsyncMock()
    return service


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for external API calls."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        "id": "test-doc-123",
        "filename": "test_document.pdf",
        "original_filename": "test_document.pdf",
        "file_path": "/test/path/test_document.pdf",
        "file_size": 1024,
        "file_type": "pdf",
        "mime_type": "application/pdf",
        "status": "completed",
        "user_id": "test_user",
        "chunks_count": 5,
        "metadata": {"test": "metadata"}
    }


@pytest.fixture
def sample_query_data():
    """Sample query data for testing."""
    return {
        "query": "What is machine learning?",
        "results": [
            {
                "id": "chunk-1",
                "score": 0.95,
                "content": "Machine learning is a subset of artificial intelligence...",
                "metadata": {"document_id": "test-doc-123", "chunk_index": 0},
                "document_id": "test-doc-123",
                "source_document": "test_document.pdf"
            },
            {
                "id": "chunk-2",
                "score": 0.87,
                "content": "There are three main types of machine learning...",
                "metadata": {"document_id": "test-doc-123", "chunk_index": 1},
                "document_id": "test-doc-123",
                "source_document": "test_document.pdf"
            }
        ],
        "total_results": 2,
        "response_time": 0.123
    }


@pytest.fixture
def sample_vector_search_response():
    """Sample vector search response."""
    return {
        "results": [
            {
                "id": "chunk-1",
                "score": 0.95,
                "content": "Machine learning is a subset of artificial intelligence...",
                "metadata": {"document_id": "test-doc-123", "chunk_index": 0}
            },
            {
                "id": "chunk-2", 
                "score": 0.87,
                "content": "There are three main types of machine learning...",
                "metadata": {"document_id": "test-doc-123", "chunk_index": 1}
            }
        ],
        "query": "What is machine learning?",
        "total_results": 2,
        "search_time": 0.05
    }


@pytest.fixture
def sample_embedding_response():
    """Sample embedding response."""
    return {
        "embeddings": [
            [0.1, 0.2, 0.3, 0.4, 0.5],  # 5-dimensional for testing
            [0.2, 0.3, 0.4, 0.5, 0.6]
        ],
        "model": "test-model",
        "dimensions": 5
    }


@pytest.fixture
def sample_llm_response():
    """Sample LLM response."""
    return {
        "response": "This is a test response from the LLM.",
        "model": "test-model",
        "created_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def sample_upload_file():
    """Sample file upload for testing."""
    from io import BytesIO
    from fastapi import UploadFile
    
    file_content = b"This is a test file content"
    file_like = BytesIO(file_content)
    
    return UploadFile(
        filename="test.txt",
        file=file_like,
        size=len(file_content),
        headers={"content-type": "text/plain"}
    )


@pytest.fixture
def sample_stats_data():
    """Sample system statistics data."""
    return {
        "total_documents": 10,
        "documents_by_status": {
            "completed": 8,
            "processing": 1,
            "failed": 1
        },
        "documents_by_type": {
            "pdf": 5,
            "docx": 3,
            "txt": 2
        },
        "total_queries": 100,
        "recent_queries": 25,
        "avg_response_time": 0.234,
        "storage_used": 1048576,
        "total_chunks": 50,
        "last_updated": "2024-01-01T00:00:00Z"
    }


# Utility functions for testing

def assert_document_response(response_data, expected_data):
    """Assert document response matches expected data."""
    assert response_data["id"] == expected_data["id"]
    assert response_data["filename"] == expected_data["filename"]
    assert response_data["original_filename"] == expected_data["original_filename"]
    assert response_data["file_type"] == expected_data["file_type"]
    assert response_data["status"] == expected_data["status"]
    assert response_data["user_id"] == expected_data["user_id"]


def assert_query_response(response_data, expected_data):
    """Assert query response matches expected data."""
    assert response_data["query"] == expected_data["query"]
    assert response_data["total_results"] == expected_data["total_results"]
    assert "response_time" in response_data
    assert "results" in response_data
    assert len(response_data["results"]) == len(expected_data["results"])


def create_test_document(db_session, **kwargs):
    """Create a test document in the database."""
    from app.models.documents import Document
    
    default_data = {
        "id": "test-doc-123",
        "filename": "test.pdf",
        "original_filename": "test.pdf",
        "file_path": "/test/path/test.pdf",
        "file_size": 1024,
        "file_type": "pdf",
        "mime_type": "application/pdf",
        "status": "completed",
        "user_id": "test_user",
        "chunks_count": 5
    }
    
    data = {**default_data, **kwargs}
    document = Document(**data)
    db_session.add(document)
    return document


def create_test_query_log(db_session, **kwargs):
    """Create a test query log in the database."""
    from app.models.queries import QueryLog
    
    default_data = {
        "query_text": "test query",
        "user_id": "test_user",
        "results_count": 5,
        "response_time": 0.123
    }
    
    data = {**default_data, **kwargs}
    query_log = QueryLog(**data)
    db_session.add(query_log)
    return query_log


async def setup_test_data(db_session):
    """Set up test data in the database."""
    # Create test documents
    doc1 = create_test_document(
        db_session,
        id="doc-1",
        filename="test1.pdf",
        original_filename="test1.pdf",
        file_type="pdf",
        status="completed"
    )
    
    doc2 = create_test_document(
        db_session,
        id="doc-2",
        filename="test2.txt",
        original_filename="test2.txt",
        file_type="text",
        status="processing"
    )
    
    # Create test query logs
    query1 = create_test_query_log(
        db_session,
        query_text="machine learning",
        results_count=5,
        response_time=0.123
    )
    
    query2 = create_test_query_log(
        db_session,
        query_text="artificial intelligence",
        results_count=3,
        response_time=0.234
    )
    
    await db_session.commit()
    
    return {
        "documents": [doc1, doc2],
        "queries": [query1, query2]
    }


# Test decorators and markers

def requires_services(*services):
    """Decorator to mark tests that require specific services."""
    def decorator(func):
        func._required_services = services
        return func
    return decorator


def integration_test(func):
    """Decorator to mark integration tests."""
    return pytest.mark.integration(func)


def unit_test(func):
    """Decorator to mark unit tests."""
    return pytest.mark.unit(func)


def slow_test(func):
    """Decorator to mark slow tests."""
    return pytest.mark.slow(func)


# Custom assertions

def assert_valid_uuid(value):
    """Assert that value is a valid UUID."""
    import uuid
    try:
        uuid.UUID(value)
    except ValueError:
        raise AssertionError(f"'{value}' is not a valid UUID")


def assert_positive_number(value):
    """Assert that value is a positive number."""
    assert isinstance(value, (int, float))
    assert value > 0


def assert_valid_timestamp(value):
    """Assert that value is a valid ISO timestamp."""
    from datetime import datetime
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise AssertionError(f"'{value}' is not a valid ISO timestamp")


# Mock factories

class MockResponse:
    """Mock HTTP response."""
    
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = str(json_data)
    
    def json(self):
        return self.json_data


def mock_vector_search(query, top_k=5, **kwargs):
    """Mock vector search response."""
    return MockResponse({
        "results": [
            {
                "id": f"chunk-{i}",
                "score": 0.9 - (i * 0.1),
                "content": f"Content for chunk {i}",
                "metadata": {"document_id": f"doc-{i}", "chunk_index": i}
            }
            for i in range(min(top_k, 3))
        ],
        "query": query,
        "total_results": min(top_k, 3),
        "search_time": 0.05
    })


def mock_embedding_generation(texts, **kwargs):
    """Mock embedding generation response."""
    return MockResponse({
        "embeddings": [
            [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i]
            for i, _ in enumerate(texts)
        ],
        "model": "test-model",
        "dimensions": 5
    })


def mock_llm_generation(prompt, **kwargs):
    """Mock LLM generation response."""
    return MockResponse({
        "response": f"Generated response for: {prompt[:50]}...",
        "model": "test-model",
        "created_at": "2024-01-01T00:00:00Z"
    })


# Test utilities

def skip_if_no_services():
    """Skip test if required services are not available."""
    def decorator(func):
        @pytest.mark.skipif(
            not hasattr(func, '_required_services'),
            reason="No required services specified"
        )
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Cleanup utilities

@pytest.fixture(autouse=True)
async def cleanup_test_files():
    """Cleanup test files after each test."""
    yield
    
    # Clean up test upload directory
    import shutil
    import os
    
    test_upload_dir = "./test_uploads"
    if os.path.exists(test_upload_dir):
        shutil.rmtree(test_upload_dir)
