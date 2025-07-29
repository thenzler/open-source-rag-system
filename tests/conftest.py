"""
Test configuration and fixtures for the RAG System
"""
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

# Set test environment variables
os.environ["ENVIRONMENT"] = "test"
os.environ["DEBUG"] = "true"
os.environ["DATABASE_PATH"] = "data/test_rag_database.db"
os.environ["AUDIT_DATABASE_PATH"] = "data/test_audit.db"
os.environ["SECRET_KEY"] = "test-secret-key"
os.environ["ENABLE_REDIS_CACHE"] = "false"
os.environ["ENABLE_PROGRESS_TRACKING"] = "true"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    from core.main import app
    
    with TestClient(app) as client:
        yield client

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        yield tmp.name
    # Cleanup
    Path(tmp.name).unlink(missing_ok=True)

@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing."""
    mock = AsyncMock()
    mock.generate_answer.return_value = {
        "answer": "Test answer",
        "confidence": 0.9,
        "sources": ["test_source.pdf"],
        "model": "test-model"
    }
    mock.health_check.return_value = True
    mock.list_models.return_value = ["test-model"]
    return mock

@pytest.fixture
def mock_document_service():
    """Mock document service for testing."""
    mock = AsyncMock()
    mock.add_document.return_value = "test-doc-id"
    mock.get_document.return_value = {
        "id": "test-doc-id",
        "filename": "test.pdf",
        "content": "Test content",
        "metadata": {}
    }
    mock.list_documents.return_value = []
    mock.health_check.return_value = True
    return mock

@pytest.fixture
def mock_query_service():
    """Mock query service for testing."""
    mock = AsyncMock()
    mock.answer_query.return_value = {
        "answer": "Test answer",
        "confidence": 0.9,
        "sources": ["test_source.pdf"],
        "response_time": 1.0
    }
    mock.health_check.return_value = True
    return mock

# Test markers
unit_test = pytest.mark.unit
integration_test = pytest.mark.integration
performance_test = pytest.mark.performance

# Helper functions for assertions
def assert_valid_uuid(value):
    """Assert that a value is a valid UUID string."""
    import uuid
    try:
        uuid.UUID(value)
        return True
    except (ValueError, TypeError):
        return False

def assert_positive_number(value):
    """Assert that a value is a positive number."""
    return isinstance(value, (int, float)) and value > 0

def assert_valid_timestamp(value):
    """Assert that a value is a valid timestamp."""
    try:
        from datetime import datetime
        if isinstance(value, str):
            datetime.fromisoformat(value.replace('Z', '+00:00'))
        elif isinstance(value, (int, float)):
            datetime.fromtimestamp(value)
        return True
    except (ValueError, TypeError):
        return False

def assert_document_response(response_data):
    """Assert that response data contains valid document fields."""
    required_fields = ["id", "filename"]
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"

def assert_query_response(response_data):
    """Assert that response data contains valid query response fields."""
    required_fields = ["answer"]
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"