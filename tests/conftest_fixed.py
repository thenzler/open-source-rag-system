"""
Fixed test configuration for the actual RAG System project structure
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
API_BASE = "http://localhost:8001"
TEST_TIMEOUT = 30

@pytest.fixture(scope="session")
def api_base():
    """Base API URL for testing"""
    return API_BASE

@pytest.fixture(scope="session")
def test_client():
    """Test client fixture - checks if API is running"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            return API_BASE
        else:
            pytest.skip("API server not healthy")
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")

@pytest.fixture
def sample_document():
    """Sample document for testing"""
    return {
        "content": """
        This is a test document for the RAG system.
        It contains information about artificial intelligence and machine learning.
        
        Machine learning is a subset of artificial intelligence that focuses on 
        algorithms that can learn from data and make predictions or decisions.
        
        Natural language processing (NLP) is another important area of AI that 
        deals with understanding and generating human language.
        
        Vector databases are used to store and retrieve high-dimensional vectors 
        efficiently, which is crucial for semantic search applications.
        """,
        "filename": "test_document.txt",
        "file_type": "text/plain"
    }

@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return {
        "query": "What is machine learning?",
        "top_k": 3
    }

@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing"""
    mock_client = MagicMock()
    mock_client.is_available.return_value = True
    mock_client.chat.return_value = {
        "message": {"content": "This is a test response from Ollama"}
    }
    return mock_client

@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer for testing"""
    mock_transformer = MagicMock()
    mock_transformer.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    return mock_transformer

@pytest.fixture
def test_upload_file(tmp_path, sample_document):
    """Create a temporary file for upload testing"""
    test_file = tmp_path / sample_document["filename"]
    test_file.write_text(sample_document["content"])
    return test_file

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Cleanup test files after each test"""
    yield
    
    # Clean up any test files
    test_files = [
        "test_document.txt",
        "test_upload.txt",
        "test_ml_guide.txt",
        "test_workflow.txt"
    ]
    
    for filename in test_files:
        if os.path.exists(filename):
            os.unlink(filename)

# Utility functions for testing
def create_test_document(content, filename="test_doc.txt"):
    """Create a test document file"""
    test_file = Path(filename)
    test_file.write_text(content)
    return test_file

def upload_test_document(api_base, test_file):
    """Upload a test document to the API"""
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'text/plain')}
            response = requests.post(f"{api_base}/api/v1/documents", files=files)
        return response
    except Exception as e:
        pytest.fail(f"Failed to upload test document: {e}")

def query_documents(api_base, query, top_k=3):
    """Query documents via API"""
    try:
        response = requests.post(
            f"{api_base}/api/v1/query",
            json={"query": query, "top_k": top_k},
            headers={"Content-Type": "application/json"}
        )
        return response
    except Exception as e:
        pytest.fail(f"Failed to query documents: {e}")

# Custom assertions
def assert_valid_response(response, expected_status=200):
    """Assert that response is valid"""
    assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}: {response.text}"

def assert_document_upload_response(response_data):
    """Assert document upload response is valid"""
    required_fields = ["filename", "status", "size"]
    for field in required_fields:
        assert field in response_data, f"Missing field: {field}"

def assert_query_response(response_data):
    """Assert query response is valid"""
    required_fields = ["results", "total_results"]
    for field in required_fields:
        assert field in response_data, f"Missing field: {field}"
    
    assert isinstance(response_data["results"], list), "Results should be a list"
    assert isinstance(response_data["total_results"], int), "Total results should be an integer"

def assert_document_list_response(response_data):
    """Assert document list response is valid"""
    required_fields = ["documents", "total"]
    for field in required_fields:
        assert field in response_data, f"Missing field: {field}"
    
    assert isinstance(response_data["documents"], list), "Documents should be a list"
    assert isinstance(response_data["total"], int), "Total should be an integer"

# Test markers
def requires_api(func):
    """Decorator to mark tests that require API server"""
    return pytest.mark.api(func)

def requires_ollama(func):
    """Decorator to mark tests that require Ollama"""
    return pytest.mark.ollama(func)

def slow_test(func):
    """Decorator to mark slow tests"""
    return pytest.mark.slow(func)

def integration_test(func):
    """Decorator to mark integration tests"""
    return pytest.mark.integration(func)

# Skip conditions
def skip_if_no_api():
    """Skip test if API is not available"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code != 200:
            pytest.skip("API server not healthy")
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")

def skip_if_no_ollama():
    """Skip test if Ollama is not available"""
    try:
        from ollama_client import get_ollama_client
        client = get_ollama_client()
        if not client.is_available():
            pytest.skip("Ollama not available")
    except ImportError:
        pytest.skip("Ollama client not available")

# Test data generators
def generate_test_documents(count=3):
    """Generate test documents"""
    documents = []
    for i in range(count):
        documents.append({
            "content": f"Test document {i+1} content about machine learning and AI.",
            "filename": f"test_doc_{i+1}.txt",
            "file_type": "text/plain"
        })
    return documents

def generate_test_queries():
    """Generate test queries"""
    return [
        "What is machine learning?",
        "How does natural language processing work?",
        "What are vector databases used for?",
        "Explain artificial intelligence"
    ]

# Mock factories
class MockAPIResponse:
    """Mock API response"""
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = str(json_data)
    
    def json(self):
        return self.json_data

def mock_successful_upload():
    """Mock successful document upload"""
    return MockAPIResponse({
        "filename": "test_document.txt",
        "status": "uploaded",
        "size": 1024,
        "id": "test-doc-123"
    })

def mock_successful_query():
    """Mock successful query response"""
    return MockAPIResponse({
        "results": [
            {
                "content": "Machine learning is a subset of artificial intelligence...",
                "score": 0.95,
                "source_document": "test_document.txt"
            }
        ],
        "total_results": 1,
        "query": "What is machine learning?"
    })

def mock_document_list():
    """Mock document list response"""
    return MockAPIResponse({
        "documents": [
            {
                "filename": "test_document.txt",
                "status": "processed",
                "size": 1024
            }
        ],
        "total": 1
    })

# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "api: marks tests as requiring API server"
    )
    config.addinivalue_line(
        "markers", "ollama: marks tests as requiring Ollama"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add skip conditions for API tests
        if "api" in item.keywords:
            item.add_marker(pytest.mark.skipif(
                not is_api_available(),
                reason="API server not available"
            ))
        
        # Add skip conditions for Ollama tests  
        if "ollama" in item.keywords:
            item.add_marker(pytest.mark.skipif(
                not is_ollama_available(),
                reason="Ollama not available"
            ))

def is_api_available():
    """Check if API is available"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def is_ollama_available():
    """Check if Ollama is available"""
    try:
        from ollama_client import get_ollama_client
        client = get_ollama_client()
        return client.is_available()
    except:
        return False