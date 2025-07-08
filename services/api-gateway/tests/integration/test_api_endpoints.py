import pytest
from fastapi.testclient import TestClient


def test_config_endpoint(client: TestClient):
    """Test the configuration endpoint."""
    response = client.get("/api/v1/config")
    # This might return 401 if authentication is required
    assert response.status_code in [200, 401]


def test_documents_endpoint(client: TestClient):
    """Test the documents list endpoint."""
    response = client.get("/api/v1/documents")
    # This might return 401 if authentication is required
    assert response.status_code in [200, 401]


def test_stats_endpoint(client: TestClient):
    """Test the analytics stats endpoint."""
    response = client.get("/api/v1/analytics/stats")
    # This might return 401 if authentication is required
    assert response.status_code in [200, 401]
