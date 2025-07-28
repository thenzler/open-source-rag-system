import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    """Test the health endpoint returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "services" in data


def test_api_health_endpoint(client: TestClient):
    """Test the API health endpoint returns 200."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "services" in data
