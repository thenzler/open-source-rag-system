import pytest
import httpx
import asyncio


@pytest.mark.asyncio
async def test_health_endpoint_integration():
    """Test health endpoint with actual HTTP client."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/health", timeout=5.0)
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
        except httpx.ConnectError:
            pytest.skip("API server not available for integration tests")


@pytest.mark.asyncio
async def test_api_health_endpoint_integration():
    """Test API health endpoint with actual HTTP client."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/api/v1/health", timeout=5.0)
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
        except httpx.ConnectError:
            pytest.skip("API server not available for integration tests")
