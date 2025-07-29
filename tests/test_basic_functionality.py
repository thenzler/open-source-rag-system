#!/usr/bin/env python3
"""
Basic functionality tests that don't require external services
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def test_import_core_modules():
    """Test that core modules can be imported"""
    # Test basic imports
    from core.main import app

    assert app is not None

    # Test router imports
    from core.routers import documents, llm, query, system

    assert documents.router is not None
    assert query.router is not None
    assert system.router is not None
    assert llm.router is not None


def test_app_creation():
    """Test that FastAPI app is created successfully"""
    from core.main import app

    assert app.title == "RAG System API"
    assert app.description == "Modular Retrieval-Augmented Generation System"
    assert app.version == "1.0.0"


def test_optional_services_graceful_handling():
    """Test that optional services are handled gracefully when unavailable"""
    # Mock Redis unavailable
    with patch("core.services.redis_cache_service.REDIS_AVAILABLE", False):
        from core.services.redis_cache_service import RedisCacheService

        cache_service = RedisCacheService()
        # Should not crash when Redis is unavailable
        assert cache_service is not None


def test_basic_api_endpoints():
    """Test basic API endpoints"""
    from core.main import app

    with TestClient(app) as client:
        # Test root redirect
        response = client.get("/", follow_redirects=False)
        assert response.status_code in [200, 301, 302]

        # Test API info
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "RAG System API"


def test_cors_middleware():
    """Test CORS middleware configuration"""
    from core.main import app
    from fastapi.testclient import TestClient

    # Test CORS headers are present in response
    with TestClient(app) as client:
        response = client.options("/", headers={"Origin": "http://localhost:3000"})
        assert "access-control-allow-origin" in response.headers


def test_security_headers_middleware():
    """Test security headers middleware"""
    from core.main import app

    with TestClient(app) as client:
        response = client.get("/api")

        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"


def test_csrf_token_endpoint():
    """Test CSRF token generation"""
    from core.main import app

    with TestClient(app) as client:
        response = client.get("/api/v1/csrf-token")
        assert response.status_code == 200
        data = response.json()
        assert "csrf_token" in data
        assert "expires_in" in data
        assert data["expires_in"] == 86400


def test_health_check_endpoints():
    """Test health check endpoints"""
    from core.main import app

    with TestClient(app) as client:
        # Test if health endpoints are accessible
        # Note: These might return errors if services aren't running, but shouldn't crash
        response = client.get("/api/v1/status")
        assert response.status_code in [200, 500, 503]  # Allow service unavailable

        response = client.get("/api/v1/health")
        assert response.status_code in [200, 500, 503]  # Allow service unavailable


def test_environment_variable_handling():
    """Test environment variable configuration"""
    from core.main import CONFIG_AVAILABLE, config

    # Should handle missing config gracefully
    if not CONFIG_AVAILABLE:
        assert config is None
    else:
        # If config is available, it should be an object
        assert config is not None


def test_metrics_service_imports():
    """Test metrics service can be imported"""
    from core.middleware.metrics_middleware import MetricsMiddleware
    from core.services.metrics_service import init_metrics_service

    assert init_metrics_service is not None
    assert MetricsMiddleware is not None


def test_tenant_middleware_imports():
    """Test tenant middleware can be imported"""
    from core.middleware import initialize_tenant_resolver, tenant_middleware
    from core.repositories.tenant_repository import TenantRepository

    assert tenant_middleware is not None
    assert initialize_tenant_resolver is not None
    assert TenantRepository is not None


def test_security_utils():
    """Test security utilities"""
    from core.utils.encryption import setup_encryption_from_config
    from core.utils.security import initialize_id_obfuscator

    assert initialize_id_obfuscator is not None
    assert setup_encryption_from_config is not None


if __name__ == "__main__":
    pytest.main([__file__])
