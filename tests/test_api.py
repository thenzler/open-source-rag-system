"""
API Tests for the RAG System
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import json
from datetime import datetime

from tests.conftest import (
    assert_document_response,
    assert_query_response,
    assert_valid_uuid,
    assert_positive_number,
    assert_valid_timestamp,
    unit_test,
    integration_test
)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    @unit_test
    def test_health_check_success(self, test_client, mock_document_service, mock_query_service):
        """Test successful health check."""
        # Mock services as healthy
        mock_document_service.health_check.return_value = True
        mock_query_service.health_check.return_value = True
        mock_query_service.check_llm_health.return_value = True
        
        with patch('app.main.document_service', mock_document_service), \
             patch('app.main.query_service', mock_query_service):
            response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert data["services"]["database"] == "healthy"
        assert data["services"]["vector_database"] == "healthy"
        assert data["services"]["llm_service"] == "healthy"
    
    @unit_test
    def test_health_check_unhealthy(self, test_client, mock_document_service, mock_query_service):
        """Test unhealthy health check."""
        # Mock services as unhealthy
        mock_document_service.health_check.return_value = False
        mock_query_service.health_check.return_value = False
        mock_query_service.check_llm_health.return_value = False
        
        with patch('app.main.document_service', mock_document_service), \
             patch('app.main.query_service', mock_query_service):
            response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["services"]["database"] == "unhealthy"
        assert data["services"]["vector_database"] == "unhealthy"
        assert data["services"]["llm_service"] == "unhealthy"
    
    @unit_test
    def test_api_health_check(self, test_client, mock_document_service, mock_query_service):
        """Test API-specific health check."""
        # Mock services as healthy
        mock_document_service.health_check.return_value = True
        mock_query_service.health_check.return_value = True
        mock_query_service.check_llm_health.return_value = True
        
        with patch('app.main.document_service', mock_document_service), \
             patch('app.main.query_service', mock_query_service):
            response = test_client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestDocumentEndpoints:
    """Test document management endpoints."""
    
    @unit_test
    def test_upload_document_success(self, test_client, mock_document_service, sample_document_data):
        """Test successful document upload."""
        # Mock document service
        mock_document_service.upload_document.return_value = MagicMock(**sample_document_data)
        
        with patch('app.main.document_service', mock_document_service):
            response = test_client.post(
                "/api/v1/documents",
                files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
                data={"metadata": json.dumps({"test": "metadata"})}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert_document_response(data, sample_document_data)
        assert_valid_uuid(data["id"])
        
        # Verify service was called
        mock_document_service.upload_document.assert_called_once()
    
    @unit_test
    def test_upload_document_file_too_large(self, test_client, mock_document_service):
        """Test document upload with file too large."""
        from app.core.exceptions import ValidationError
        
        # Mock validation error
        mock_document_service.upload_document.side_effect = ValidationError("File too large")
        
        with patch('app.main.document_service', mock_document_service):
            response = test_client.post(
                "/api/v1/documents",
                files={"file": ("large.pdf", b"x" * 1000000, "application/pdf")}
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "File too large" in data["detail"]
    
    @unit_test
    def test_upload_document_unsupported_format(self, test_client, mock_document_service):
        """Test document upload with unsupported format."""
        from app.core.exceptions import ValidationError
        
        # Mock validation error
        mock_document_service.upload_document.side_effect = ValidationError("Unsupported file type")
        
        with patch('app.main.document_service', mock_document_service):
            response = test_client.post(
                "/api/v1/documents",
                files={"file": ("test.xyz", b"content", "application/octet-stream")}
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "Unsupported file type" in data["detail"]
    
    @unit_test
    def test_list_documents_success(self, test_client, mock_document_service, sample_document_data):
        """Test successful document listing."""
        # Mock document service
        mock_documents = [MagicMock(**sample_document_data)]
        mock_document_service.list_documents.return_value = (mock_documents, 1)
        
        with patch('app.main.document_service', mock_document_service):
            response = test_client.get("/api/v1/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["documents"]) == 1
        assert data["skip"] == 0
        assert data["limit"] == 50
        
        # Verify service was called
        mock_document_service.list_documents.assert_called_once()
    
    @unit_test
    def test_list_documents_with_filters(self, test_client, mock_document_service):
        """Test document listing with filters."""
        # Mock document service
        mock_document_service.list_documents.return_value = ([], 0)
        
        with patch('app.main.document_service', mock_document_service):
            response = test_client.get(
                "/api/v1/documents?skip=10&limit=20&status=completed&category=pdf&search=test"
            )
        
        assert response.status_code == 200
        
        # Verify service was called with correct parameters
        mock_document_service.list_documents.assert_called_once_with(
            skip=10,
            limit=20,
            status="completed",
            category="pdf",
            search="test",
            user_id="test_user",
            db=None
        )
    
    @unit_test
    def test_get_document_success(self, test_client, mock_document_service, sample_document_data):
        """Test successful document retrieval."""
        # Mock document service
        mock_document_service.get_document.return_value = MagicMock(**sample_document_data)
        
        with patch('app.main.document_service', mock_document_service):
            response = test_client.get("/api/v1/documents/test-doc-123")
        
        assert response.status_code == 200
        data = response.json()
        assert_document_response(data, sample_document_data)
        
        # Verify service was called
        mock_document_service.get_document.assert_called_once_with(
            document_id="test-doc-123",
            user_id="test_user",
            db=None
        )
    
    @unit_test
    def test_get_document_not_found(self, test_client, mock_document_service):
        """Test document retrieval for non-existent document."""
        from app.core.exceptions import DocumentNotFoundError
        
        # Mock document not found
        mock_document_service.get_document.side_effect = DocumentNotFoundError("Document not found")
        
        with patch('app.main.document_service', mock_document_service):
            response = test_client.get("/api/v1/documents/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "Document not found" in data["detail"]
    
    @unit_test
    def test_delete_document_success(self, test_client, mock_document_service):
        """Test successful document deletion."""
        # Mock document service
        mock_document_service.delete_document.return_value = {
            "deleted_chunks": 5,
            "deleted_vectors": 5
        }
        
        with patch('app.main.document_service', mock_document_service):
            response = test_client.delete("/api/v1/documents/test-doc-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Document deleted successfully"
        assert data["document_id"] == "test-doc-123"
        assert data["deleted_chunks"] == 5
        assert data["deleted_vectors"] == 5
        
        # Verify service was called
        mock_document_service.delete_document.assert_called_once_with(
            document_id="test-doc-123",
            user_id="test_user",
            db=None
        )
    
    @unit_test
    def test_delete_document_not_found(self, test_client, mock_document_service):
        """Test document deletion for non-existent document."""
        from app.core.exceptions import DocumentNotFoundError
        
        # Mock document not found
        mock_document_service.delete_document.side_effect = DocumentNotFoundError("Document not found")
        
        with patch('app.main.document_service', mock_document_service):
            response = test_client.delete("/api/v1/documents/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "Document not found" in data["detail"]


class TestQueryEndpoints:
    """Test query and search endpoints."""
    
    @unit_test
    def test_query_documents_success(self, test_client, mock_query_service, sample_query_data):
        """Test successful document query."""
        # Mock query service
        mock_query_service.query_documents.return_value = sample_query_data
        
        with patch('app.main.query_service', mock_query_service):
            response = test_client.post(
                "/api/v1/query",
                json={
                    "query": "What is machine learning?",
                    "top_k": 5,
                    "min_score": 0.0
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert_query_response(data, sample_query_data)
        assert_positive_number(data["response_time"])
        
        # Verify service was called
        mock_query_service.query_documents.assert_called_once()
    
    @unit_test
    def test_query_documents_empty_query(self, test_client, mock_query_service):
        """Test query with empty query string."""
        from app.core.exceptions import ValidationError
        
        # Mock validation error
        mock_query_service.query_documents.side_effect = ValidationError("Query cannot be empty")
        
        with patch('app.main.query_service', mock_query_service):
            response = test_client.post(
                "/api/v1/query",
                json={"query": "", "top_k": 5}
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "Query cannot be empty" in data["detail"]
    
    @unit_test
    def test_query_documents_too_long(self, test_client, mock_query_service):
        """Test query with too long query string."""
        from app.core.exceptions import ValidationError
        
        # Mock validation error
        mock_query_service.query_documents.side_effect = ValidationError("Query too long")
        
        with patch('app.main.query_service', mock_query_service):
            response = test_client.post(
                "/api/v1/query",
                json={"query": "x" * 2000, "top_k": 5}
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "Query too long" in data["detail"]
    
    @unit_test
    def test_query_documents_with_filters(self, test_client, mock_query_service, sample_query_data):
        """Test query with filters."""
        # Mock query service
        mock_query_service.query_documents.return_value = sample_query_data
        
        with patch('app.main.query_service', mock_query_service):
            response = test_client.post(
                "/api/v1/query",
                json={
                    "query": "machine learning",
                    "top_k": 3,
                    "min_score": 0.8,
                    "filters": {"document_type": "pdf"}
                }
            )
        
        assert response.status_code == 200
        
        # Verify service was called with correct parameters
        mock_query_service.query_documents.assert_called_once_with(
            query="machine learning",
            top_k=3,
            min_score=0.8,
            filters={"document_type": "pdf"},
            user_id="test_user",
            db=None
        )
    
    @unit_test
    def test_query_documents_no_results(self, test_client, mock_query_service):
        """Test query with no results."""
        # Mock query service with no results
        mock_query_service.query_documents.return_value = {
            "query": "nonexistent topic",
            "results": [],
            "total_results": 0,
            "response_time": 0.05
        }
        
        with patch('app.main.query_service', mock_query_service):
            response = test_client.post(
                "/api/v1/query",
                json={"query": "nonexistent topic", "top_k": 5}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 0
        assert data["results"] == []


class TestAnalyticsEndpoints:
    """Test analytics and statistics endpoints."""
    
    @unit_test
    def test_get_system_stats_success(self, test_client, mock_analytics_service, sample_stats_data):
        """Test successful system statistics retrieval."""
        # Mock analytics service
        mock_analytics_service.get_system_stats.return_value = sample_stats_data
        
        with patch('app.main.analytics_service', mock_analytics_service):
            response = test_client.get("/api/v1/analytics/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 10
        assert data["total_queries"] == 100
        assert_positive_number(data["avg_response_time"])
        assert_positive_number(data["storage_used"])
        assert_valid_timestamp(data["last_updated"])
        
        # Verify service was called
        mock_analytics_service.get_system_stats.assert_called_once()
    
    @unit_test
    def test_get_system_stats_error(self, test_client, mock_analytics_service):
        """Test system statistics retrieval error."""
        from app.core.exceptions import ProcessingError
        
        # Mock processing error
        mock_analytics_service.get_system_stats.side_effect = ProcessingError("Stats unavailable")
        
        with patch('app.main.analytics_service', mock_analytics_service):
            response = test_client.get("/api/v1/analytics/stats")
        
        assert response.status_code == 500
        data = response.json()
        assert "Stats unavailable" in data["detail"]


class TestConfigurationEndpoints:
    """Test configuration endpoints."""
    
    @unit_test
    def test_get_configuration(self, test_client):
        """Test configuration retrieval."""
        response = test_client.get("/api/v1/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "embedding_model" in data
        assert "llm_model" in data
        assert "chunk_size" in data
        assert "chunk_overlap" in data
        assert "max_query_length" in data
        assert "features" in data
        
        # Verify feature flags
        features = data["features"]
        assert "query_expansion" in features
        assert "reranking" in features
        assert "caching" in features


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @unit_test
    def test_404_not_found(self, test_client):
        """Test 404 error handling."""
        response = test_client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404
    
    @unit_test
    def test_method_not_allowed(self, test_client):
        """Test 405 method not allowed."""
        response = test_client.patch("/api/v1/documents")
        
        assert response.status_code == 405
    
    @unit_test
    def test_invalid_json(self, test_client):
        """Test invalid JSON handling."""
        response = test_client.post(
            "/api/v1/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    @unit_test
    def test_missing_required_fields(self, test_client):
        """Test missing required fields."""
        response = test_client.post(
            "/api/v1/query",
            json={"top_k": 5}  # Missing required 'query' field
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "validation error" in data["detail"][0]["type"]
    
    @unit_test
    def test_invalid_field_types(self, test_client):
        """Test invalid field types."""
        response = test_client.post(
            "/api/v1/query",
            json={
                "query": "test",
                "top_k": "invalid"  # Should be integer
            }
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "validation error" in data["detail"][0]["type"]
    
    @unit_test
    def test_internal_server_error(self, test_client, mock_document_service):
        """Test internal server error handling."""
        # Mock unexpected error
        mock_document_service.list_documents.side_effect = Exception("Unexpected error")
        
        with patch('app.main.document_service', mock_document_service):
            response = test_client.get("/api/v1/documents")
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"


class TestCORSAndMiddleware:
    """Test CORS and middleware functionality."""
    
    @unit_test
    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        response = test_client.options("/api/v1/documents")
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
    
    @unit_test
    def test_cors_preflight(self, test_client):
        """Test CORS preflight requests."""
        response = test_client.options(
            "/api/v1/documents",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


class TestAuthentication:
    """Test authentication and authorization."""
    
    @unit_test
    def test_get_current_user(self, test_client):
        """Test current user retrieval."""
        # Current implementation returns test user
        with patch('app.main.get_current_user') as mock_get_user:
            mock_get_user.return_value = {"user_id": "test_user", "username": "test_user"}
            
            response = test_client.get("/api/v1/documents")
            
            assert response.status_code == 200
            mock_get_user.assert_called()


@integration_test
class TestIntegrationScenarios:
    """Integration test scenarios."""
    
    async def test_full_document_workflow(self, test_client):
        """Test full document upload and query workflow."""
        # This would be a more complex integration test
        # that tests the full workflow end-to-end
        pass
    
    async def test_concurrent_uploads(self, test_client):
        """Test concurrent document uploads."""
        # Test concurrent upload handling
        pass
    
    async def test_large_document_processing(self, test_client):
        """Test large document processing."""
        # Test handling of large files
        pass
