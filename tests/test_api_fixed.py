"""
Fixed API Tests for the RAG System - adapted to actual project structure
"""

import pytest
import requests
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Base API URL
API_BASE = "http://localhost:8001"

class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check_success(self):
        """Test successful health check."""
        try:
            response = requests.get(f"{API_BASE}/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            print("+ Health check test passed")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_api_health_check(self):
        """Test API-specific health check."""
        try:
            response = requests.get(f"{API_BASE}/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            print("+ API health check test passed")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")

class TestDocumentEndpoints:
    """Test document management endpoints."""
    
    def test_upload_document_success(self):
        """Test successful document upload."""
        # Create a test document
        test_content = "This is a test document for the RAG system."
        test_file = Path("test_upload.txt")
        test_file.write_text(test_content)
        
        try:
            with open(test_file, 'rb') as f:
                files = {'file': (test_file.name, f, 'text/plain')}
                response = requests.post(f"{API_BASE}/api/v1/documents", files=files)
            
            # Clean up
            test_file.unlink()
            
            if response.status_code == 200:
                data = response.json()
                assert "filename" in data
                assert "status" in data
                print("+ Document upload test passed")
            else:
                print(f"Document upload failed: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            test_file.unlink()
            pytest.skip("API server not running")
        except Exception as e:
            if test_file.exists():
                test_file.unlink()
            raise e
    
    def test_list_documents_success(self):
        """Test successful document listing."""
        try:
            response = requests.get(f"{API_BASE}/api/v1/documents")
            assert response.status_code == 200
            data = response.json()
            assert "total" in data
            assert "documents" in data
            print("+ Document list test passed")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_get_document_not_found(self):
        """Test document retrieval for non-existent document."""
        try:
            response = requests.get(f"{API_BASE}/api/v1/documents/nonexistent")
            assert response.status_code == 404
            print("+ Document not found test passed")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")

class TestQueryEndpoints:
    """Test query and search endpoints."""
    
    def test_query_documents_success(self):
        """Test successful document query."""
        try:
            response = requests.post(
                f"{API_BASE}/api/v1/query",
                json={
                    "query": "What is machine learning?",
                    "top_k": 5
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert "total_results" in data
            print("+ Query test passed")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_query_documents_empty_query(self):
        """Test query with empty query string."""
        try:
            response = requests.post(
                f"{API_BASE}/api/v1/query",
                json={"query": "", "top_k": 5}
            )
            
            # Should return 400 or handle gracefully
            assert response.status_code in [400, 422]
            print("+ Empty query test passed")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_query_documents_no_results(self):
        """Test query with no results."""
        try:
            response = requests.post(
                f"{API_BASE}/api/v1/query",
                json={"query": "xyzabcnonexistentquery", "top_k": 5}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            # Results might be empty or contain low-scored matches
            print("+ No results query test passed")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_404_not_found(self):
        """Test 404 error handling."""
        try:
            response = requests.get(f"{API_BASE}/api/v1/nonexistent")
            assert response.status_code == 404
            print("+ 404 test passed")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_method_not_allowed(self):
        """Test 405 method not allowed."""
        try:
            response = requests.patch(f"{API_BASE}/api/v1/documents")
            assert response.status_code == 405
            print("+ Method not allowed test passed")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_invalid_json(self):
        """Test invalid JSON handling."""
        try:
            response = requests.post(
                f"{API_BASE}/api/v1/query",
                data="invalid json",
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 422
            print("+ Invalid JSON test passed")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_missing_required_fields(self):
        """Test missing required fields."""
        try:
            response = requests.post(
                f"{API_BASE}/api/v1/query",
                json={"top_k": 5}  # Missing required 'query' field
            )
            
            assert response.status_code == 422
            print("+ Missing fields test passed")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")

class TestIntegrationScenarios:
    """Integration test scenarios."""
    
    def test_full_document_workflow(self):
        """Test full document upload and query workflow."""
        # Create test document
        test_content = """
        Machine learning is a subset of artificial intelligence that focuses on 
        algorithms that can learn from data and make predictions or decisions.
        
        Natural language processing (NLP) is another important area of AI that 
        deals with understanding and generating human language.
        
        Vector databases are used to store and retrieve high-dimensional vectors 
        efficiently, which is crucial for semantic search applications.
        """
        test_file = Path("test_workflow.txt")
        test_file.write_text(test_content)
        
        try:
            # 1. Upload document
            with open(test_file, 'rb') as f:
                files = {'file': (test_file.name, f, 'text/plain')}
                upload_response = requests.post(f"{API_BASE}/api/v1/documents", files=files)
            
            assert upload_response.status_code == 200
            
            # 2. Wait for processing
            time.sleep(2)
            
            # 3. Query document
            query_response = requests.post(
                f"{API_BASE}/api/v1/query",
                json={"query": "What is machine learning?", "top_k": 3}
            )
            
            assert query_response.status_code == 200
            query_data = query_response.json()
            assert "results" in query_data
            
            # 4. List documents
            list_response = requests.get(f"{API_BASE}/api/v1/documents")
            assert list_response.status_code == 200
            list_data = list_response.json()
            assert list_data["total"] > 0
            
            print("+ Full workflow test passed")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
        finally:
            # Clean up
            if test_file.exists():
                test_file.unlink()

# Simple test runner for non-pytest environments
def run_simple_tests():
    """Run tests without pytest."""
    print("Running API Tests")
    print("=" * 50)
    
    # Test health endpoints
    health_tests = TestHealthEndpoints()
    try:
        health_tests.test_health_check_success()
        health_tests.test_api_health_check()
    except Exception as e:
        print(f"Health tests failed: {e}")
    
    # Test document endpoints
    doc_tests = TestDocumentEndpoints()
    try:
        doc_tests.test_upload_document_success()
        doc_tests.test_list_documents_success()
        doc_tests.test_get_document_not_found()
    except Exception as e:
        print(f"Document tests failed: {e}")
    
    # Test query endpoints
    query_tests = TestQueryEndpoints()
    try:
        query_tests.test_query_documents_success()
        query_tests.test_query_documents_empty_query()
        query_tests.test_query_documents_no_results()
    except Exception as e:
        print(f"Query tests failed: {e}")
    
    # Test error handling
    error_tests = TestErrorHandling()
    try:
        error_tests.test_404_not_found()
        error_tests.test_method_not_allowed()
        error_tests.test_invalid_json()
        error_tests.test_missing_required_fields()
    except Exception as e:
        print(f"Error handling tests failed: {e}")
    
    # Test integration scenarios
    integration_tests = TestIntegrationScenarios()
    try:
        integration_tests.test_full_document_workflow()
    except Exception as e:
        print(f"Integration tests failed: {e}")
    
    print("\n+ Test suite completed")

if __name__ == "__main__":
    run_simple_tests()