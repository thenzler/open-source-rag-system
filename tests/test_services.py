"""
Service Tests for the RAG System
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import os
import tempfile
from pathlib import Path

from app.services.document_service import DocumentService
from app.services.query_service import QueryService
from app.services.analytics_service import AnalyticsService
from app.core.exceptions import DocumentNotFoundError, ValidationError, ProcessingError
from tests.conftest import (
    unit_test,
    integration_test,
    assert_valid_uuid,
    assert_positive_number,
    setup_test_data
)


class TestDocumentService:
    """Test document service functionality."""
    
    @unit_test
    async def test_initialize_service(self):
        """Test service initialization."""
        service = DocumentService()
        await service.initialize()
        
        # Check that upload directory was created
        assert service.upload_directory.exists()
        assert service.upload_directory.is_dir()
    
    @unit_test
    def test_get_file_type(self):
        """Test file type detection."""
        service = DocumentService()
        
        assert service._get_file_type("test.pdf") == "pdf"
        assert service._get_file_type("test.docx") == "word"
        assert service._get_file_type("test.txt") == "text"
        assert service._get_file_type("test.unknown") == "unknown"
        assert service._get_file_type("TEST.PDF") == "pdf"  # Case insensitive
    
    @unit_test
    def test_validate_file_success(self, sample_upload_file):
        """Test successful file validation."""
        service = DocumentService()
        
        result = service._validate_file(sample_upload_file)
        
        assert result["file_type"] == "text"
        assert result["size"] >= 0
        assert result["mime_type"] == "text/plain"
    
    @unit_test
    def test_validate_file_no_filename(self):
        """Test file validation with no filename."""
        from fastapi import UploadFile
        from io import BytesIO
        
        service = DocumentService()
        file_like = BytesIO(b"content")
        upload_file = UploadFile(filename=None, file=file_like)
        
        with pytest.raises(ValidationError, match="No filename provided"):
            service._validate_file(upload_file)
    
    @unit_test
    def test_validate_file_unsupported_type(self):
        """Test file validation with unsupported type."""
        from fastapi import UploadFile
        from io import BytesIO
        
        service = DocumentService()
        file_like = BytesIO(b"content")
        upload_file = UploadFile(filename="test.xyz", file=file_like)
        
        with pytest.raises(ValidationError, match="Unsupported file type"):
            service._validate_file(upload_file)
    
    @unit_test
    def test_validate_file_too_large(self):
        """Test file validation with file too large."""
        from fastapi import UploadFile
        from io import BytesIO
        
        service = DocumentService()
        service.max_file_size = 100  # 100 bytes limit
        
        file_like = BytesIO(b"x" * 200)  # 200 bytes
        upload_file = UploadFile(filename="test.txt", file=file_like, size=200)
        
        with pytest.raises(ValidationError, match="File too large"):
            service._validate_file(upload_file)
    
    @unit_test
    async def test_upload_document_success(self, sample_upload_file, test_db):
        """Test successful document upload."""
        service = DocumentService()
        await service.initialize()
        
        # Mock file processing
        with patch('aiofiles.open', create=True) as mock_open:
            mock_file = AsyncMock()
            mock_open.return_value.__aenter__.return_value = mock_file
            
            document = await service.upload_document(
                file=sample_upload_file,
                metadata="test metadata",
                user_id="test_user",
                db=test_db
            )
        
        assert document.original_filename == "test.txt"
        assert document.file_type == "text"
        assert document.user_id == "test_user"
        assert document.status == "pending"
        assert_valid_uuid(document.id)
    
    @unit_test
    async def test_upload_document_file_error(self, sample_upload_file, test_db):
        """Test document upload with file error."""
        service = DocumentService()
        await service.initialize()
        
        # Mock file write error
        with patch('aiofiles.open', side_effect=OSError("File write error")):
            with pytest.raises(ProcessingError, match="Failed to upload document"):
                await service.upload_document(
                    file=sample_upload_file,
                    user_id="test_user",
                    db=test_db
                )
    
    @unit_test
    async def test_get_document_success(self, test_db):
        """Test successful document retrieval."""
        service = DocumentService()
        
        # Set up test data
        test_data = await setup_test_data(test_db)
        document = test_data["documents"][0]
        
        result = await service.get_document(
            document_id=document.id,
            user_id=document.user_id,
            db=test_db
        )
        
        assert result.id == document.id
        assert result.user_id == document.user_id
    
    @unit_test
    async def test_get_document_not_found(self, test_db):
        """Test document retrieval for non-existent document."""
        service = DocumentService()
        
        with pytest.raises(DocumentNotFoundError):
            await service.get_document(
                document_id="nonexistent",
                user_id="test_user",
                db=test_db
            )
    
    @unit_test
    async def test_list_documents_success(self, test_db):
        """Test successful document listing."""
        service = DocumentService()
        
        # Set up test data
        test_data = await setup_test_data(test_db)
        
        documents, total = await service.list_documents(
            skip=0,
            limit=10,
            user_id="test_user",
            db=test_db
        )
        
        assert len(documents) == 2
        assert total == 2
        assert all(doc.user_id == "test_user" for doc in documents)
    
    @unit_test
    async def test_list_documents_with_filters(self, test_db):
        """Test document listing with filters."""
        service = DocumentService()
        
        # Set up test data
        test_data = await setup_test_data(test_db)
        
        # Filter by status
        documents, total = await service.list_documents(
            skip=0,
            limit=10,
            status="completed",
            user_id="test_user",
            db=test_db
        )
        
        assert len(documents) == 1
        assert documents[0].status == "completed"
        
        # Filter by file type
        documents, total = await service.list_documents(
            skip=0,
            limit=10,
            category="pdf",
            user_id="test_user",
            db=test_db
        )
        
        assert len(documents) == 1
        assert documents[0].file_type == "pdf"
    
    @unit_test
    async def test_list_documents_with_search(self, test_db):
        """Test document listing with search."""
        service = DocumentService()
        
        # Set up test data
        test_data = await setup_test_data(test_db)
        
        # Search by filename
        documents, total = await service.list_documents(
            skip=0,
            limit=10,
            search="test1",
            user_id="test_user",
            db=test_db
        )
        
        assert len(documents) == 1
        assert "test1" in documents[0].original_filename
    
    @unit_test
    async def test_delete_document_success(self, test_db):
        """Test successful document deletion."""
        service = DocumentService()
        
        # Set up test data
        test_data = await setup_test_data(test_db)
        document = test_data["documents"][0]
        
        # Mock external service calls
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"deleted_count": 5}
            mock_client.return_value.__aenter__.return_value.delete.return_value = mock_response
            
            # Mock file deletion
            with patch('aiofiles.os.remove') as mock_remove:
                result = await service.delete_document(
                    document_id=document.id,
                    user_id=document.user_id,
                    db=test_db
                )
        
        assert result["deleted_chunks"] == 0  # No chunks in test data
        assert result["deleted_vectors"] == 5
    
    @unit_test
    async def test_delete_document_not_found(self, test_db):
        """Test document deletion for non-existent document."""
        service = DocumentService()
        
        with pytest.raises(DocumentNotFoundError):
            await service.delete_document(
                document_id="nonexistent",
                user_id="test_user",
                db=test_db
            )
    
    @unit_test
    async def test_process_document_async(self, test_db):
        """Test asynchronous document processing."""
        service = DocumentService()
        
        # Set up test data
        test_data = await setup_test_data(test_db)
        document = test_data["documents"][0]
        
        # Mock external service call
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"document_id": document.id, "status": "queued"}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            await service.process_document_async(document.id, test_db)
        
        # Check that document status was updated
        await test_db.refresh(document)
        assert document.status == "processing"
    
    @unit_test
    async def test_health_check_success(self):
        """Test successful health check."""
        service = DocumentService()
        await service.initialize()
        
        # Mock external service call
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await service.health_check()
        
        assert result is True
    
    @unit_test
    async def test_health_check_failure(self):
        """Test failed health check."""
        service = DocumentService()
        await service.initialize()
        
        # Mock external service failure
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await service.health_check()
        
        assert result is False


class TestQueryService:
    """Test query service functionality."""
    
    @unit_test
    async def test_initialize_service(self):
        """Test service initialization."""
        service = QueryService()
        
        # Mock health check calls
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            await service.initialize()
        
        # Service should be initialized
        assert service.vector_engine_url is not None
        assert service.llm_service_url is not None
    
    @unit_test
    def test_validate_query_success(self):
        """Test successful query validation."""
        service = QueryService()
        
        query = "What is machine learning?"
        result = service._validate_query(query)
        
        assert result == query
    
    @unit_test
    def test_validate_query_empty(self):
        """Test query validation with empty query."""
        service = QueryService()
        
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            service._validate_query("")
        
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            service._validate_query("   ")
    
    @unit_test
    def test_validate_query_too_long(self):
        """Test query validation with too long query."""
        service = QueryService()
        service.max_query_length = 100
        
        long_query = "x" * 200
        with pytest.raises(ValidationError, match="Query too long"):
            service._validate_query(long_query)
    
    @unit_test
    async def test_expand_query_disabled(self):
        """Test query expansion when disabled."""
        service = QueryService()
        service.enable_query_expansion = False
        
        query = "machine learning"
        result = await service._expand_query(query)
        
        assert result == [query]
    
    @unit_test
    async def test_expand_query_enabled(self):
        """Test query expansion when enabled."""
        service = QueryService()
        service.enable_query_expansion = True
        
        # Mock LLM service response
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "artificial intelligence\nAI algorithms\nneural networks"
            }
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            query = "machine learning"
            result = await service._expand_query(query)
        
        assert len(result) > 1
        assert query in result
        assert "artificial intelligence" in result
    
    @unit_test
    async def test_search_vectors_success(self, sample_vector_search_response):
        """Test successful vector search."""
        service = QueryService()
        
        # Mock vector engine response
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_vector_search_response
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            result = await service._search_vectors("test query", top_k=5)
        
        assert len(result) == 2
        assert all("id" in item for item in result)
        assert all("score" in item for item in result)
        assert all("content" in item for item in result)
    
    @unit_test
    async def test_search_vectors_failure(self):
        """Test vector search failure."""
        service = QueryService()
        
        # Mock vector engine failure
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal server error"
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            result = await service._search_vectors("test query", top_k=5)
        
        assert result == []
    
    @unit_test
    async def test_query_documents_success(self, test_db, sample_document_data):
        """Test successful document querying."""
        service = QueryService()
        
        # Mock vector search
        with patch.object(service, '_search_vectors') as mock_search:
            mock_search.return_value = [
                {
                    "id": "chunk-1",
                    "score": 0.95,
                    "content": "Machine learning content",
                    "metadata": {"document_id": "doc-1", "chunk_index": 0}
                }
            ]
            
            # Mock document info retrieval
            with patch.object(service, '_get_document_info') as mock_doc_info:
                mock_doc_info.return_value = {
                    "doc-1": MagicMock(**sample_document_data)
                }
                
                result = await service.query_documents(
                    query="machine learning",
                    top_k=5,
                    user_id="test_user",
                    db=test_db
                )
        
        assert result["query"] == "machine learning"
        assert result["total_results"] == 1
        assert len(result["results"]) == 1
        assert_positive_number(result["response_time"])
    
    @unit_test
    async def test_query_documents_no_results(self, test_db):
        """Test document querying with no results."""
        service = QueryService()
        
        # Mock empty vector search
        with patch.object(service, '_search_vectors') as mock_search:
            mock_search.return_value = []
            
            result = await service.query_documents(
                query="nonexistent topic",
                top_k=5,
                user_id="test_user",
                db=test_db
            )
        
        assert result["query"] == "nonexistent topic"
        assert result["total_results"] == 0
        assert result["results"] == []
    
    @unit_test
    async def test_query_documents_with_filters(self, test_db):
        """Test document querying with filters."""
        service = QueryService()
        
        # Mock vector search
        with patch.object(service, '_search_vectors') as mock_search:
            mock_search.return_value = []
            
            filters = {"document_type": "pdf"}
            await service.query_documents(
                query="test",
                top_k=5,
                filters=filters,
                user_id="test_user",
                db=test_db
            )
        
        # Verify filters were passed to vector search
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert call_args[1]["filters"]["document_type"] == "pdf"
        assert call_args[1]["filters"]["user_id"] == "test_user"
    
    @unit_test
    async def test_health_check_success(self):
        """Test successful health check."""
        service = QueryService()
        
        # Mock vector engine health check
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await service.health_check()
        
        assert result is True
    
    @unit_test
    async def test_health_check_failure(self):
        """Test failed health check."""
        service = QueryService()
        
        # Mock vector engine failure
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await service.health_check()
        
        assert result is False
    
    @unit_test
    async def test_check_llm_health_success(self):
        """Test successful LLM health check."""
        service = QueryService()
        
        # Mock LLM service health check
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await service.check_llm_health()
        
        assert result is True
    
    @unit_test
    async def test_check_llm_health_failure(self):
        """Test failed LLM health check."""
        service = QueryService()
        
        # Mock LLM service failure
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection failed")
            
            result = await service.check_llm_health()
        
        assert result is False


class TestAnalyticsService:
    """Test analytics service functionality."""
    
    @unit_test
    async def test_initialize_service(self):
        """Test service initialization."""
        service = AnalyticsService()
        await service.initialize()
        
        # Service should be initialized
        assert service.retention_days == 90
    
    @unit_test
    async def test_log_query_success(self, test_db, sample_query_data):
        """Test successful query logging."""
        service = AnalyticsService()
        
        query_log = await service.log_query(
            query="test query",
            user_id="test_user",
            response=sample_query_data,
            db=test_db
        )
        
        assert query_log.query_text == "test query"
        assert query_log.user_id == "test_user"
        assert query_log.results_count == sample_query_data["total_results"]
        assert query_log.response_time == sample_query_data["response_time"]
        assert_valid_uuid(str(query_log.id))
    
    @unit_test
    async def test_get_system_stats_success(self, test_db):
        """Test successful system stats retrieval."""
        service = AnalyticsService()
        
        # Set up test data
        test_data = await setup_test_data(test_db)
        
        stats = await service.get_system_stats(test_db)
        
        assert "total_documents" in stats
        assert "total_queries" in stats
        assert "avg_response_time" in stats
        assert "storage_used" in stats
        assert "last_updated" in stats
        assert stats["total_documents"] == 2
        assert stats["total_queries"] == 2
    
    @unit_test
    async def test_get_document_stats(self, test_db):
        """Test document statistics retrieval."""
        service = AnalyticsService()
        
        # Set up test data
        test_data = await setup_test_data(test_db)
        
        stats = await service._get_document_stats(test_db)
        
        assert stats["total_documents"] == 2
        assert "completed" in stats["documents_by_status"]
        assert "processing" in stats["documents_by_status"]
        assert "pdf" in stats["documents_by_type"]
        assert "text" in stats["documents_by_type"]
        assert_positive_number(stats["processing_success_rate"])
    
    @unit_test
    async def test_get_query_stats(self, test_db):
        """Test query statistics retrieval."""
        service = AnalyticsService()
        
        # Set up test data
        test_data = await setup_test_data(test_db)
        
        stats = await service._get_query_stats(test_db)
        
        assert stats["total_queries"] == 2
        assert_positive_number(stats["avg_results_per_query"])
        assert_positive_number(stats["query_success_rate"])
    
    @unit_test
    async def test_get_time_series_data(self, test_db):
        """Test time series data retrieval."""
        service = AnalyticsService()
        
        # Set up test data
        test_data = await setup_test_data(test_db)
        
        # Test document uploads time series
        data = await service.get_time_series_data(
            metric="document_uploads",
            days=30,
            db=test_db
        )
        
        assert isinstance(data, list)
        if data:  # If there's data
            assert "date" in data[0]
            assert "value" in data[0]
    
    @unit_test
    async def test_get_popular_queries(self, test_db):
        """Test popular queries retrieval."""
        service = AnalyticsService()
        
        # Set up test data
        test_data = await setup_test_data(test_db)
        
        popular = await service.get_popular_queries(
            limit=10,
            days=30,
            db=test_db
        )
        
        assert isinstance(popular, list)
        assert len(popular) <= 10
        
        if popular:  # If there are popular queries
            assert "query" in popular[0]
            assert "count" in popular[0]
            assert "avg_response_time" in popular[0]
            assert "avg_results" in popular[0]
    
    @unit_test
    async def test_cleanup_old_logs(self, test_db):
        """Test old log cleanup."""
        service = AnalyticsService()
        service.retention_days = 1  # Very short retention for testing
        
        # Create old log entry
        from app.models.queries import QueryLog
        old_log = QueryLog(
            query_text="old query",
            user_id="test_user",
            results_count=5,
            response_time=0.1,
            created_at=datetime.utcnow() - timedelta(days=2)
        )
        test_db.add(old_log)
        await test_db.commit()
        
        # Run cleanup
        deleted_count = await service.cleanup_old_logs(test_db)
        
        assert deleted_count >= 1


@integration_test
class TestServiceIntegration:
    """Integration tests for service interactions."""
    
    async def test_document_upload_and_query_flow(self, test_db):
        """Test complete document upload and query flow."""
        document_service = DocumentService()
        query_service = QueryService()
        analytics_service = AnalyticsService()
        
        # Initialize services
        await document_service.initialize()
        await query_service.initialize()
        await analytics_service.initialize()
        
        # This would test the full flow with actual service interactions
        # but requires running services, so we'll mock the external calls
        
        # Mock external services
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"results": []}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            # Test document processing workflow
            # This would involve actual file processing, vector storage, etc.
            pass
    
    async def test_service_health_checks(self):
        """Test all service health checks."""
        document_service = DocumentService()
        query_service = QueryService()
        analytics_service = AnalyticsService()
        
        # Mock external service calls
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            # Test health checks
            doc_health = await document_service.health_check()
            query_health = await query_service.health_check()
            llm_health = await query_service.check_llm_health()
            
            assert doc_health is True
            assert query_health is True
            assert llm_health is True
    
    async def test_concurrent_operations(self, test_db):
        """Test concurrent service operations."""
        import asyncio
        
        document_service = DocumentService()
        query_service = QueryService()
        
        # Mock external services
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"results": []}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            # Test concurrent queries
            tasks = []
            for i in range(5):
                task = query_service.query_documents(
                    query=f"test query {i}",
                    top_k=5,
                    user_id="test_user",
                    db=test_db
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            assert all(isinstance(result, dict) for result in results)
