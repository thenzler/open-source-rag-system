"""
Service Tests for the RAG System
Test core services functionality
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


class TestMetricsService:
    """Test metrics service functionality."""
    
    def test_metrics_service_import(self):
        """Test that metrics service can be imported."""
        from core.services.metrics_service import init_metrics_service
        assert init_metrics_service is not None
    
    def test_metrics_middleware_import(self):
        """Test that metrics middleware can be imported."""
        from core.middleware.metrics_middleware import MetricsMiddleware
        assert MetricsMiddleware is not None


class TestAsyncProcessingService:
    """Test async processing service."""
    
    def test_async_processing_import(self):
        """Test that async processing can be imported."""
        from core.services.async_processing_service import AsyncDocumentProcessor
        assert AsyncDocumentProcessor is not None
    
    @pytest.mark.asyncio
    async def test_async_processor_creation(self):
        """Test creating an async processor."""
        from core.services.async_processing_service import AsyncDocumentProcessor
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            processor = AsyncDocumentProcessor(queue_persistence_file=tmp.name)
            assert processor is not None
            
            # Cleanup
            Path(tmp.name).unlink(missing_ok=True)


class TestComplianceService:
    """Test compliance service functionality."""
    
    def test_compliance_service_import(self):
        """Test that compliance service can be imported."""
        from core.services.compliance_service import SwissDataProtectionService
        assert SwissDataProtectionService is not None
    
    def test_compliance_service_creation(self):
        """Test creating a compliance service."""
        from core.services.compliance_service import SwissDataProtectionService
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = SwissDataProtectionService(
                storage_path=tmp_dir,
                enable_audit_logging=False
            )
            assert service is not None


class TestOptionalServices:
    """Test optional services that may not be available."""
    
    def test_progress_tracking_optional_import(self):
        """Test progress tracking service import (optional)."""
        try:
            from core.services.progress_tracking_service import ProgressTracker
            assert ProgressTracker is not None
        except ImportError:
            # This is acceptable - service is optional
            pytest.skip("Progress tracking service not available")
    
    def test_redis_cache_optional_import(self):
        """Test Redis cache service import (optional)."""
        try:
            from core.services.redis_cache_service import RedisCacheService
            assert RedisCacheService is not None
        except ImportError:
            # This is acceptable - service is optional
            pytest.skip("Redis cache service not available")


class TestSecurityUtils:
    """Test security utilities."""
    
    def test_security_utils_import(self):
        """Test security utilities can be imported."""
        from core.utils.security import initialize_id_obfuscator
        from core.utils.encryption import setup_encryption_from_config
        
        assert initialize_id_obfuscator is not None
        assert setup_encryption_from_config is not None
    
    def test_id_obfuscator_initialization(self):
        """Test ID obfuscator initialization."""
        from core.utils.security import initialize_id_obfuscator
        
        # Should not raise exception
        initialize_id_obfuscator("test-secret-key")


class TestMiddleware:
    """Test middleware components."""
    
    def test_tenant_middleware_import(self):
        """Test tenant middleware can be imported."""
        from core.middleware import tenant_middleware, initialize_tenant_resolver
        assert tenant_middleware is not None
        assert initialize_tenant_resolver is not None
    
    def test_metrics_middleware_import(self):
        """Test metrics middleware can be imported."""
        from core.middleware.metrics_middleware import MetricsMiddleware
        assert MetricsMiddleware is not None


class TestRepositories:
    """Test repository components."""
    
    def test_tenant_repository_import(self):
        """Test tenant repository can be imported."""
        from core.repositories.tenant_repository import TenantRepository
        assert TenantRepository is not None
    
    def test_tenant_repository_creation(self):
        """Test creating a tenant repository."""
        from core.repositories.tenant_repository import TenantRepository
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            repo = TenantRepository(tmp.name)
            assert repo is not None
            
            # Cleanup
            Path(tmp.name).unlink(missing_ok=True)


class TestProcessors:
    """Test document processors."""
    
    def test_document_processors_import(self):
        """Test document processors can be imported."""
        from core.processors.document_processors import DocumentProcessors
        assert DocumentProcessors is not None
    
    def test_register_processors_import(self):
        """Test register processors function can be imported."""
        from core.processors import register_document_processors
        assert register_document_processors is not None


class TestDependencyInjection:
    """Test dependency injection system."""
    
    def test_di_services_import(self):
        """Test DI services can be imported."""
        from core.di.services import ServiceConfiguration, initialize_services, shutdown_services
        assert ServiceConfiguration is not None
        assert initialize_services is not None
        assert shutdown_services is not None
    
    @pytest.mark.asyncio
    async def test_service_configuration(self):
        """Test service configuration."""
        from core.di.services import ServiceConfiguration
        
        # Should not raise exception
        ServiceConfiguration.configure_all()


if __name__ == "__main__":
    pytest.main([__file__])