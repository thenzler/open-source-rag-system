"""
Service Loader for Dependency Injection
Loads and initializes all services for the application
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Simple service container for dependency injection"""

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._initialized = False

    def register(self, name: str, service: Any):
        """Register a service"""
        self._services[name] = service
        logger.info(f"Registered service: {name}")

    def get(self, name: str) -> Any:
        """Get a service by name"""
        return self._services.get(name)

    def initialize_services(self):
        """Initialize all core services"""
        if self._initialized:
            return

        logger.info("Initializing services...")

        # TODO: Initialize actual services when we extract them from simple_api.py
        # For now, register placeholder services

        self.register("documents_storage", None)
        self.register("document_processor", None)
        self.register("file_validator", None)
        self.register("query_service", None)
        self.register("llm_service", None)
        self.register("vector_service", None)
        self.register("system_monitor", None)
        self.register("cache_manager", None)
        self.register("analytics_service", None)
        self.register("llm_manager", None)
        self.register("ollama_client", None)

        self._initialized = True
        logger.info("Services initialized successfully")

    def is_initialized(self) -> bool:
        """Check if services are initialized"""
        return self._initialized


# Global service container
services = ServiceContainer()


def get_service_container() -> ServiceContainer:
    """Get the global service container"""
    return services
