"""
Production Dependency Injection Container
Lightweight DI framework specifically designed for the RAG system
"""

import asyncio
import inspect
import logging
import threading
from enum import Enum
from functools import wraps
from typing import (Any, Callable, Dict, Optional, Type, TypeVar, Union,
                    get_type_hints)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Lifetime(Enum):
    """Service lifetime management"""

    SINGLETON = "singleton"  # One instance for the entire application
    SCOPED = "scoped"  # One instance per request/scope
    TRANSIENT = "transient"  # New instance every time


class ServiceDescriptor:
    """Describes how to create and manage a service"""

    def __init__(
        self,
        service_type: Type[T],
        implementation: Union[Type[T], Callable[[], T], T],
        lifetime: Lifetime = Lifetime.SINGLETON,
        dependencies: Optional[Dict[str, Type]] = None,
    ):
        self.service_type = service_type
        self.implementation = implementation
        self.lifetime = lifetime
        self.dependencies = dependencies or {}
        self.instance = None
        self._lock = threading.RLock()

    def __repr__(self):
        return f"ServiceDescriptor({self.service_type.__name__}, {self.lifetime.value})"


class DIContainer:
    """Dependency Injection Container for RAG System"""

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._lock = threading.RLock()
        self._initialization_order: list = []
        self._initialized = False

        logger.info("Initialized DI Container")

    def register_singleton(
        self, service_type: Type[T], implementation: Union[Type[T], Callable[[], T], T]
    ) -> "DIContainer":
        """Register a singleton service"""
        return self.register(service_type, implementation, Lifetime.SINGLETON)

    def register_scoped(
        self, service_type: Type[T], implementation: Union[Type[T], Callable[[], T]]
    ) -> "DIContainer":
        """Register a scoped service"""
        return self.register(service_type, implementation, Lifetime.SCOPED)

    def register_transient(
        self, service_type: Type[T], implementation: Union[Type[T], Callable[[], T]]
    ) -> "DIContainer":
        """Register a transient service"""
        return self.register(service_type, implementation, Lifetime.TRANSIENT)

    def register(
        self,
        service_type: Type[T],
        implementation: Union[Type[T], Callable[[], T], T],
        lifetime: Lifetime = Lifetime.SINGLETON,
    ) -> "DIContainer":
        """Register a service with the container"""

        with self._lock:
            # Auto-detect dependencies from type hints
            dependencies = {}

            if inspect.isclass(implementation) and hasattr(implementation, "__init__"):
                try:
                    type_hints = get_type_hints(implementation.__init__)
                    # Skip 'self' parameter
                    dependencies = {
                        name: hint
                        for name, hint in type_hints.items()
                        if name != "return"
                    }
                except Exception as e:
                    logger.debug(
                        f"Could not extract type hints for {implementation.__name__}: {e}"
                    )

            descriptor = ServiceDescriptor(
                service_type, implementation, lifetime, dependencies
            )
            self._services[service_type] = descriptor

            # Track registration order for initialization
            if service_type not in self._initialization_order:
                self._initialization_order.append(service_type)

            logger.debug(f"Registered {service_type.__name__} as {lifetime.value}")
            return self

    def register_instance(self, service_type: Type[T], instance: T) -> "DIContainer":
        """Register an existing instance as singleton"""
        with self._lock:
            descriptor = ServiceDescriptor(service_type, instance, Lifetime.SINGLETON)
            descriptor.instance = instance
            self._services[service_type] = descriptor
            self._instances[service_type] = instance

            if service_type not in self._initialization_order:
                self._initialization_order.append(service_type)

            logger.debug(f"Registered instance of {service_type.__name__}")
            return self

    def get(self, service_type: Type[T], scope_id: str = "default") -> T:
        """Get a service instance"""

        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} not registered")

        descriptor = self._services[service_type]

        # Handle different lifetimes
        if descriptor.lifetime == Lifetime.SINGLETON:
            return self._get_singleton(descriptor)
        elif descriptor.lifetime == Lifetime.SCOPED:
            return self._get_scoped(descriptor, scope_id)
        else:  # TRANSIENT
            return self._create_instance(descriptor)

    def get_optional(self, service_type_name: str, scope_id: str = "default"):
        """Get a service instance by string name, return None if not found"""
        try:
            for service_type in self._services:
                if service_type.__name__ == service_type_name:
                    return self.get(service_type, scope_id)
            return None
        except Exception:
            return None

    def _get_singleton(self, descriptor: ServiceDescriptor) -> Any:
        """Get or create singleton instance"""
        with descriptor._lock:
            if descriptor.instance is None:
                descriptor.instance = self._create_instance(descriptor)
                self._instances[descriptor.service_type] = descriptor.instance
                logger.debug(
                    f"Created singleton instance of {descriptor.service_type.__name__}"
                )
            return descriptor.instance

    def _get_scoped(self, descriptor: ServiceDescriptor, scope_id: str) -> Any:
        """Get or create scoped instance"""
        with self._lock:
            if scope_id not in self._scoped_instances:
                self._scoped_instances[scope_id] = {}

            scope_instances = self._scoped_instances[scope_id]

            if descriptor.service_type not in scope_instances:
                instance = self._create_instance(descriptor)
                scope_instances[descriptor.service_type] = instance
                logger.debug(
                    f"Created scoped instance of {descriptor.service_type.__name__} for scope {scope_id}"
                )

            return scope_instances[descriptor.service_type]

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create a new instance of the service"""
        implementation = descriptor.implementation

        # If it's already an instance, return it
        if not inspect.isclass(implementation) and not callable(implementation):
            return implementation

        # If it's a factory function
        if callable(implementation) and not inspect.isclass(implementation):
            try:
                return implementation()
            except Exception as e:
                logger.error(
                    f"Error creating instance via factory for {descriptor.service_type.__name__}: {e}"
                )
                raise

        # If it's a class, create instance with dependency injection
        if inspect.isclass(implementation):
            try:
                # Get constructor parameters
                sig = inspect.signature(implementation.__init__)
                kwargs = {}

                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue

                    # Try to resolve dependency
                    if param.annotation != inspect.Parameter.empty:
                        try:
                            dependency = self.get(param.annotation)
                            kwargs[param_name] = dependency
                        except ValueError:
                            # Dependency not registered, use default if available
                            if param.default != inspect.Parameter.empty:
                                kwargs[param_name] = param.default
                            else:
                                logger.warning(
                                    f"Could not resolve dependency {param.annotation.__name__} for {implementation.__name__}"
                                )

                return implementation(**kwargs)

            except Exception as e:
                logger.error(
                    f"Error creating instance of {implementation.__name__}: {e}"
                )
                raise

        raise ValueError(
            f"Cannot create instance for {descriptor.service_type.__name__}"
        )

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service is registered"""
        return service_type in self._services

    def clear_scope(self, scope_id: str):
        """Clear all scoped instances for a scope"""
        with self._lock:
            if scope_id in self._scoped_instances:
                count = len(self._scoped_instances[scope_id])
                del self._scoped_instances[scope_id]
                logger.debug(f"Cleared {count} scoped instances for scope {scope_id}")

    async def initialize_all(self) -> bool:
        """Initialize all singleton services in order"""
        try:
            with self._lock:
                if self._initialized:
                    return True

                logger.info("Initializing all services...")

                for service_type in self._initialization_order:
                    descriptor = self._services.get(service_type)
                    if descriptor and descriptor.lifetime == Lifetime.SINGLETON:
                        try:
                            instance = self.get(service_type)

                            # Call initialize method if it exists
                            if hasattr(instance, "initialize") and callable(
                                getattr(instance, "initialize")
                            ):
                                if asyncio.iscoroutinefunction(instance.initialize):
                                    await instance.initialize()
                                else:
                                    instance.initialize()

                                logger.debug(f"Initialized {service_type.__name__}")

                        except Exception as e:
                            logger.error(
                                f"Failed to initialize {service_type.__name__}: {e}"
                            )
                            return False

                self._initialized = True
                logger.info(
                    f"Successfully initialized {len(self._initialization_order)} services"
                )
                return True

        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            return False

    async def shutdown_all(self):
        """Shutdown all services"""
        try:
            logger.info("Shutting down all services...")

            # Shutdown in reverse order
            for service_type in reversed(self._initialization_order):
                instance = self._instances.get(service_type)
                if (
                    instance
                    and hasattr(instance, "shutdown")
                    and callable(getattr(instance, "shutdown"))
                ):
                    try:
                        if asyncio.iscoroutinefunction(instance.shutdown):
                            await instance.shutdown()
                        else:
                            instance.shutdown()

                        logger.debug(f"Shutdown {service_type.__name__}")
                    except Exception as e:
                        logger.error(
                            f"Error shutting down {service_type.__name__}: {e}"
                        )

            # Clear all instances
            with self._lock:
                self._instances.clear()
                self._scoped_instances.clear()
                self._initialized = False

            logger.info("All services shutdown completed")

        except Exception as e:
            logger.error(f"Service shutdown failed: {e}")

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about all registered services"""
        return {
            "total_services": len(self._services),
            "singleton_instances": len(self._instances),
            "scoped_scopes": len(self._scoped_instances),
            "initialized": self._initialized,
            "services": [
                {
                    "type": desc.service_type.__name__,
                    "lifetime": desc.lifetime.value,
                    "has_instance": desc.service_type in self._instances,
                }
                for desc in self._services.values()
            ],
        }


# Global container instance
_container: Optional[DIContainer] = None
_container_lock = threading.RLock()


def get_container() -> DIContainer:
    """Get the global DI container"""
    global _container

    with _container_lock:
        if _container is None:
            _container = DIContainer()
        return _container


def reset_container():
    """Reset the global container (mainly for testing)"""
    global _container

    with _container_lock:
        _container = None


# Decorators for dependency injection
def inject(func: Callable) -> Callable:
    """Decorator for automatic dependency injection"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        container = get_container()
        sig = inspect.signature(func)

        # Resolve dependencies
        for param_name, param in sig.parameters.items():
            if param_name not in kwargs and param.annotation != inspect.Parameter.empty:
                try:
                    if container.is_registered(param.annotation):
                        kwargs[param_name] = container.get(param.annotation)
                except Exception as e:
                    logger.debug(
                        f"Could not inject {param.annotation.__name__} into {func.__name__}: {e}"
                    )

        return func(*args, **kwargs)

    return wrapper


def async_inject(func: Callable) -> Callable:
    """Decorator for automatic dependency injection in async functions"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        container = get_container()
        sig = inspect.signature(func)

        # Resolve dependencies
        for param_name, param in sig.parameters.items():
            if param_name not in kwargs and param.annotation != inspect.Parameter.empty:
                try:
                    if container.is_registered(param.annotation):
                        kwargs[param_name] = container.get(param.annotation)
                except Exception as e:
                    logger.debug(
                        f"Could not inject {param.annotation.__name__} into {func.__name__}: {e}"
                    )

        return await func(*args, **kwargs)

    return wrapper
