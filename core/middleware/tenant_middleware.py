"""
Multi-Tenant Middleware
Handles tenant resolution and context injection
"""

import logging
from typing import Optional

from fastapi import HTTPException, Request

from ..models import Tenant
from ..repositories.tenant_repository import TenantRepository

logger = logging.getLogger(__name__)


class TenantContext:
    """Thread-local tenant context"""

    _current_tenant: Optional[Tenant] = None

    @classmethod
    def set_current_tenant(cls, tenant: Tenant):
        """Set current tenant for request"""
        cls._current_tenant = tenant

    @classmethod
    def get_current_tenant(cls) -> Optional[Tenant]:
        """Get current tenant"""
        return cls._current_tenant

    @classmethod
    def get_current_tenant_id(cls) -> int:
        """Get current tenant ID (defaults to 1 for backward compatibility)"""
        if cls._current_tenant:
            return cls._current_tenant.id
        return 1  # Default tenant

    @classmethod
    def clear(cls):
        """Clear tenant context"""
        cls._current_tenant = None


class TenantResolver:
    """Resolves tenant from request"""

    def __init__(self, tenant_repo: TenantRepository):
        self.tenant_repo = tenant_repo
        self._cache = {}  # Simple in-memory cache

    async def resolve_tenant(self, request: Request) -> Optional[Tenant]:
        """Resolve tenant from request"""
        try:
            # Strategy 1: Tenant from subdomain
            tenant = await self._resolve_from_subdomain(request)
            if tenant:
                return tenant

            # Strategy 2: Tenant from custom domain
            tenant = await self._resolve_from_domain(request)
            if tenant:
                return tenant

            # Strategy 3: Tenant from header
            tenant = await self._resolve_from_header(request)
            if tenant:
                return tenant

            # Strategy 4: Tenant from URL path
            tenant = await self._resolve_from_path(request)
            if tenant:
                return tenant

            # Fallback: Default tenant
            return await self._get_default_tenant()

        except Exception as e:
            logger.error(f"Tenant resolution failed: {e}")
            return await self._get_default_tenant()

    async def _resolve_from_subdomain(self, request: Request) -> Optional[Tenant]:
        """Resolve tenant from subdomain (e.g., acme.rag-system.com)"""
        try:
            host = request.headers.get("host", "")
            if not host:
                return None

            # Extract subdomain
            parts = host.split(".")
            if len(parts) >= 3:  # subdomain.domain.tld
                subdomain = parts[0]

                # Skip common subdomains
                if subdomain in ["www", "api", "admin"]:
                    return None

                # Check cache first
                cache_key = f"subdomain:{subdomain}"
                if cache_key in self._cache:
                    return self._cache[cache_key]

                # Resolve from database
                tenant = await self.tenant_repo.get_by_slug(subdomain)
                if tenant:
                    self._cache[cache_key] = tenant
                    return tenant

            return None

        except Exception as e:
            logger.warning(f"Subdomain resolution failed: {e}")
            return None

    async def _resolve_from_domain(self, request: Request) -> Optional[Tenant]:
        """Resolve tenant from custom domain"""
        try:
            host = request.headers.get("host", "")
            if not host:
                return None

            # Remove port if present
            domain = host.split(":")[0]

            # Check cache first
            cache_key = f"domain:{domain}"
            if cache_key in self._cache:
                return self._cache[cache_key]

            # Resolve from database
            tenant = await self.tenant_repo.get_by_domain(domain)
            if tenant:
                self._cache[cache_key] = tenant
                return tenant

            return None

        except Exception as e:
            logger.warning(f"Domain resolution failed: {e}")
            return None

    async def _resolve_from_header(self, request: Request) -> Optional[Tenant]:
        """Resolve tenant from X-Tenant header"""
        try:
            tenant_header = request.headers.get("x-tenant")
            if not tenant_header:
                return None

            # Check if it's a slug or ID
            if tenant_header.isdigit():
                tenant_id = int(tenant_header)
                return await self.tenant_repo.get_by_id(tenant_id)
            else:
                return await self.tenant_repo.get_by_slug(tenant_header)

        except Exception as e:
            logger.warning(f"Header resolution failed: {e}")
            return None

    async def _resolve_from_path(self, request: Request) -> Optional[Tenant]:
        """Resolve tenant from URL path (e.g., /tenant/acme/api/...)"""
        try:
            path = request.url.path
            if path.startswith("/tenant/"):
                path_parts = path.split("/")
                if len(path_parts) >= 3:
                    tenant_slug = path_parts[2]
                    return await self.tenant_repo.get_by_slug(tenant_slug)

            return None

        except Exception as e:
            logger.warning(f"Path resolution failed: {e}")
            return None

    async def _get_default_tenant(self) -> Tenant:
        """Get default tenant"""
        cache_key = "default_tenant"
        if cache_key in self._cache:
            return self._cache[cache_key]

        tenant = await self.tenant_repo.get_by_id(1)
        if tenant:
            self._cache[cache_key] = tenant
            return tenant

        # Create default tenant if it doesn't exist
        from ..models import Tenant

        default_tenant = Tenant(
            id=1,
            name="Default Organization",
            slug="default",
            is_active=True,
            settings={},
            limits={"max_documents": 1000, "max_storage_mb": 1024},
        )
        return await self.tenant_repo.create_tenant(default_tenant)


# Global tenant resolver instance
_tenant_resolver: Optional[TenantResolver] = None


def get_tenant_resolver() -> TenantResolver:
    """Get global tenant resolver"""
    global _tenant_resolver
    if _tenant_resolver is None:
        raise RuntimeError("Tenant resolver not initialized")
    return _tenant_resolver


def initialize_tenant_resolver(tenant_repo: TenantRepository):
    """Initialize global tenant resolver"""
    global _tenant_resolver
    _tenant_resolver = TenantResolver(tenant_repo)


async def tenant_middleware(request: Request, call_next):
    """FastAPI middleware for tenant resolution"""
    try:
        # Clear any existing tenant context
        TenantContext.clear()

        # Resolve tenant for this request
        resolver = get_tenant_resolver()
        tenant = await resolver.resolve_tenant(request)

        if not tenant:
            raise HTTPException(status_code=400, detail="Could not resolve tenant")

        if not tenant.is_active:
            raise HTTPException(status_code=403, detail="Tenant is not active")

        # Set tenant context
        TenantContext.set_current_tenant(tenant)

        # Add tenant info to request state
        request.state.tenant = tenant
        request.state.tenant_id = tenant.id

        # Process request
        response = await call_next(request)

        # Add tenant info to response headers (optional)
        response.headers["X-Tenant-ID"] = str(tenant.id)
        response.headers["X-Tenant-Slug"] = tenant.slug

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tenant middleware error: {e}")
        # Fallback to default tenant
        TenantContext.clear()
        response = await call_next(request)
        return response
    finally:
        # Always clear context after request
        TenantContext.clear()


def require_tenant_access(allowed_roles: list = None):
    """Decorator to require tenant access"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            tenant = TenantContext.get_current_tenant()
            if not tenant:
                raise HTTPException(status_code=401, detail="No tenant context")

            # Add role checking here if needed
            # if allowed_roles and user_role not in allowed_roles:
            #     raise HTTPException(status_code=403, detail="Insufficient permissions")

            return await func(*args, **kwargs)

        return wrapper

    return decorator
