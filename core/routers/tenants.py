"""
Tenant Management Router
Handles tenant CRUD operations for multi-tenancy
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, validator
import logging
import re
from datetime import datetime

from ..repositories.tenant_repository import TenantRepository
from ..models import Tenant
from ..middleware import TenantContext, require_tenant_access

router = APIRouter(prefix="/api/v1/tenants", tags=["tenants"])
logger = logging.getLogger(__name__)

class TenantCreate(BaseModel):
    """Tenant creation model"""
    name: str
    slug: str
    domain: Optional[str] = None
    settings: Dict[str, Any] = {}
    limits: Dict[str, Any] = {}
    
    @validator('slug')
    def validate_slug(cls, v):
        """Validate tenant slug"""
        if not re.match(r'^[a-z0-9-]+$', v):
            raise ValueError('Slug must contain only lowercase letters, numbers, and hyphens')
        if len(v) < 2 or len(v) > 50:
            raise ValueError('Slug must be between 2 and 50 characters')
        if v in ['api', 'www', 'admin', 'app', 'default']:
            raise ValueError('Slug is reserved')
        return v
    
    @validator('name')
    def validate_name(cls, v):
        """Validate tenant name"""
        if len(v.strip()) < 2:
            raise ValueError('Name must be at least 2 characters')
        if len(v) > 100:
            raise ValueError('Name cannot exceed 100 characters')
        return v.strip()

class TenantUpdate(BaseModel):
    """Tenant update model"""
    name: Optional[str] = None
    domain: Optional[str] = None
    is_active: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None
    limits: Optional[Dict[str, Any]] = None

# Dependency to get tenant repository
def get_tenant_repository(request: Request) -> TenantRepository:
    """Get tenant repository from request state"""
    # In a real implementation, this would come from DI
    db_path = 'data/rag_database.db'  # Should come from config
    return TenantRepository(db_path)

@router.post("", response_model=Dict[str, Any])
async def create_tenant(
    tenant_data: TenantCreate,
    tenant_repo: TenantRepository = Depends(get_tenant_repository)
):
    """Create a new tenant - Super admin only"""
    try:
        # Create tenant object
        tenant = Tenant(
            name=tenant_data.name,
            slug=tenant_data.slug,
            domain=tenant_data.domain,
            is_active=True,
            settings=tenant_data.settings,
            limits=tenant_data.limits or {
                "max_documents": 1000,
                "max_storage_mb": 1024,
                "max_users": 10
            }
        )
        
        # Create tenant
        created_tenant = await tenant_repo.create_tenant(tenant)
        
        logger.info(f"Created tenant: {created_tenant.name} (ID: {created_tenant.id})")
        
        return {
            "message": "Tenant created successfully",
            "tenant": created_tenant.to_dict()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create tenant: {e}")
        raise HTTPException(status_code=500, detail="Failed to create tenant")

@router.get("", response_model=Dict[str, Any])
async def list_tenants(
    tenant_repo: TenantRepository = Depends(get_tenant_repository)
):
    """List all tenants - Super admin only"""
    try:
        tenants = await tenant_repo.list_all()
        
        return {
            "tenants": [tenant.to_dict() for tenant in tenants],
            "total": len(tenants)
        }
        
    except Exception as e:
        logger.error(f"Failed to list tenants: {e}")
        raise HTTPException(status_code=500, detail="Failed to list tenants")

@router.get("/current", response_model=Dict[str, Any])
async def get_current_tenant():
    """Get current tenant information"""
    try:
        tenant = TenantContext.get_current_tenant()
        if not tenant:
            raise HTTPException(status_code=404, detail="No tenant context")
        
        return {
            "tenant": tenant.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to get current tenant: {e}")
        raise HTTPException(status_code=500, detail="Failed to get current tenant")

@router.get("/current/stats", response_model=Dict[str, Any])
async def get_current_tenant_stats(
    tenant_repo: TenantRepository = Depends(get_tenant_repository)
):
    """Get current tenant usage statistics"""
    try:
        tenant = TenantContext.get_current_tenant()
        if not tenant:
            raise HTTPException(status_code=404, detail="No tenant context")
        
        stats = await tenant_repo.get_tenant_stats(tenant.id)
        
        return {
            "tenant_id": tenant.id,
            "tenant_name": tenant.name,
            "stats": stats,
            "limits": tenant.limits
        }
        
    except Exception as e:
        logger.error(f"Failed to get tenant stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get tenant stats")

@router.get("/{tenant_id}", response_model=Dict[str, Any])
async def get_tenant(
    tenant_id: int,
    tenant_repo: TenantRepository = Depends(get_tenant_repository)
):
    """Get specific tenant - Super admin only"""
    try:
        tenant = await tenant_repo.get_by_id(tenant_id)
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")
        
        return {
            "tenant": tenant.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get tenant")

@router.put("/{tenant_id}", response_model=Dict[str, Any])
async def update_tenant(
    tenant_id: int,
    updates: TenantUpdate,
    tenant_repo: TenantRepository = Depends(get_tenant_repository)
):
    """Update tenant - Super admin only"""
    try:
        # Prepare update data
        update_data = updates.dict(exclude_unset=True)
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No update data provided")
        
        # Update tenant
        updated_tenant = await tenant_repo.update_tenant(tenant_id, update_data)
        if not updated_tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")
        
        logger.info(f"Updated tenant {tenant_id}: {list(update_data.keys())}")
        
        return {
            "message": "Tenant updated successfully",
            "tenant": updated_tenant.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update tenant")

@router.delete("/{tenant_id}", response_model=Dict[str, Any])
async def delete_tenant(
    tenant_id: int,
    tenant_repo: TenantRepository = Depends(get_tenant_repository)
):
    """Delete (deactivate) tenant - Super admin only"""
    try:
        success = await tenant_repo.delete_tenant(tenant_id)
        if not success:
            raise HTTPException(status_code=404, detail="Tenant not found or cannot be deleted")
        
        logger.info(f"Deleted tenant {tenant_id}")
        
        return {
            "message": f"Tenant {tenant_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete tenant")

@router.get("/{tenant_id}/stats", response_model=Dict[str, Any])
async def get_tenant_stats(
    tenant_id: int,
    tenant_repo: TenantRepository = Depends(get_tenant_repository)
):
    """Get tenant usage statistics - Super admin only"""
    try:
        # Verify tenant exists
        tenant = await tenant_repo.get_by_id(tenant_id)
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")
        
        stats = await tenant_repo.get_tenant_stats(tenant_id)
        
        return {
            "tenant_id": tenant_id,
            "tenant_name": tenant.name,
            "stats": stats,
            "limits": tenant.limits
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tenant stats for {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get tenant stats")

@router.post("/check-slug", response_model=Dict[str, Any])
async def check_slug_availability(
    slug_data: Dict[str, str],
    tenant_repo: TenantRepository = Depends(get_tenant_repository)
):
    """Check if tenant slug is available"""
    try:
        slug = slug_data.get("slug", "").strip()
        if not slug:
            raise HTTPException(status_code=400, detail="Slug is required")
        
        # Validate slug format
        if not re.match(r'^[a-z0-9-]+$', slug):
            return {"available": False, "reason": "Invalid format"}
        
        if slug in ['api', 'www', 'admin', 'app', 'default']:
            return {"available": False, "reason": "Reserved slug"}
        
        # Check database
        existing_tenant = await tenant_repo.get_by_slug(slug)
        
        return {
            "slug": slug,
            "available": existing_tenant is None,
            "reason": "Slug already exists" if existing_tenant else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check slug availability: {e}")
        raise HTTPException(status_code=500, detail="Failed to check slug availability")