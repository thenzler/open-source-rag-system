"""
Confidence System Management API
Allows runtime configuration of the intelligent confidence system
"""
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class ProfileRequest(BaseModel):
    profile_name: str

class DomainRequest(BaseModel):    
    domain_name: str
    enabled: Optional[bool] = None
    terms: Optional[List[str]] = None
    adjustment: Optional[float] = None
    priority: Optional[int] = None

class ConfidenceStatusResponse(BaseModel):
    intelligent_system_enabled: bool
    active_profile: Optional[str]
    simple_threshold: float
    base_thresholds: Dict[str, float]
    query_analysis_enabled: bool
    enabled_domains: List[str]
    disabled_domains: List[str]
    external_knowledge_enabled: bool
    enabled_external_categories: List[str]
    disabled_external_categories: List[str]
    response_validation_enabled: bool

# Create router
router = APIRouter(prefix="/api/v1/confidence", tags=["confidence"])

def get_confidence_manager():
    """Dependency to get confidence manager from query service"""
    try:
        from core.repositories.factory import get_rag_repository
        from core.services.query_service import QueryProcessingService
        
        # Get the repository and services
        rag_repo = get_rag_repository()
        
        # Create a temporary query service to access confidence manager
        # In production, this would be properly injected
        doc_repo = rag_repo.documents
        vector_repo = rag_repo.vector_search
        audit_repo = rag_repo.audit
        
        query_service = QueryProcessingService(doc_repo, vector_repo, audit_repo)
        return query_service.confidence_manager
        
    except Exception as e:
        logger.error(f"Error getting confidence manager: {e}")
        raise HTTPException(status_code=500, detail="Failed to access confidence manager")

@router.get("/status", response_model=ConfidenceStatusResponse)
async def get_confidence_status(
    confidence_manager = Depends(get_confidence_manager)
):
    """Get current confidence system status and configuration"""
    try:
        status = confidence_manager.get_status_summary()
        
        return ConfidenceStatusResponse(
            intelligent_system_enabled=status.get("intelligent_system_enabled", True),
            active_profile=status.get("active_profile"),
            simple_threshold=status.get("simple_threshold", 0.4),
            base_thresholds=status.get("base_thresholds", {}),
            query_analysis_enabled=status.get("query_analysis_enabled", True),
            enabled_domains=status.get("enabled_domains", []),
            disabled_domains=status.get("disabled_domains", []),
            external_knowledge_enabled=status.get("external_knowledge_enabled", True),
            enabled_external_categories=status.get("enabled_external_categories", []),
            disabled_external_categories=status.get("disabled_external_categories", []),
            response_validation_enabled=status.get("response_validation_enabled", True)
        )
        
    except Exception as e:
        logger.error(f"Error getting confidence status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.post("/profile")
async def set_confidence_profile(
    request: ProfileRequest,
    confidence_manager = Depends(get_confidence_manager)
):
    """Set active configuration profile (strict, permissive, simple)"""
    try:
        success = confidence_manager.set_profile(request.profile_name)
        
        if not success:
            raise HTTPException(
                status_code=400, 
                detail=f"Profile '{request.profile_name}' not found. Available profiles: strict, permissive, simple"
            )
        
        return {
            "success": True,
            "message": f"Applied profile: {request.profile_name}",
            "active_profile": request.profile_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set profile: {str(e)}")

@router.post("/domain/enable")
async def enable_domain(
    request: DomainRequest,
    confidence_manager = Depends(get_confidence_manager)
):
    """Enable a specific domain for query analysis"""
    try:
        success = confidence_manager.enable_domain(request.domain_name)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Domain '{request.domain_name}' not found in configuration"
            )
        
        return {
            "success": True,
            "message": f"Enabled domain: {request.domain_name}",
            "domain": request.domain_name,
            "enabled": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling domain: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enable domain: {str(e)}")

@router.post("/domain/disable")
async def disable_domain(
    request: DomainRequest,
    confidence_manager = Depends(get_confidence_manager)
):
    """Disable a specific domain for query analysis"""
    try:
        success = confidence_manager.disable_domain(request.domain_name)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Domain '{request.domain_name}' not found in configuration"
            )
        
        return {
            "success": True,
            "message": f"Disabled domain: {request.domain_name}",
            "domain": request.domain_name,
            "enabled": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling domain: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to disable domain: {str(e)}")

@router.post("/domain/add")
async def add_custom_domain(
    request: DomainRequest,
    confidence_manager = Depends(get_confidence_manager)
):
    """Add a new custom domain to the configuration"""
    try:
        if not request.terms:
            raise HTTPException(
                status_code=400,
                detail="Terms are required when adding a new domain"
            )
        
        success = confidence_manager.add_custom_domain(
            domain_name=request.domain_name,
            terms=request.terms,
            adjustment=request.adjustment or -0.05,
            priority=request.priority or 3
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to add domain '{request.domain_name}'"
            )
        
        return {
            "success": True,
            "message": f"Added custom domain: {request.domain_name}",
            "domain": request.domain_name,
            "terms": request.terms,
            "adjustment": request.adjustment or -0.05,
            "priority": request.priority or 3,
            "enabled": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding custom domain: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add domain: {str(e)}")

@router.get("/test/{query}")
async def test_query_analysis(
    query: str,
    confidence_manager = Depends(get_confidence_manager)
):
    """Test how the confidence system would analyze a specific query"""
    try:
        # Determine confidence tiers for the query
        confidence_tiers = confidence_manager.determine_confidence_tiers(query, use_llm=True)
        
        # Check external knowledge detection
        is_external, external_reason = confidence_manager.detect_external_knowledge(query)
        
        return {
            "query": query,
            "intelligent_system_enabled": confidence_manager.is_enabled(),
            "confidence_tiers": confidence_tiers,
            "external_knowledge_detected": is_external,
            "external_knowledge_reason": external_reason,
            "would_block": is_external,
            "analysis": {
                "high_threshold": confidence_tiers.get("high_threshold", 0),
                "medium_threshold": confidence_tiers.get("medium_threshold", 0),
                "low_threshold": confidence_tiers.get("low_threshold", 0),
                "refusal_threshold": confidence_tiers.get("refusal_threshold", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error testing query analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze query: {str(e)}")

@router.get("/profiles")
async def list_available_profiles():
    """List all available configuration profiles"""
    return {
        "profiles": {
            "strict": {
                "description": "Maximum safety with high confidence thresholds",
                "use_case": "Critical applications requiring highest accuracy",
                "features": {
                    "external_knowledge_blocking": True,
                    "response_validation": "strict",
                    "thresholds": "high"
                }
            },
            "permissive": {
                "description": "More answers with lower confidence requirements",
                "use_case": "General use where broader coverage is preferred",
                "features": {
                    "external_knowledge_blocking": False,
                    "response_validation": "lenient",
                    "thresholds": "low"
                }
            },
            "simple": {
                "description": "Basic threshold-based system without intelligent features",
                "use_case": "Simple deployments or legacy compatibility",
                "features": {
                    "intelligent_analysis": False,
                    "external_knowledge_blocking": False,
                    "single_threshold": True
                }
            }
        },
        "current_profile": None,
        "can_switch_runtime": True
    }

@router.get("/domains")
async def list_available_domains(
    confidence_manager = Depends(get_confidence_manager)
):
    """List all available domains and their configuration"""
    try:
        status = confidence_manager.get_status_summary()
        
        # Get domain details from config
        config = confidence_manager.config
        domain_terms = config.get("query_analysis", {}).get("domain_terms", {})
        
        domains = {}
        for domain_name, settings in domain_terms.items():
            domains[domain_name] = {
                "enabled": settings.get("enabled", True),
                "terms": settings.get("terms", []),
                "adjustment": settings.get("adjustment", 0.0),
                "priority": settings.get("priority", 1),
                "description": f"Domain with {len(settings.get('terms', []))} terms"
            }
        
        return {
            "domains": domains,
            "enabled_count": len(status.get("enabled_domains", [])),
            "disabled_count": len(status.get("disabled_domains", [])),
            "total_count": len(domains)
        }
        
    except Exception as e:
        logger.error(f"Error listing domains: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list domains: {str(e)}")