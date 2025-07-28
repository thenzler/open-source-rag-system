"""
Simple RAG API
Clean, professional API with AI answers only
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Simple request/response models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="User question")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="AI-generated answer")
    sources: list = Field(default=[], description="Source documents")
    confidence: float = Field(..., description="Confidence score")
    timestamp: str = Field(..., description="Response timestamp")
    query: str = Field(..., description="Original query")

class StatusResponse(BaseModel):
    service: str
    mode: str
    config: Dict[str, Any]
    healthy: bool

# Create router
router = APIRouter(prefix="/api/v1/rag", tags=["rag"])

def get_rag_service():
    """Get RAG service instance"""
    try:
        from core.repositories.factory import get_rag_repository
        from core.ollama_client import OllamaClient
        from core.services.simple_rag_service import SimpleRAGService
        
        # Get repositories
        rag_repo = get_rag_repository()
        vector_repo = rag_repo.vector_search
        audit_repo = rag_repo.audit
        
        # Get LLM client
        llm_client = OllamaClient()
        
        # Create service
        return SimpleRAGService(vector_repo, llm_client, audit_repo)
        
    except Exception as e:
        logger.error(f"Failed to create RAG service: {e}")
        raise HTTPException(status_code=500, detail="Service unavailable")

@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    rag_service = Depends(get_rag_service)
):
    """
    Ask a question and get an AI answer with sources
    
    - **query**: Your question (3-500 characters)
    - Returns AI answer with source citations
    """
    try:
        logger.info(f"Processing query: {request.query[:50]}...")
        
        # Process query
        response = await rag_service.answer_query(request.query)
        
        # Check for errors
        if "error" in response:
            raise HTTPException(status_code=400, detail=response["error"])
        
        return QueryResponse(**response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")

@router.get("/status", response_model=StatusResponse)
async def get_status(
    rag_service = Depends(get_rag_service)
):
    """Get RAG service status and configuration"""
    try:
        status = rag_service.get_status()
        return StatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")

@router.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "service": "Simple RAG API"}