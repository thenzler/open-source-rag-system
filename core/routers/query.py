"""
Simple Query Router
Single AI-only endpoint for RAG queries
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Simple request/response models for single endpoint
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

router = APIRouter(prefix="/api/v1", tags=["query"])

# Dependency to get SimpleRAGService
def get_rag_service():
    """Get SimpleRAGService instance"""
    try:
        from ..repositories.factory import RepositoryFactory
        from ..services.simple_rag_service import SimpleRAGService
        from ..ollama_client import OllamaClient
        
        # Get repositories
        rag_repo = RepositoryFactory.create_production_repository()
        vector_repo = rag_repo.vector_search
        audit_repo = rag_repo.audit
        
        # Get LLM client with faster timeout for better user experience
        llm_client = OllamaClient(timeout=60)  # 1 minute instead of 3 minutes
        
        # Create simple RAG service
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
    
    Single endpoint for all RAG queries - AI answers only
    - **query**: Your question (3-500 characters)
    - Returns AI answer with source citations
    """
    try:
        logger.info(f"Processing query: {request.query[:50]}...")
        
        # Process query using SimpleRAGService
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
    return {"status": "healthy", "service": "Simple RAG Query API"}

