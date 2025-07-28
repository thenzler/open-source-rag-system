#!/usr/bin/env python3
"""
Municipal API Integration
Extends the main RAG API with municipal-specific endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.municipal.municipal_rag import MunicipalRAG
from sentence_transformers import SentenceTransformer
from ollama_client import get_ollama_client

# Configure logging
logger = logging.getLogger(__name__)

# Request/Response models
class MunicipalQueryRequest(BaseModel):
    query: str
    municipality: str
    category: Optional[str] = None
    language: Optional[str] = "de"

class MunicipalQueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    municipality: str
    category: Optional[str] = None
    processing_time: float

class MunicipalStatsResponse(BaseModel):
    municipality: str
    total_chunks: int
    categories: Dict[str, int]
    average_importance: float
    high_importance_chunks: int
    languages: List[str]

# Global municipal RAG instances
municipal_rag_instances: Dict[str, MunicipalRAG] = {}

# Router for municipal endpoints
municipal_router = APIRouter(prefix="/api/municipal", tags=["municipal"])

def get_municipal_rag(municipality: str) -> MunicipalRAG:
    """Get or create municipal RAG instance"""
    if municipality not in municipal_rag_instances:
        # Load embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get Ollama client
        ollama_client = get_ollama_client()
        
        # Create municipal RAG instance
        municipal_rag = MunicipalRAG(municipality, embedding_model, ollama_client)
        
        # Try to load existing data
        try:
            municipal_rag.load_municipal_data("municipal_data")
            logger.info(f"Loaded existing data for {municipality}")
        except Exception as e:
            logger.warning(f"Could not load data for {municipality}: {e}")
        
        municipal_rag_instances[municipality] = municipal_rag
    
    return municipal_rag_instances[municipality]

@municipal_router.post("/query", response_model=MunicipalQueryResponse)
async def query_municipal_rag(request: MunicipalQueryRequest):
    """
    Query municipal RAG system with specialized municipal knowledge
    
    This endpoint provides answers specifically tailored to municipal services,
    administration, and local information with weighted importance scoring.
    """
    try:
        import time
        start_time = time.time()
        
        # Get municipal RAG instance
        municipal_rag = get_municipal_rag(request.municipality)
        
        # Generate answer
        result = municipal_rag.generate_municipal_answer(
            request.query, 
            request.category
        )
        
        processing_time = time.time() - start_time
        
        return MunicipalQueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            confidence=result['confidence'],
            municipality=result['municipality'],
            category=request.category,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing municipal query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query for {request.municipality}: {str(e)}"
        )

@municipal_router.get("/stats/{municipality}", response_model=MunicipalStatsResponse)
async def get_municipal_stats(municipality: str):
    """
    Get statistics about municipal knowledge base
    
    Returns information about available documents, categories, and system health
    for the specified municipality.
    """
    try:
        municipal_rag = get_municipal_rag(municipality)
        stats = municipal_rag.get_municipal_stats()
        
        return MunicipalStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting stats for {municipality}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats for {municipality}: {str(e)}"
        )

@municipal_router.get("/categories/{municipality}")
async def get_municipal_categories(municipality: str):
    """
    Get available categories for a municipality
    
    Returns a dictionary of categories and their document counts.
    """
    try:
        municipal_rag = get_municipal_rag(municipality)
        categories = municipal_rag.get_municipal_categories()
        
        return {
            "municipality": municipality,
            "categories": categories,
            "total_categories": len(categories)
        }
        
    except Exception as e:
        logger.error(f"Error getting categories for {municipality}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting categories for {municipality}: {str(e)}"
        )

@municipal_router.get("/search/{municipality}")
async def search_municipal_content(
    municipality: str,
    query: str = Query(..., description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    top_k: int = Query(5, description="Number of results to return")
):
    """
    Search municipal content without LLM processing
    
    Returns raw search results with similarity scores and metadata.
    """
    try:
        municipal_rag = get_municipal_rag(municipality)
        results = municipal_rag.municipal_search(query, top_k, category)
        
        return {
            "municipality": municipality,
            "query": query,
            "category_filter": category,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching {municipality}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching {municipality}: {str(e)}"
        )

@municipal_router.get("/available")
async def get_available_municipalities():
    """
    Get list of available municipalities
    
    Returns municipalities that have been loaded or are available for loading.
    """
    # Check for available municipal data
    municipal_data_dir = Path("municipal_data")
    available_municipalities = []
    
    if municipal_data_dir.exists():
        for subdir in municipal_data_dir.iterdir():
            if subdir.is_dir():
                # Check if it has the required data file
                if (subdir / "documents.json").exists():
                    available_municipalities.append({
                        "name": subdir.name,
                        "loaded": subdir.name in municipal_rag_instances,
                        "data_path": str(subdir)
                    })
    
    return {
        "available_municipalities": available_municipalities,
        "loaded_municipalities": list(municipal_rag_instances.keys()),
        "total_available": len(available_municipalities)
    }

@municipal_router.post("/reload/{municipality}")
async def reload_municipal_data(municipality: str):
    """
    Reload municipal data from disk
    
    Useful when municipal data has been updated and needs to be reloaded.
    """
    try:
        # Remove existing instance
        if municipality in municipal_rag_instances:
            del municipal_rag_instances[municipality]
        
        # Create new instance (will load fresh data)
        municipal_rag = get_municipal_rag(municipality)
        stats = municipal_rag.get_municipal_stats()
        
        return {
            "municipality": municipality,
            "status": "reloaded",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error reloading {municipality}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reloading {municipality}: {str(e)}"
        )

# Health check for municipal system
@municipal_router.get("/health")
async def municipal_health_check():
    """
    Health check for municipal RAG system
    
    Returns system status and loaded municipalities.
    """
    try:
        # Check if embedding model is available
        embedding_available = True
        try:
            SentenceTransformer('all-MiniLM-L6-v2')
        except:
            embedding_available = False
        
        # Check if Ollama is available
        ollama_available = False
        try:
            ollama_client = get_ollama_client()
            ollama_available = ollama_client.is_available()
        except:
            pass
        
        return {
            "status": "healthy",
            "embedding_model_available": embedding_available,
            "ollama_available": ollama_available,
            "loaded_municipalities": list(municipal_rag_instances.keys()),
            "total_loaded": len(municipal_rag_instances)
        }
        
    except Exception as e:
        logger.error(f"Municipal health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Municipal health check failed: {str(e)}"
        )

# Function to add municipal routes to main app
def add_municipal_routes(app):
    """Add municipal routes to the main FastAPI app"""
    app.include_router(municipal_router)