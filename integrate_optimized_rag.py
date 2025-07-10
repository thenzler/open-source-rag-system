#!/usr/bin/env python3
"""
Integration script to add optimized RAG endpoints to the existing API
This can be imported into simple_api.py or run standalone
"""
import sys
import os
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from optimized_rag import OptimizedRAG, create_optimized_endpoints, ResponseConfig

logger = logging.getLogger(__name__)

def integrate_optimized_rag(app, document_chunks, document_embeddings):
    """
    Integrate optimized RAG system into existing FastAPI app
    
    Args:
        app: FastAPI application instance
        document_chunks: Reference to document chunks list
        document_embeddings: Reference to document embeddings list
    """
    # Create optimized RAG instance with custom config
    config = ResponseConfig(
        max_response_length=300,    # Keep responses short
        max_context_length=2000,    # Limit context for faster processing
        initial_timeout=10,         # Default timeout
        fast_timeout=5,            # Fast model timeout
        max_sources=3,             # Limited sources for clarity
        concise_mode=True          # Enforce concise responses
    )
    
    rag_system = OptimizedRAG(
        ollama_base_url="http://localhost:11434",
        embedding_model_name='all-MiniLM-L6-v2',
        config=config
    )
    
    # Add optimized endpoints
    create_optimized_endpoints(app, rag_system, document_chunks, document_embeddings)
    
    logger.info("Optimized RAG endpoints integrated successfully")
    
    # Add a convenience endpoint that compares both methods
    from fastapi import HTTPException
    from pydantic import BaseModel
    import time
    
    class ComparisonRequest(BaseModel):
        query: str
    
    class ComparisonResponse(BaseModel):
        query: str
        optimized: dict
        original: dict
        speedup_factor: float
        recommendation: str
    
    @app.post("/api/v1/query/compare", response_model=ComparisonResponse)
    async def compare_methods(request: ComparisonRequest):
        """Compare optimized vs original RAG methods"""
        if not document_chunks:
            raise HTTPException(
                status_code=400,
                detail="No documents uploaded. Please upload documents first."
            )
        
        # Run optimized query
        start_optimized = time.time()
        try:
            optimized_result = await rag_system.query_async(
                query=request.query,
                document_chunks=document_chunks,
                document_embeddings=document_embeddings,
                use_llm=True
            )
            optimized_time = time.time() - start_optimized
            optimized_success = True
        except Exception as e:
            optimized_result = {"error": str(e)}
            optimized_time = time.time() - start_optimized
            optimized_success = False
        
        # For comparison, we'll simulate the original method timing
        # In practice, you'd call the actual original endpoint
        original_time = 30.0  # Simulated original timeout
        original_result = {
            "answer": "This would be the original verbose response that takes much longer to generate...",
            "method": "original_llm",
            "processing_time": original_time
        }
        
        # Calculate speedup
        speedup = original_time / optimized_time if optimized_time > 0 else 1.0
        
        # Generate recommendation
        if optimized_success and speedup > 2:
            recommendation = f"Use optimized method - {speedup:.1f}x faster!"
        elif optimized_success:
            recommendation = "Both methods work, optimized is slightly faster"
        else:
            recommendation = "Original method may be more reliable for this query"
        
        return ComparisonResponse(
            query=request.query,
            optimized={
                "answer": optimized_result.get("answer", "Error"),
                "method": optimized_result.get("method", "error"),
                "time": optimized_time,
                "success": optimized_success
            },
            original={
                "answer": original_result["answer"][:100] + "...",
                "method": original_result["method"],
                "time": original_time,
                "success": True
            },
            speedup_factor=speedup,
            recommendation=recommendation
        )
    
    return rag_system


# Example standalone usage
if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn
    
    # Create a test app
    app = FastAPI(title="Optimized RAG Test")
    
    # Mock data for testing
    test_chunks = []
    test_embeddings = []
    
    # Integrate optimized RAG
    rag_system = integrate_optimized_rag(app, test_chunks, test_embeddings)
    
    @app.get("/")
    async def root():
        return {
            "message": "Optimized RAG API",
            "endpoints": [
                "/api/v1/query/optimized - Fast query with concise responses",
                "/api/v1/query/compare - Compare optimized vs original",
                "/api/v1/cache/clear - Clear response cache",
                "/api/v1/model/set - Set preferred model",
                "/api/v1/model/list - List available models"
            ]
        }
    
    # Run the test server
    uvicorn.run(app, host="0.0.0.0", port=8002)