"""
FastAPI Application - API Gateway for RAG System
Main entry point providing RESTful API endpoints for document management and querying.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from prometheus_fastapi_instrumentator import Instrumentator
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.core.database import Base, get_database
from app.core.security import verify_token
from app.models.documents import Document, DocumentChunk
from app.models.queries import QueryLog
from app.schemas.documents import DocumentResponse, DocumentUploadRequest, DocumentListResponse
from app.schemas.queries import QueryRequest, QueryResponse, AdvancedQueryRequest
from app.schemas.common import HealthResponse, StatsResponse
from app.services.document_service import DocumentService
from app.services.query_service import QueryService
from app.services.analytics_service import AnalyticsService
from app.core.exceptions import DocumentNotFoundError, ProcessingError, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global settings
settings = get_settings()
security = HTTPBearer()

# Services
document_service = DocumentService()
query_service = QueryService()
analytics_service = AnalyticsService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAG System API Gateway")
    
    # Initialize Redis for rate limiting
    redis_client = redis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis_client)
    
    # Initialize database
    engine = create_async_engine(settings.database_url, echo=settings.debug)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Initialize services
    await document_service.initialize()
    await query_service.initialize()
    
    logger.info("RAG System API Gateway started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG System API Gateway")
    await redis_client.close()
    await engine.dispose()


# FastAPI Application
app = FastAPI(
    title="RAG System API",
    description="Open Source Retrieval-Augmented Generation System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add Prometheus metrics
if settings.enable_metrics:
    Instrumentator().instrument(app).expose(app)


# Dependencies
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token and return user info."""
    if not settings.enable_authentication:
        return {"user_id": "anonymous", "username": "anonymous"}
    
    try:
        payload = verify_token(credentials.credentials)
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token")


async def get_db_session():
    """Get database session."""
    async for session in get_database():
        yield session


# Health Check Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check."""
    try:
        # Check database
        db_healthy = await document_service.health_check()
        
        # Check vector database
        vector_healthy = await query_service.health_check()
        
        # Check LLM service
        llm_healthy = await query_service.check_llm_health()
        
        status = "healthy" if all([db_healthy, vector_healthy, llm_healthy]) else "unhealthy"
        
        return HealthResponse(
            status=status,
            services={
                "database": "healthy" if db_healthy else "unhealthy",
                "vector_database": "healthy" if vector_healthy else "unhealthy",
                "llm_service": "healthy" if llm_healthy else "unhealthy"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="unhealthy", services={})


@app.get("/api/v1/health", response_model=HealthResponse)
async def api_health_check():
    """API-specific health check."""
    return await health_check()


# Document Management Endpoints
@app.post("/api/v1/documents", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    rate_limit: dict = Depends(RateLimiter(times=100, hours=1))
):
    """Upload a new document for processing."""
    try:
        # Validate file
        if file.size > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        
        if not any(file.content_type.startswith(mime) for mime in settings.allowed_mime_types):
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Process upload
        document = await document_service.upload_document(
            file=file,
            metadata=metadata,
            user_id=user["user_id"],
            db=db
        )
        
        # Queue background processing
        background_tasks.add_task(
            document_service.process_document_async,
            document.id,
            db
        )
        
        return DocumentResponse.from_orm(document)
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")


@app.get("/api/v1/documents", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    rate_limit: dict = Depends(RateLimiter(times=60, minutes=1))
):
    """Get paginated list of documents."""
    try:
        documents, total = await document_service.list_documents(
            skip=skip,
            limit=min(limit, 100),  # Max 100 per request
            status=status,
            category=category,
            search=search,
            user_id=user["user_id"],
            db=db
        )
        
        return DocumentListResponse(
            documents=[DocumentResponse.from_orm(doc) for doc in documents],
            total=total,
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


@app.get("/api/v1/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    rate_limit: dict = Depends(RateLimiter(times=60, minutes=1))
):
    """Get detailed information about a specific document."""
    try:
        document = await document_service.get_document(
            document_id=document_id,
            user_id=user["user_id"],
            db=db
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentResponse.from_orm(document)
        
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document")


@app.get("/api/v1/documents/{document_id}/status")
async def get_document_status(
    document_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    rate_limit: dict = Depends(RateLimiter(times=60, minutes=1))
):
    """Get processing status of a document."""
    try:
        status = await document_service.get_processing_status(
            document_id=document_id,
            user_id=user["user_id"],
            db=db
        )
        
        return status
        
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to check status")


@app.delete("/api/v1/documents/{document_id}")
async def delete_document(
    document_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    rate_limit: dict = Depends(RateLimiter(times=60, minutes=1))
):
    """Delete a document and all associated data."""
    try:
        result = await document_service.delete_document(
            document_id=document_id,
            user_id=user["user_id"],
            db=db
        )
        
        return {
            "message": "Document deleted successfully",
            "document_id": document_id,
            "deleted_chunks": result["deleted_chunks"],
            "deleted_vectors": result["deleted_vectors"]
        }
        
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


# Query Endpoints
@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(
    query_request: QueryRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    rate_limit: dict = Depends(RateLimiter(times=60, minutes=1))
):
    """Perform semantic search across documents."""
    try:
        # Validate query
        if len(query_request.query) > settings.max_query_length:
            raise HTTPException(status_code=400, detail="Query too long")
        
        # Execute query
        result = await query_service.query_documents(
            query=query_request.query,
            top_k=min(query_request.top_k, settings.max_search_results),
            min_score=query_request.min_score,
            filters=query_request.filters,
            user_id=user["user_id"],
            db=db
        )
        
        # Log query
        await analytics_service.log_query(
            query=query_request.query,
            user_id=user["user_id"],
            response=result,
            db=db
        )
        
        return QueryResponse(**result)
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")


@app.post("/api/v1/query/advanced", response_model=QueryResponse)
async def advanced_query(
    query_request: AdvancedQueryRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    rate_limit: dict = Depends(RateLimiter(times=30, minutes=1))
):
    """Perform advanced query with re-ranking and filtering."""
    try:
        result = await query_service.advanced_query(
            query_request=query_request,
            user_id=user["user_id"],
            db=db
        )
        
        # Log query
        await analytics_service.log_query(
            query=query_request.query,
            user_id=user["user_id"],
            response=result,
            db=db
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Advanced query failed: {e}")
        raise HTTPException(status_code=500, detail="Advanced query failed")


@app.get("/api/v1/documents/{document_id}/similar")
async def find_similar_documents(
    document_id: str,
    top_k: int = 5,
    min_score: float = 0.7,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    rate_limit: dict = Depends(RateLimiter(times=60, minutes=1))
):
    """Find documents similar to a given document."""
    try:
        result = await query_service.find_similar_documents(
            document_id=document_id,
            top_k=min(top_k, 20),
            min_score=min_score,
            user_id=user["user_id"],
            db=db
        )
        
        return result
        
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Similar document search failed: {e}")
        raise HTTPException(status_code=500, detail="Similar document search failed")


# Analytics Endpoints
@app.get("/api/v1/analytics/stats", response_model=StatsResponse)
async def get_system_stats(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    rate_limit: dict = Depends(RateLimiter(times=10, minutes=1))
):
    """Get system statistics."""
    try:
        stats = await analytics_service.get_system_stats(db=db)
        return StatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@app.get("/api/v1/analytics/queries")
async def get_query_analytics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: str = "day",
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    rate_limit: dict = Depends(RateLimiter(times=10, minutes=1))
):
    """Get query analytics for a time period."""
    try:
        analytics = await analytics_service.get_query_analytics(
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            db=db
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Query analytics failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")


# Configuration Endpoints
@app.get("/api/v1/config")
async def get_configuration(
    user: dict = Depends(get_current_user),
    rate_limit: dict = Depends(RateLimiter(times=10, minutes=1))
):
    """Get system configuration."""
    return {
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "similarity_threshold": 0.7,
        "max_query_length": settings.max_query_length,
        "rate_limits": {
            "queries_per_minute": 60,
            "uploads_per_hour": 100
        },
        "features": {
            "query_expansion": settings.enable_query_expansion,
            "reranking": settings.enable_reranking,
            "caching": settings.enable_caching
        }
    }


# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "timestamp": "2025-07-04T09:45:00Z"
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "timestamp": "2025-07-04T09:45:00Z"
            }
        }
    )


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers if not settings.debug else 1,
        reload=settings.reload and settings.debug,
        access_log=settings.access_log,
        log_level=settings.log_level.lower()
    )
