"""
Enhanced API endpoints for advanced query features.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import json
from pydantic import BaseModel, Field

from app.services.advanced_query_service import (
    AdvancedQueryService,
    QueryContext,
    SemanticFilter,
    EnhancedQueryResult
)
from app.core.database import get_database
from app.core.config import get_settings
from app.schemas.queries import QueryRequest, QueryResponse

# Initialize services
settings = get_settings()
advanced_query_service = AdvancedQueryService()

# Create router
router = APIRouter(prefix="/api/v1/advanced", tags=["Advanced Query"])


class AdvancedQueryRequest(BaseModel):
    """Request model for advanced queries."""
    query: str = Field(..., description="The search query")
    context: Optional[Dict[str, Any]] = Field(None, description="Query context")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    
    # Query expansion options
    enable_expansion: bool = Field(True, description="Enable query expansion")
    max_expansions: int = Field(5, description="Maximum number of expanded queries")
    
    # Semantic filtering options
    similarity_threshold: float = Field(0.5, description="Minimum similarity threshold")
    categories: Optional[List[str]] = Field(None, description="Document categories to include")
    content_types: Optional[List[str]] = Field(None, description="Content types to include")
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range filter")
    exclude_keywords: Optional[List[str]] = Field(None, description="Keywords to exclude")
    
    # Result options
    top_k: int = Field(10, description="Number of results to return")
    enable_clustering: bool = Field(True, description="Enable result clustering")
    include_suggestions: bool = Field(True, description="Include query suggestions")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "artificial intelligence applications",
                "context": {
                    "user_preferences": {"domain": "technology"},
                    "query_history": ["machine learning", "neural networks"]
                },
                "filters": {
                    "categories": ["research", "technology"],
                    "content_types": ["pdf", "docx"]
                },
                "enable_expansion": True,
                "similarity_threshold": 0.7,
                "top_k": 15,
                "enable_clustering": True
            }
        }


class StreamingQueryRequest(BaseModel):
    """Request model for streaming queries."""
    query: str = Field(..., description="The search query")
    stream_results: bool = Field(True, description="Enable result streaming")
    batch_size: int = Field(5, description="Number of results per batch")
    delay_ms: int = Field(100, description="Delay between batches in milliseconds")


class QueryAnalyticsRequest(BaseModel):
    """Request model for query analytics."""
    time_range: str = Field("7d", description="Time range (1d, 7d, 30d)")
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    include_failed: bool = Field(False, description="Include failed queries")


@router.post("/query", response_model=Dict[str, Any])
async def advanced_query(
    request: AdvancedQueryRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Perform advanced semantic search with expansion and filtering.
    
    Features:
    - Query expansion with synonyms and related concepts
    - Semantic filtering and clustering
    - Context-aware results
    - Query suggestions
    """
    try:
        # Build query context
        context = QueryContext(
            user_id=user["user_id"],
            query_history=request.context.get("query_history", []) if request.context else [],
            filters=request.filters or {},
            preferences=request.context.get("preferences", {}) if request.context else {}
        )
        
        # Build semantic filter
        semantic_filter = None
        if any([request.categories, request.content_types, request.date_range, request.exclude_keywords]):
            date_range = None
            if request.date_range:
                try:
                    date_range = (
                        datetime.fromisoformat(request.date_range["start"]),
                        datetime.fromisoformat(request.date_range["end"])
                    )
                except (KeyError, ValueError):
                    pass
            
            semantic_filter = SemanticFilter(
                categories=request.categories or [],
                content_types=request.content_types or [],
                date_range=date_range,
                similarity_threshold=request.similarity_threshold,
                exclude_keywords=request.exclude_keywords or []
            )
        
        # Execute advanced query
        result = await advanced_query_service.process_advanced_query(
            query=request.query,
            context=context,
            semantic_filter=semantic_filter,
            top_k=request.top_k,
            enable_expansion=request.enable_expansion,
            enable_clustering=request.enable_clustering
        )
        
        # Log query in background
        background_tasks.add_task(
            _log_advanced_query,
            request.query,
            user["user_id"],
            result,
            db
        )
        
        # Format response
        response = {
            "query": result.query,
            "results": result.results,
            "total_results": result.total_results,
            "processing_time": result.processing_time,
            "confidence_score": result.confidence_score,
            "query_expansion": {
                "expanded_queries": result.query_expansion.expanded_queries if result.query_expansion else [],
                "synonyms": result.query_expansion.synonyms if result.query_expansion else [],
                "related_concepts": result.query_expansion.related_concepts if result.query_expansion else []
            },
            "semantic_clusters": result.semantic_clusters,
            "suggestions": result.suggestions,
            "metadata": {
                "user_id": user["user_id"],
                "timestamp": datetime.now().isoformat(),
                "features_used": {
                    "expansion": request.enable_expansion,
                    "clustering": request.enable_clustering,
                    "filtering": semantic_filter is not None
                }
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Advanced query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Advanced query processing failed: {str(e)}"
        )


@router.post("/query/streaming")
async def streaming_query(
    request: StreamingQueryRequest,
    user: dict = Depends(get_current_user)
):
    """
    Stream query results in real-time as they become available.
    
    Features:
    - Real-time result streaming
    - Configurable batch sizes
    - Progress updates
    """
    try:
        async def generate_stream():
            # Execute query
            basic_request = AdvancedQueryRequest(
                query=request.query,
                top_k=50,  # Get more results for streaming
                enable_expansion=True,
                enable_clustering=False  # Disable clustering for streaming
            )
            
            context = QueryContext(
                user_id=user["user_id"],
                query_history=[],
                filters={},
                preferences={}
            )
            
            result = await advanced_query_service.process_advanced_query(
                query=request.query,
                context=context,
                top_k=50,
                enable_expansion=True,
                enable_clustering=False
            )
            
            # Stream results in batches
            results = result.results
            total_batches = (len(results) + request.batch_size - 1) // request.batch_size
            
            for i in range(0, len(results), request.batch_size):
                batch = results[i:i + request.batch_size]
                batch_num = i // request.batch_size + 1
                
                stream_data = {
                    "type": "results",
                    "batch": batch_num,
                    "total_batches": total_batches,
                    "results": batch,
                    "progress": min(100, (i + len(batch)) / len(results) * 100)
                }
                
                yield f"data: {json.dumps(stream_data)}\n\n"
                
                # Add delay between batches
                if request.delay_ms > 0:
                    await asyncio.sleep(request.delay_ms / 1000)
            
            # Send completion message
            completion_data = {
                "type": "complete",
                "total_results": len(results),
                "processing_time": result.processing_time,
                "confidence_score": result.confidence_score
            }
            
            yield f"data: {json.dumps(completion_data)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Streaming query failed: {str(e)}"
        )


@router.get("/query/suggestions")
async def get_query_suggestions(
    query: str = Query(..., description="Partial query to get suggestions for"),
    limit: int = Query(10, description="Maximum number of suggestions"),
    user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get query suggestions based on partial input and user history.
    
    Features:
    - Auto-complete suggestions
    - Personalized recommendations
    - Popular queries
    """
    try:
        # Get user's query history
        query_history = await _get_user_query_history(user["user_id"], db, limit=20)
        
        # Generate suggestions
        suggestions = []
        
        # 1. History-based suggestions
        for past_query in query_history:
            if query.lower() in past_query.lower() and past_query != query:
                suggestions.append({
                    "text": past_query,
                    "type": "history",
                    "confidence": 0.8
                })
        
        # 2. Expansion-based suggestions
        if len(query) > 2:
            try:
                expansion_result = await advanced_query_service._expand_query(
                    query,
                    QueryContext(
                        user_id=user["user_id"],
                        query_history=query_history,
                        filters={},
                        preferences={}
                    )
                )
                
                for expanded in expansion_result.expanded_queries:
                    suggestions.append({
                        "text": expanded,
                        "type": "expansion",
                        "confidence": 0.6
                    })
                
                for concept in expansion_result.related_concepts:
                    suggestions.append({
                        "text": f"{query} {concept}",
                        "type": "concept",
                        "confidence": 0.5
                    })
                    
            except Exception as e:
                logger.error(f"Expansion suggestions failed: {e}")
        
        # 3. Popular queries (mock implementation)
        popular_queries = [
            "artificial intelligence applications",
            "machine learning algorithms",
            "data science techniques",
            "neural network architectures",
            "natural language processing"
        ]
        
        for popular in popular_queries:
            if query.lower() in popular.lower() and popular not in [s["text"] for s in suggestions]:
                suggestions.append({
                    "text": popular,
                    "type": "popular",
                    "confidence": 0.4
                })
        
        # Sort by confidence and limit
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        suggestions = suggestions[:limit]
        
        return {
            "query": query,
            "suggestions": suggestions,
            "count": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Query suggestions failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate suggestions: {str(e)}"
        )


@router.get("/analytics/queries")
async def get_query_analytics(
    request: QueryAnalyticsRequest = Depends(),
    user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get analytics data for queries.
    
    Features:
    - Query performance metrics
    - Popular queries
    - User behavior analysis
    - Trend analysis
    """
    try:
        # Parse time range
        time_ranges = {
            "1d": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        
        time_delta = time_ranges.get(request.time_range, timedelta(days=7))
        start_date = datetime.now() - time_delta
        
        # Get query analytics (mock implementation)
        analytics = {
            "time_range": request.time_range,
            "start_date": start_date.isoformat(),
            "end_date": datetime.now().isoformat(),
            "metrics": {
                "total_queries": 1250,
                "successful_queries": 1180,
                "failed_queries": 70,
                "average_response_time": 0.45,
                "average_confidence": 0.78,
                "unique_users": 85
            },
            "popular_queries": [
                {"query": "artificial intelligence", "count": 45, "avg_confidence": 0.85},
                {"query": "machine learning", "count": 38, "avg_confidence": 0.82},
                {"query": "data science", "count": 32, "avg_confidence": 0.79},
                {"query": "neural networks", "count": 28, "avg_confidence": 0.81},
                {"query": "deep learning", "count": 24, "avg_confidence": 0.83}
            ],
            "performance_trends": [
                {"date": "2024-01-01", "avg_response_time": 0.42, "query_count": 180},
                {"date": "2024-01-02", "avg_response_time": 0.45, "query_count": 195},
                {"date": "2024-01-03", "avg_response_time": 0.48, "query_count": 210},
                {"date": "2024-01-04", "avg_response_time": 0.43, "query_count": 185},
                {"date": "2024-01-05", "avg_response_time": 0.41, "query_count": 220}
            ],
            "confidence_distribution": {
                "0.0-0.2": 5,
                "0.2-0.4": 12,
                "0.4-0.6": 38,
                "0.6-0.8": 145,
                "0.8-1.0": 180
            }
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Query analytics failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analytics: {str(e)}"
        )


@router.post("/query/feedback")
async def submit_query_feedback(
    query_id: str,
    feedback: Dict[str, Any],
    user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Submit feedback for a query result.
    
    Features:
    - Result relevance feedback
    - Quality ratings
    - Improvement suggestions
    """
    try:
        # Validate feedback
        required_fields = ["rating", "relevance"]
        for field in required_fields:
            if field not in feedback:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        # Store feedback (mock implementation)
        feedback_record = {
            "id": f"fb_{query_id}_{datetime.now().timestamp()}",
            "query_id": query_id,
            "user_id": user["user_id"],
            "rating": feedback["rating"],
            "relevance": feedback["relevance"],
            "comments": feedback.get("comments", ""),
            "suggestions": feedback.get("suggestions", []),
            "timestamp": datetime.now().isoformat()
        }
        
        # Log feedback for analytics
        logger.info(f"Query feedback received: {feedback_record}")
        
        return {
            "message": "Feedback submitted successfully",
            "feedback_id": feedback_record["id"],
            "status": "received"
        }
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit feedback: {str(e)}"
        )


# Helper functions
async def _log_advanced_query(query: str, user_id: str, result: EnhancedQueryResult, db):
    """Log advanced query for analytics."""
    try:
        # Implementation would store in database
        logger.info(f"Advanced query logged: {query} by {user_id}")
    except Exception as e:
        logger.error(f"Failed to log query: {e}")


async def _get_user_query_history(user_id: str, db, limit: int = 20) -> List[str]:
    """Get user's query history."""
    try:
        # Mock implementation - would query database
        return [
            "artificial intelligence applications",
            "machine learning algorithms",
            "data science techniques",
            "neural network architectures",
            "natural language processing"
        ]
    except Exception as e:
        logger.error(f"Failed to get query history: {e}")
        return []


async def get_current_user():
    """Get current user (simplified for testing)."""
    return {"user_id": "test_user", "username": "test_user"}
