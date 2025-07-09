"""
Analytics service for tracking usage and generating insights.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.documents import Document, DocumentChunk
from app.models.queries import QueryLog, UserSession

logger = logging.getLogger(__name__)
settings = get_settings()


class AnalyticsService:
    """Service for analytics and usage tracking."""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize the analytics service."""
        logger.info("Initializing AnalyticsService")
        self.initialized = True
    
    async def log_query(
        self,
        query: str,
        user_id: str,
        response: Dict[str, Any],
        db: AsyncSession
    ):
        """Log a query for analytics."""
        try:
            query_log = QueryLog(
                query_text=query,
                user_id=user_id,
                result_count=response.get("total_results", 0),
                max_score=response.get("max_score", 0.0),
                avg_score=response.get("avg_score", 0.0),
                response_time_ms=response.get("processing_time_ms", 0),
                embedding_model=settings.embedding_model,
                llm_model=settings.llm_model_name,
                success="true"
            )
            
            db.add(query_log)
            await db.commit()
            
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
    
    async def get_system_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get overall system statistics."""
        try:
            # Document statistics
            doc_count_result = await db.execute(select(func.count(Document.id)))
            total_documents = doc_count_result.scalar() or 0
            
            chunk_count_result = await db.execute(select(func.count(DocumentChunk.id)))
            total_chunks = chunk_count_result.scalar() or 0
            
            # Query statistics
            query_count_result = await db.execute(select(func.count(QueryLog.id)))
            total_queries = query_count_result.scalar() or 0
            
            # User statistics
            user_count_result = await db.execute(
                select(func.count(func.distinct(Document.user_id)))
            )
            total_users = user_count_result.scalar() or 0
            
            # Storage statistics
            storage_result = await db.execute(
                select(func.sum(Document.file_size))
            )
            storage_used = storage_result.scalar() or 0
            
            # Processing queue
            queue_result = await db.execute(
                select(func.count(Document.id)).where(
                    Document.status.in_(["pending", "processing"])
                )
            )
            processing_queue = queue_result.scalar() or 0
            
            # Recent activity (last hour)
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            
            recent_queries_result = await db.execute(
                select(func.count(QueryLog.id)).where(
                    QueryLog.created_at >= one_hour_ago
                )
            )
            queries_last_hour = recent_queries_result.scalar() or 0
            
            recent_uploads_result = await db.execute(
                select(func.count(Document.id)).where(
                    Document.created_at >= one_hour_ago
                )
            )
            uploads_last_hour = recent_uploads_result.scalar() or 0
            
            # Performance metrics
            avg_query_time_result = await db.execute(
                select(func.avg(QueryLog.response_time_ms)).where(
                    QueryLog.created_at >= one_hour_ago
                )
            )
            avg_query_time = avg_query_time_result.scalar()
            
            # Success rate
            success_rate_result = await db.execute(
                select(
                    func.count(QueryLog.id).filter(QueryLog.success == "true") * 100.0 /
                    func.count(QueryLog.id)
                ).where(
                    QueryLog.created_at >= one_hour_ago
                )
            )
            success_rate = success_rate_result.scalar()
            
            return {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "total_queries": total_queries,
                "total_users": total_users,
                "storage_used_bytes": storage_used,
                "processing_queue_size": processing_queue,
                "queries_last_hour": queries_last_hour,
                "uploads_last_hour": uploads_last_hour,
                "avg_query_time_ms": float(avg_query_time) if avg_query_time else None,
                "success_rate": float(success_rate) if success_rate else None,
                "uptime_seconds": 3600,  # Placeholder
                "active_sessions": 0  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
    
    async def get_query_analytics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        granularity: str = "day",
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Get query analytics for a time period."""
        try:
            # Parse dates
            if start_date:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            else:
                start_dt = datetime.utcnow() - timedelta(days=7)
            
            if end_date:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            else:
                end_dt = datetime.utcnow()
            
            # Base query for the time period
            base_query = select(QueryLog).where(
                and_(
                    QueryLog.created_at >= start_dt,
                    QueryLog.created_at <= end_dt
                )
            )
            
            # Total queries
            total_result = await db.execute(
                select(func.count(QueryLog.id)).where(
                    and_(
                        QueryLog.created_at >= start_dt,
                        QueryLog.created_at <= end_dt
                    )
                )
            )
            total_queries = total_result.scalar() or 0
            
            # Unique users
            unique_users_result = await db.execute(
                select(func.count(func.distinct(QueryLog.user_id))).where(
                    and_(
                        QueryLog.created_at >= start_dt,
                        QueryLog.created_at <= end_dt
                    )
                )
            )
            unique_users = unique_users_result.scalar() or 0
            
            # Average response time
            avg_time_result = await db.execute(
                select(func.avg(QueryLog.response_time_ms)).where(
                    and_(
                        QueryLog.created_at >= start_dt,
                        QueryLog.created_at <= end_dt
                    )
                )
            )
            avg_response_time = float(avg_time_result.scalar() or 0)
            
            # Success rate
            success_rate_result = await db.execute(
                select(
                    func.count(QueryLog.id).filter(QueryLog.success == "true") * 100.0 /
                    func.count(QueryLog.id)
                ).where(
                    and_(
                        QueryLog.created_at >= start_dt,
                        QueryLog.created_at <= end_dt
                    )
                )
            )
            success_rate = float(success_rate_result.scalar() or 0)
            
            # Popular queries
            popular_queries_result = await db.execute(
                select(
                    QueryLog.query_text,
                    func.count(QueryLog.id).label('count')
                ).where(
                    and_(
                        QueryLog.created_at >= start_dt,
                        QueryLog.created_at <= end_dt
                    )
                ).group_by(QueryLog.query_text)
                .order_by(desc('count'))
                .limit(10)
            )
            popular_queries = [
                {"query": row.query_text, "count": row.count}
                for row in popular_queries_result.fetchall()
            ]
            
            # Query types
            query_types_result = await db.execute(
                select(
                    QueryLog.query_type,
                    func.count(QueryLog.id).label('count')
                ).where(
                    and_(
                        QueryLog.created_at >= start_dt,
                        QueryLog.created_at <= end_dt
                    )
                ).group_by(QueryLog.query_type)
            )
            query_types = {
                row.query_type: row.count
                for row in query_types_result.fetchall()
            }
            
            # Queries over time (simplified)
            queries_over_time = [
                {"date": start_dt.isoformat(), "count": total_queries}
            ]
            
            return {
                "total_queries": total_queries,
                "unique_users": unique_users,
                "avg_response_time_ms": avg_response_time,
                "success_rate": success_rate,
                "popular_queries": popular_queries,
                "queries_over_time": queries_over_time,
                "query_types": query_types
            }
            
        except Exception as e:
            logger.error(f"Failed to get query analytics: {e}")
            return {}
