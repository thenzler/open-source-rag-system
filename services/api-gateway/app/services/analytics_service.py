"""
Analytics Service - Handles logging and analytics for the RAG system
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, text
from sqlalchemy.orm import selectinload

from app.models.queries import QueryLog
from app.models.documents import Document, DocumentChunk
from app.core.config import get_settings
from app.core.exceptions import ProcessingError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

class AnalyticsService:
    """Service for handling analytics, logging, and statistics."""
    
    def __init__(self):
        self.retention_days = 90  # Keep logs for 90 days
        
    async def initialize(self):
        """Initialize the analytics service."""
        logger.info("Analytics service initialized")
    
    async def log_query(
        self,
        query: str,
        user_id: str,
        response: Dict[str, Any],
        db: AsyncSession
    ) -> QueryLog:
        """Log a query and its response."""
        try:
            query_log = QueryLog(
                query_text=query,
                user_id=user_id,
                results_count=response.get('total_results', 0),
                response_time=response.get('response_time', 0.0),
                filters=response.get('filters_applied'),
                results=response.get('results', [])
            )
            
            db.add(query_log)
            await db.commit()
            await db.refresh(query_log)
            
            return query_log
            
        except Exception as e:
            logger.error(f"Error logging query: {e}")
            raise ProcessingError(f"Failed to log query: {str(e)}")
    
    async def get_system_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            # Document statistics
            doc_stats = await self._get_document_stats(db)
            
            # Query statistics
            query_stats = await self._get_query_stats(db)
            
            # Performance statistics
            perf_stats = await self._get_performance_stats(db)
            
            # User statistics
            user_stats = await self._get_user_stats(db)
            
            # Storage statistics
            storage_stats = await self._get_storage_stats(db)
            
            return {
                **doc_stats,
                **query_stats,
                **perf_stats,
                **user_stats,
                **storage_stats,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            raise ProcessingError(f"Failed to get system stats: {str(e)}")
    
    async def _get_document_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get document-related statistics."""
        try:
            # Total documents
            total_docs_query = select(func.count(Document.id))
            total_docs_result = await db.execute(total_docs_query)
            total_documents = total_docs_result.scalar() or 0
            
            # Documents by status
            status_query = select(
                Document.status,
                func.count(Document.id).label('count')
            ).group_by(Document.status)
            
            status_result = await db.execute(status_query)
            documents_by_status = {
                status: count for status, count in status_result.fetchall()
            }
            
            # Documents by type
            type_query = select(
                Document.file_type,
                func.count(Document.id).label('count')
            ).group_by(Document.file_type)
            
            type_result = await db.execute(type_query)
            documents_by_type = {
                file_type: count for file_type, count in type_result.fetchall()
            }
            
            # Recent uploads (last 7 days)
            recent_cutoff = datetime.utcnow() - timedelta(days=7)
            recent_query = select(func.count(Document.id)).where(
                Document.upload_date >= recent_cutoff
            )
            recent_result = await db.execute(recent_query)
            recent_uploads = recent_result.scalar() or 0
            
            # Processing success rate
            completed_query = select(func.count(Document.id)).where(
                Document.status == 'completed'
            )
            completed_result = await db.execute(completed_query)
            completed_docs = completed_result.scalar() or 0
            
            processing_success_rate = (
                (completed_docs / total_documents * 100) if total_documents > 0 else 0
            )
            
            return {
                'total_documents': total_documents,
                'documents_by_status': documents_by_status,
                'documents_by_type': documents_by_type,
                'recent_uploads': recent_uploads,
                'processing_success_rate': round(processing_success_rate, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {}
    
    async def _get_query_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get query-related statistics."""
        try:
            # Total queries
            total_queries_query = select(func.count(QueryLog.id))
            total_queries_result = await db.execute(total_queries_query)
            total_queries = total_queries_result.scalar() or 0
            
            # Recent queries (last 24 hours)
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_queries_query = select(func.count(QueryLog.id)).where(
                QueryLog.created_at >= recent_cutoff
            )
            recent_queries_result = await db.execute(recent_queries_query)
            recent_queries = recent_queries_result.scalar() or 0
            
            # Average results per query
            avg_results_query = select(func.avg(QueryLog.results_count))
            avg_results_result = await db.execute(avg_results_query)
            avg_results = avg_results_result.scalar() or 0.0
            
            # Queries with no results
            no_results_query = select(func.count(QueryLog.id)).where(
                QueryLog.results_count == 0
            )
            no_results_result = await db.execute(no_results_query)
            no_results = no_results_result.scalar() or 0
            
            # Query success rate
            success_rate = (
                ((total_queries - no_results) / total_queries * 100) 
                if total_queries > 0 else 0
            )
            
            return {
                'total_queries': total_queries,
                'recent_queries': recent_queries,
                'avg_results_per_query': round(float(avg_results), 2),
                'queries_with_no_results': no_results,
                'query_success_rate': round(success_rate, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting query stats: {e}")
            return {}
    
    async def _get_performance_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get performance-related statistics."""
        try:
            # Average response time
            avg_response_query = select(func.avg(QueryLog.response_time))
            avg_response_result = await db.execute(avg_response_query)
            avg_response_time = avg_response_result.scalar() or 0.0
            
            # Median response time (approximate)
            median_query = select(QueryLog.response_time).order_by(QueryLog.response_time)
            median_result = await db.execute(median_query)
            response_times = median_result.scalars().all()
            
            median_response_time = 0.0
            if response_times:
                n = len(response_times)
                if n % 2 == 0:
                    median_response_time = (response_times[n//2 - 1] + response_times[n//2]) / 2
                else:
                    median_response_time = response_times[n//2]
            
            # 95th percentile response time
            percentile_95_index = int(len(response_times) * 0.95)
            percentile_95_time = (
                response_times[percentile_95_index] if response_times else 0.0
            )
            
            # Recent performance (last 24 hours)
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_avg_query = select(func.avg(QueryLog.response_time)).where(
                QueryLog.created_at >= recent_cutoff
            )
            recent_avg_result = await db.execute(recent_avg_query)
            recent_avg_response = recent_avg_result.scalar() or 0.0
            
            return {
                'avg_response_time': round(float(avg_response_time), 3),
                'median_response_time': round(float(median_response_time), 3),
                'percentile_95_response_time': round(float(percentile_95_time), 3),
                'recent_avg_response_time': round(float(recent_avg_response), 3)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
    
    async def _get_user_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get user-related statistics."""
        try:
            # Unique users who uploaded documents
            doc_users_query = select(func.count(func.distinct(Document.user_id)))
            doc_users_result = await db.execute(doc_users_query)
            doc_users = doc_users_result.scalar() or 0
            
            # Unique users who made queries
            query_users_query = select(func.count(func.distinct(QueryLog.user_id)))
            query_users_result = await db.execute(query_users_query)
            query_users = query_users_result.scalar() or 0
            
            # Active users (last 7 days)
            recent_cutoff = datetime.utcnow() - timedelta(days=7)
            
            # Active document users
            active_doc_users_query = select(func.count(func.distinct(Document.user_id))).where(
                Document.upload_date >= recent_cutoff
            )
            active_doc_users_result = await db.execute(active_doc_users_query)
            active_doc_users = active_doc_users_result.scalar() or 0
            
            # Active query users
            active_query_users_query = select(func.count(func.distinct(QueryLog.user_id))).where(
                QueryLog.created_at >= recent_cutoff
            )
            active_query_users_result = await db.execute(active_query_users_query)
            active_query_users = active_query_users_result.scalar() or 0
            
            # Top users by queries
            top_users_query = select(
                QueryLog.user_id,
                func.count(QueryLog.id).label('query_count')
            ).group_by(QueryLog.user_id).order_by(
                func.count(QueryLog.id).desc()
            ).limit(5)
            
            top_users_result = await db.execute(top_users_query)
            top_users = [
                {'user_id': user_id, 'query_count': count}
                for user_id, count in top_users_result.fetchall()
            ]
            
            return {
                'total_document_users': doc_users,
                'total_query_users': query_users,
                'active_document_users': active_doc_users,
                'active_query_users': active_query_users,
                'top_users_by_queries': top_users
            }
            
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {}
    
    async def _get_storage_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get storage-related statistics."""
        try:
            # Total file size
            total_size_query = select(func.sum(Document.file_size))
            total_size_result = await db.execute(total_size_query)
            total_size = total_size_result.scalar() or 0
            
            # Average file size
            avg_size_query = select(func.avg(Document.file_size))
            avg_size_result = await db.execute(avg_size_query)
            avg_size = avg_size_result.scalar() or 0
            
            # Largest file
            max_size_query = select(func.max(Document.file_size))
            max_size_result = await db.execute(max_size_query)
            max_size = max_size_result.scalar() or 0
            
            # Total chunks
            total_chunks_query = select(func.sum(Document.chunks_count))
            total_chunks_result = await db.execute(total_chunks_query)
            total_chunks = total_chunks_result.scalar() or 0
            
            # Average chunks per document
            avg_chunks_query = select(func.avg(Document.chunks_count))
            avg_chunks_result = await db.execute(avg_chunks_query)
            avg_chunks = avg_chunks_result.scalar() or 0
            
            return {
                'storage_used': int(total_size),
                'avg_file_size': int(avg_size),
                'largest_file_size': int(max_size),
                'total_chunks': int(total_chunks),
                'avg_chunks_per_document': round(float(avg_chunks), 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}
    
    async def get_time_series_data(
        self,
        metric: str,
        days: int = 30,
        db: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """Get time series data for a specific metric."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            if metric == 'document_uploads':
                query = select(
                    func.date(Document.upload_date).label('date'),
                    func.count(Document.id).label('value')
                ).where(
                    Document.upload_date >= cutoff_date
                ).group_by(func.date(Document.upload_date)).order_by('date')
                
            elif metric == 'queries':
                query = select(
                    func.date(QueryLog.created_at).label('date'),
                    func.count(QueryLog.id).label('value')
                ).where(
                    QueryLog.created_at >= cutoff_date
                ).group_by(func.date(QueryLog.created_at)).order_by('date')
                
            elif metric == 'avg_response_time':
                query = select(
                    func.date(QueryLog.created_at).label('date'),
                    func.avg(QueryLog.response_time).label('value')
                ).where(
                    QueryLog.created_at >= cutoff_date
                ).group_by(func.date(QueryLog.created_at)).order_by('date')
                
            else:
                return []
            
            result = await db.execute(query)
            
            return [
                {
                    'date': str(date),
                    'value': float(value) if value else 0.0
                }
                for date, value in result.fetchall()
            ]
            
        except Exception as e:
            logger.error(f"Error getting time series data: {e}")
            return []
    
    async def get_popular_queries(
        self,
        limit: int = 10,
        days: int = 30,
        db: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """Get most popular queries."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = select(
                QueryLog.query_text,
                func.count(QueryLog.id).label('count'),
                func.avg(QueryLog.response_time).label('avg_response_time'),
                func.avg(QueryLog.results_count).label('avg_results')
            ).where(
                QueryLog.created_at >= cutoff_date
            ).group_by(QueryLog.query_text).order_by(
                func.count(QueryLog.id).desc()
            ).limit(limit)
            
            result = await db.execute(query)
            
            return [
                {
                    'query': query_text,
                    'count': count,
                    'avg_response_time': round(float(avg_response_time), 3),
                    'avg_results': round(float(avg_results), 1)
                }
                for query_text, count, avg_response_time, avg_results in result.fetchall()
            ]
            
        except Exception as e:
            logger.error(f"Error getting popular queries: {e}")
            return []
    
    async def get_error_statistics(
        self,
        days: int = 7,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Get error and failure statistics."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Failed document processing
            failed_docs_query = select(func.count(Document.id)).where(
                and_(
                    Document.status == 'failed',
                    Document.upload_date >= cutoff_date
                )
            )
            failed_docs_result = await db.execute(failed_docs_query)
            failed_docs = failed_docs_result.scalar() or 0
            
            # Queries with no results
            no_results_query = select(func.count(QueryLog.id)).where(
                and_(
                    QueryLog.results_count == 0,
                    QueryLog.created_at >= cutoff_date
                )
            )
            no_results_result = await db.execute(no_results_query)
            no_results = no_results_result.scalar() or 0
            
            # Slow queries (> 5 seconds)
            slow_queries_query = select(func.count(QueryLog.id)).where(
                and_(
                    QueryLog.response_time > 5.0,
                    QueryLog.created_at >= cutoff_date
                )
            )
            slow_queries_result = await db.execute(slow_queries_query)
            slow_queries = slow_queries_result.scalar() or 0
            
            return {
                'failed_document_processing': failed_docs,
                'queries_with_no_results': no_results,
                'slow_queries': slow_queries,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting error statistics: {e}")
            return {}
    
    async def cleanup_old_logs(self, db: AsyncSession):
        """Clean up old log entries."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            
            # Delete old query logs
            delete_query = text(
                "DELETE FROM query_logs WHERE created_at < :cutoff_date"
            ).bindparam(cutoff_date=cutoff_date)
            
            result = await db.execute(delete_query)
            deleted_count = result.rowcount
            
            await db.commit()
            
            logger.info(f"Cleaned up {deleted_count} old log entries")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up logs: {e}")
            raise ProcessingError(f"Failed to cleanup logs: {str(e)}")
    
    async def export_analytics_data(
        self,
        start_date: datetime,
        end_date: datetime,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Export analytics data for a date range."""
        try:
            # Query logs in date range
            query_logs_query = select(QueryLog).where(
                and_(
                    QueryLog.created_at >= start_date,
                    QueryLog.created_at <= end_date
                )
            )
            
            query_logs_result = await db.execute(query_logs_query)
            query_logs = query_logs_result.scalars().all()
            
            # Document uploads in date range
            docs_query = select(Document).where(
                and_(
                    Document.upload_date >= start_date,
                    Document.upload_date <= end_date
                )
            )
            
            docs_result = await db.execute(docs_query)
            documents = docs_result.scalars().all()
            
            # Format data for export
            export_data = {
                'export_date': datetime.utcnow().isoformat(),
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'query_logs': [
                    {
                        'id': str(log.id),
                        'query_text': log.query_text,
                        'user_id': log.user_id,
                        'results_count': log.results_count,
                        'response_time': log.response_time,
                        'created_at': log.created_at.isoformat()
                    }
                    for log in query_logs
                ],
                'documents': [
                    {
                        'id': str(doc.id),
                        'filename': doc.original_filename,
                        'file_type': doc.file_type,
                        'file_size': doc.file_size,
                        'status': doc.status,
                        'user_id': doc.user_id,
                        'upload_date': doc.upload_date.isoformat() if doc.upload_date else None
                    }
                    for doc in documents
                ]
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting analytics data: {e}")
            raise ProcessingError(f"Failed to export analytics data: {str(e)}")
