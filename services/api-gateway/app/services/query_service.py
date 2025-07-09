"""
Query Service - Handles semantic search and query processing
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from sqlalchemy.orm import selectinload

from app.models.queries import QueryLog
from app.models.documents import Document, DocumentChunk
from app.core.config import get_settings
from app.core.exceptions import ValidationError, ProcessingError
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

class QueryService:
    """Service for handling semantic search and query processing."""
    
    def __init__(self):
        self.vector_engine_url = settings.vector_engine_url
        self.llm_service_url = settings.llm_service_url
        self.max_query_length = settings.max_query_length
        self.max_search_results = settings.max_search_results
        self.enable_query_expansion = settings.enable_query_expansion
        self.enable_reranking = settings.enable_reranking
        self.enable_caching = settings.enable_caching
        
    async def initialize(self):
        """Initialize the query service."""
        logger.info("Query service initialized")
        await self._check_services_health()
    
    async def _check_services_health(self):
        """Check if required services are healthy."""
        try:
            # Check vector engine
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.vector_engine_url}/health", timeout=5.0)
                if response.status_code != 200:
                    logger.warning("Vector engine not healthy")
                else:
                    logger.info("Vector engine is healthy")
        except Exception as e:
            logger.warning(f"Cannot reach vector engine: {e}")
        
        try:
            # Check LLM service
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.llm_service_url}/api/health", timeout=5.0)
                if response.status_code != 200:
                    logger.warning("LLM service not healthy")
                else:
                    logger.info("LLM service is healthy")
        except Exception as e:
            logger.warning(f"Cannot reach LLM service: {e}")
    
    def _validate_query(self, query: str) -> str:
        """Validate and clean query input."""
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        
        query = query.strip()
        
        if len(query) > self.max_query_length:
            raise ValidationError(f"Query too long. Maximum length: {self.max_query_length}")
        
        return query
    
    async def _expand_query(self, query: str) -> List[str]:
        """Expand query using LLM if enabled."""
        if not self.enable_query_expansion:
            return [query]
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.llm_service_url}/api/generate",
                    json={
                        "model": settings.llm_model_name,
                        "prompt": f"Generate 3 similar search queries for: '{query}'. Return only the queries, one per line.",
                        "stream": False
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    expanded_queries = [query]  # Original query first
                    
                    # Parse expanded queries
                    if "response" in result:
                        lines = result["response"].strip().split('\n')
                        for line in lines:
                            line = line.strip()
                            if line and line != query:
                                expanded_queries.append(line)
                    
                    return expanded_queries[:4]  # Limit to 4 queries max
                else:
                    logger.warning(f"Query expansion failed: {response.status_code}")
                    return [query]
                    
        except Exception as e:
            logger.warning(f"Query expansion error: {e}")
            return [query]
    
    async def _search_vectors(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search vectors using the vector engine."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.vector_engine_url}/vectors/search",
                    json={
                        "query": query,
                        "top_k": top_k,
                        "min_score": min_score,
                        "filters": filters or {},
                        "collection_name": "documents"
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("results", [])
                else:
                    logger.error(f"Vector search failed: {response.status_code} - {response.text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    async def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank results using LLM if enabled."""
        if not self.enable_reranking or len(results) <= 1:
            return results
        
        try:
            # Prepare context for reranking
            contexts = []
            for i, result in enumerate(results):
                contexts.append({
                    "index": i,
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0)
                })
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.llm_service_url}/api/generate",
                    json={
                        "model": settings.llm_model_name,
                        "prompt": f"""Rerank these search results for the query: "{query}"
                        
Consider relevance, accuracy, and helpfulness. Return only the indices in order of relevance (most relevant first).
Format: just the numbers separated by commas (e.g., 2,0,1,3)

Results:
{chr(10).join([f"{i}: {ctx['content'][:200]}..." for i, ctx in enumerate(contexts)])}

Reranked order:""",
                        "stream": False
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "response" in result:
                        # Parse reranked indices
                        indices_str = result["response"].strip()
                        try:
                            indices = [int(x.strip()) for x in indices_str.split(',')]
                            # Reorder results based on LLM ranking
                            reranked = []
                            for idx in indices:
                                if 0 <= idx < len(results):
                                    reranked.append(results[idx])
                            
                            # Add any missing results at the end
                            used_indices = set(indices)
                            for i, result in enumerate(results):
                                if i not in used_indices:
                                    reranked.append(result)
                            
                            return reranked[:len(results)]
                        except (ValueError, IndexError):
                            logger.warning("Failed to parse reranking response")
                            return results
                else:
                    logger.warning(f"Reranking failed: {response.status_code}")
                    return results
                    
        except Exception as e:
            logger.warning(f"Reranking error: {e}")
            return results
    
    async def _get_document_info(
        self,
        document_ids: List[str],
        db: AsyncSession
    ) -> Dict[str, Document]:
        """Get document information for search results."""
        try:
            query = select(Document).where(Document.id.in_(document_ids))
            result = await db.execute(query)
            documents = result.scalars().all()
            
            return {doc.id: doc for doc in documents}
            
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            return {}
    
    async def query_documents(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        user_id: str = None,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Perform semantic search across documents."""
        start_time = time.time()
        
        try:
            # Validate query
            query = self._validate_query(query)
            
            # Limit results
            top_k = min(top_k, self.max_search_results)
            
            # Add user filter if provided
            if user_id and filters:
                filters['user_id'] = user_id
            elif user_id:
                filters = {'user_id': user_id}
            
            # Expand query if enabled
            queries = await self._expand_query(query)
            
            # Search vectors for all queries
            all_results = []
            for search_query in queries:
                results = await self._search_vectors(
                    search_query,
                    top_k=top_k * 2,  # Get more results for better reranking
                    min_score=min_score,
                    filters=filters
                )
                all_results.extend(results)
            
            # Deduplicate results by ID
            seen_ids = set()
            unique_results = []
            for result in all_results:
                result_id = result.get('id')
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    unique_results.append(result)
            
            # Sort by score and limit
            unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            unique_results = unique_results[:top_k * 2]
            
            # Rerank results if enabled
            if self.enable_reranking:
                unique_results = await self._rerank_results(query, unique_results)
            
            # Limit to final top_k
            unique_results = unique_results[:top_k]
            
            # Get document information
            document_ids = [r.get('metadata', {}).get('document_id') for r in unique_results]
            document_ids = [doc_id for doc_id in document_ids if doc_id]
            
            documents_info = await self._get_document_info(document_ids, db)
            
            # Format results
            formatted_results = []
            for result in unique_results:
                metadata = result.get('metadata', {})
                document_id = metadata.get('document_id')
                
                formatted_result = {
                    'id': result.get('id'),
                    'score': result.get('score', 0.0),
                    'content': result.get('content', ''),
                    'metadata': metadata,
                    'document_id': document_id
                }
                
                # Add document information
                if document_id in documents_info:
                    doc = documents_info[document_id]
                    formatted_result.update({
                        'source_document': doc.original_filename,
                        'document_type': doc.file_type,
                        'document_upload_date': doc.upload_date.isoformat() if doc.upload_date else None
                    })
                
                formatted_results.append(formatted_result)
            
            response_time = time.time() - start_time
            
            return {
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results),
                'response_time': response_time,
                'expanded_queries': queries if len(queries) > 1 else None,
                'filters_applied': filters
            }
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            raise ProcessingError(f"Query processing failed: {str(e)}")
    
    async def get_similar_documents(
        self,
        document_id: str,
        top_k: int = 5,
        user_id: str = None,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Find documents similar to a given document."""
        try:
            # Get document chunks
            chunks_query = select(DocumentChunk).where(
                DocumentChunk.document_id == document_id
            ).limit(3)  # Use first 3 chunks as representative
            
            result = await db.execute(chunks_query)
            chunks = result.scalars().all()
            
            if not chunks:
                return {
                    'document_id': document_id,
                    'results': [],
                    'total_results': 0,
                    'response_time': 0.0
                }
            
            # Use first chunk content as query
            query_content = chunks[0].content
            
            # Search for similar documents
            search_results = await self.query_documents(
                query=query_content,
                top_k=top_k + 1,  # +1 to exclude the original document
                user_id=user_id,
                db=db
            )
            
            # Filter out the original document
            filtered_results = [
                r for r in search_results['results'] 
                if r.get('document_id') != document_id
            ]
            
            return {
                'document_id': document_id,
                'results': filtered_results[:top_k],
                'total_results': len(filtered_results),
                'response_time': search_results['response_time']
            }
            
        except Exception as e:
            logger.error(f"Similar documents error: {e}")
            raise ProcessingError(f"Similar documents search failed: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check if the query service is healthy."""
        try:
            # Check vector engine
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.vector_engine_url}/health", timeout=5.0)
                if response.status_code != 200:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Query service health check failed: {e}")
            return False
    
    async def check_llm_health(self) -> bool:
        """Check if the LLM service is healthy."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.llm_service_url}/api/health", timeout=5.0)
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"LLM service health check failed: {e}")
            return False
    
    async def get_query_suggestions(
        self,
        partial_query: str,
        limit: int = 5,
        db: AsyncSession = None
    ) -> List[str]:
        """Get query suggestions based on partial input."""
        try:
            # Get recent queries that match the partial input
            query = select(QueryLog.query_text).where(
                QueryLog.query_text.ilike(f"%{partial_query}%")
            ).group_by(QueryLog.query_text).limit(limit)
            
            result = await db.execute(query)
            suggestions = [row[0] for row in result.fetchall()]
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Query suggestions error: {e}")
            return []
    
    async def get_search_analytics(
        self,
        days: int = 30,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Get search analytics data."""
        try:
            # Query stats for the last N days
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Total queries
            total_query = select(func.count(QueryLog.id)).where(
                QueryLog.created_at >= cutoff_date
            )
            total_result = await db.execute(total_query)
            total_queries = total_result.scalar() or 0
            
            # Average response time
            avg_time_query = select(func.avg(QueryLog.response_time)).where(
                QueryLog.created_at >= cutoff_date
            )
            avg_time_result = await db.execute(avg_time_query)
            avg_response_time = avg_time_result.scalar() or 0.0
            
            # Top queries
            top_queries_query = select(
                QueryLog.query_text,
                func.count(QueryLog.id).label('count')
            ).where(
                QueryLog.created_at >= cutoff_date
            ).group_by(QueryLog.query_text).order_by(
                func.count(QueryLog.id).desc()
            ).limit(10)
            
            top_queries_result = await db.execute(top_queries_query)
            top_queries = [
                {'query': query, 'count': count}
                for query, count in top_queries_result.fetchall()
            ]
            
            # Queries by day
            daily_query = select(
                func.date(QueryLog.created_at).label('date'),
                func.count(QueryLog.id).label('count')
            ).where(
                QueryLog.created_at >= cutoff_date
            ).group_by(func.date(QueryLog.created_at)).order_by('date')
            
            daily_result = await db.execute(daily_query)
            daily_stats = [
                {'date': str(date), 'count': count}
                for date, count in daily_result.fetchall()
            ]
            
            return {
                'total_queries': total_queries,
                'avg_response_time': float(avg_response_time),
                'top_queries': top_queries,
                'daily_stats': daily_stats,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Search analytics error: {e}")
            return {}
