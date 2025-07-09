"""
Query service for handling search and retrieval operations.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.exceptions import DocumentNotFoundError, ProcessingError
from app.models.documents import Document, DocumentChunk
from app.schemas.queries import AdvancedQueryRequest

logger = logging.getLogger(__name__)
settings = get_settings()


class QueryService:
    """Service for handling document queries and search operations."""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize the query service."""
        logger.info("Initializing QueryService")
        self.initialized = True
    
    async def health_check(self) -> bool:
        """Check if the query service is healthy."""
        try:
            return self.initialized
        except Exception as e:
            logger.error(f"Query service health check failed: {e}")
            return False
    
    async def check_llm_health(self) -> bool:
        """Check if the LLM service is healthy."""
        try:
            # Simulate LLM health check
            await asyncio.sleep(0.1)
            return True
        except Exception as e:
            logger.error(f"LLM service health check failed: {e}")
            return False
    
    async def query_documents(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        user_id: str = "anonymous",
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Perform semantic search across documents."""
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Build query to get document chunks
            db_query = select(DocumentChunk).options(
                selectinload(DocumentChunk.document)
            ).join(Document)
            
            # Add user filter
            db_query = db_query.where(Document.user_id == user_id)
            
            # Add additional filters
            if filters:
                if "category" in filters:
                    db_query = db_query.where(Document.category == filters["category"])
                if "document_ids" in filters:
                    db_query = db_query.where(Document.id.in_(filters["document_ids"]))
            
            # Execute query
            result = await db.execute(db_query)
            chunks = result.scalars().all()
            
            # Simulate semantic search scoring
            search_results = []
            for chunk in chunks[:top_k]:
                # Simple keyword matching for demo
                score = self._calculate_similarity_score(query, chunk.content)
                if score >= min_score:
                    search_results.append({
                        "document_id": str(chunk.document.id),
                        "chunk_id": str(chunk.id),
                        "content": chunk.content,
                        "score": score,
                        "document_title": chunk.document.title,
                        "document_filename": chunk.document.filename,
                        "chunk_index": chunk.chunk_index,
                        "page_number": chunk.page_number,
                        "metadata": chunk.metadata or {},
                        "highlights": self._extract_highlights(query, chunk.content)
                    })
            
            # Sort by score
            search_results.sort(key=lambda x: x["score"], reverse=True)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "query": query,
                "results": search_results,
                "total_results": len(search_results),
                "max_score": max([r["score"] for r in search_results]) if search_results else 0.0,
                "avg_score": sum([r["score"] for r in search_results]) / len(search_results) if search_results else 0.0,
                "processing_time_ms": processing_time,
                "used_models": {
                    "embedding": settings.embedding_model,
                    "llm": settings.llm_model_name
                },
                "query_expansion": []
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise ProcessingError(f"Query failed: {str(e)}")
    
    def _calculate_similarity_score(self, query: str, content: str) -> float:
        """Simple similarity calculation for demo purposes."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words)
    
    def _extract_highlights(self, query: str, content: str) -> List[str]:
        """Extract highlighted snippets from content."""
        query_words = query.lower().split()
        highlights = []
        
        for word in query_words:
            if word in content.lower():
                # Find the word in context
                words = content.split()
                for i, w in enumerate(words):
                    if word in w.lower():
                        start = max(0, i - 5)
                        end = min(len(words), i + 6)
                        snippet = " ".join(words[start:end])
                        highlights.append(snippet)
                        break
        
        return highlights[:3]  # Return max 3 highlights
    
    async def advanced_query(
        self,
        query_request: AdvancedQueryRequest,
        user_id: str = "anonymous",
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Perform advanced query with additional features."""
        try:
            # For now, just call the basic query with expanded parameters
            filters = query_request.filters or {}
            
            if query_request.document_ids:
                filters["document_ids"] = query_request.document_ids
            
            result = await self.query_documents(
                query=query_request.query,
                top_k=query_request.top_k,
                min_score=query_request.min_score,
                filters=filters,
                user_id=user_id,
                db=db
            )
            
            # Add advanced query features
            if query_request.enable_query_expansion:
                result["query_expansion"] = self._expand_query(query_request.query)
            
            return result
            
        except Exception as e:
            logger.error(f"Advanced query failed: {e}")
            raise ProcessingError(f"Advanced query failed: {str(e)}")
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms."""
        # Simple query expansion for demo
        expansions = []
        words = query.lower().split()
        
        # Add some simple expansions
        synonym_map = {
            "search": ["find", "locate", "discover"],
            "document": ["file", "paper", "text"],
            "information": ["data", "details", "facts"]
        }
        
        for word in words:
            if word in synonym_map:
                expansions.extend(synonym_map[word])
        
        return expansions[:5]  # Return max 5 expansions
    
    async def find_similar_documents(
        self,
        document_id: str,
        top_k: int = 5,
        min_score: float = 0.7,
        user_id: str = "anonymous",
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Find documents similar to a given document."""
        start_time = time.time()
        
        try:
            # Get the source document
            source_query = select(Document).where(
                and_(Document.id == document_id, Document.user_id == user_id)
            )
            result = await db.execute(source_query)
            source_doc = result.scalar_one_or_none()
            
            if not source_doc:
                raise DocumentNotFoundError("Source document not found")
            
            # Get all other documents for the user
            docs_query = select(Document).where(
                and_(
                    Document.user_id == user_id,
                    Document.id != document_id,
                    Document.status == "completed"
                )
            )
            result = await db.execute(docs_query)
            all_docs = result.scalars().all()
            
            # Calculate similarity (simplified)
            similar_docs = []
            for doc in all_docs:
                score = self._calculate_document_similarity(source_doc, doc)
                if score >= min_score:
                    similar_docs.append({
                        "document_id": str(doc.id),
                        "title": doc.title,
                        "filename": doc.filename,
                        "similarity_score": score,
                        "category": doc.category,
                        "created_at": doc.created_at,
                        "chunk_matches": 1,  # Simplified
                        "content_preview": (doc.summary or doc.content or "")[:200] if doc.summary or doc.content else None
                    })
            
            # Sort by similarity score
            similar_docs.sort(key=lambda x: x["similarity_score"], reverse=True)
            similar_docs = similar_docs[:top_k]
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "source_document_id": document_id,
                "similar_documents": similar_docs,
                "total_found": len(similar_docs),
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            logger.error(f"Similar document search failed: {e}")
            raise ProcessingError(f"Similar document search failed: {str(e)}")
    
    def _calculate_document_similarity(self, doc1: Document, doc2: Document) -> float:
        """Calculate similarity between two documents."""
        # Simple similarity based on category and content overlap
        score = 0.0
        
        # Category match
        if doc1.category and doc2.category and doc1.category == doc2.category:
            score += 0.3
        
        # Title similarity
        if doc1.title and doc2.title:
            title_sim = self._calculate_similarity_score(doc1.title, doc2.title)
            score += title_sim * 0.4
        
        # Content similarity (if available)
        if doc1.content and doc2.content:
            content_sim = self._calculate_similarity_score(doc1.content[:500], doc2.content[:500])
            score += content_sim * 0.3
        
        return min(score, 1.0)  # Cap at 1.0
