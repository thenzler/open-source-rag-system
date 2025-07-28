"""
Advanced Query Service with Enhanced Capabilities
Provides query expansion, semantic filtering, and intelligent response generation.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import re
from dataclasses import dataclass, asdict

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import get_settings
from app.models.documents import Document, DocumentChunk
from app.models.queries import QueryLog, QueryExpansion
from app.services.query_service import QueryService
from app.core.exceptions import ValidationError, ProcessingError

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class QueryContext:
    """Context information for query processing."""
    user_id: str
    query_history: List[str]
    filters: Dict[str, Any]
    preferences: Dict[str, Any]
    session_id: Optional[str] = None


@dataclass
class SemanticFilter:
    """Semantic filtering configuration."""
    categories: List[str]
    date_range: Optional[Tuple[datetime, datetime]]
    content_types: List[str]
    similarity_threshold: float = 0.5
    exclude_keywords: List[str] = None


@dataclass
class QueryExpansionResult:
    """Result of query expansion."""
    original_query: str
    expanded_queries: List[str]
    synonyms: List[str]
    related_concepts: List[str]
    confidence_score: float


@dataclass
class EnhancedQueryResult:
    """Enhanced query result with additional metadata."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time: float
    confidence_score: float
    query_expansion: QueryExpansionResult
    semantic_clusters: List[Dict[str, Any]]
    suggestions: List[str]
    context: QueryContext


class AdvancedQueryService:
    """Advanced query service with enhanced capabilities."""
    
    def __init__(self):
        self.base_query_service = QueryService()
        self.embedding_model = None
        self.query_cache = {}
        self.synonym_cache = {}
        self.concept_cache = {}
        
    async def initialize(self):
        """Initialize the service."""
        await self.base_query_service.initialize()
        
        # Initialize embedding model for query expansion
        try:
            self.embedding_model = SentenceTransformer(
                settings.embedding_model,
                device=settings.device
            )
            logger.info("Advanced query service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise ProcessingError(f"Service initialization failed: {e}")
    
    async def process_advanced_query(
        self,
        query: str,
        context: QueryContext,
        semantic_filter: Optional[SemanticFilter] = None,
        top_k: int = 10,
        enable_expansion: bool = True,
        enable_clustering: bool = True
    ) -> EnhancedQueryResult:
        """
        Process an advanced query with expansion and filtering.
        
        Args:
            query: The user's query
            context: Query context and user information
            semantic_filter: Optional semantic filtering configuration
            top_k: Number of results to return
            enable_expansion: Whether to enable query expansion
            enable_clustering: Whether to enable result clustering
            
        Returns:
            Enhanced query result with additional metadata
        """
        start_time = datetime.now()
        
        try:
            # 1. Query expansion
            expansion_result = None
            if enable_expansion:
                expansion_result = await self._expand_query(query, context)
                queries_to_process = [query] + expansion_result.expanded_queries
            else:
                queries_to_process = [query]
            
            # 2. Execute queries
            all_results = []
            for q in queries_to_process:
                results = await self._execute_semantic_search(
                    q, semantic_filter, top_k * 2  # Get more results for deduplication
                )
                all_results.extend(results)
            
            # 3. Deduplicate and rank results
            final_results = await self._deduplicate_and_rank(
                all_results, query, top_k
            )
            
            # 4. Apply semantic filtering
            if semantic_filter:
                final_results = await self._apply_semantic_filter(
                    final_results, semantic_filter
                )
            
            # 5. Generate semantic clusters
            clusters = []
            if enable_clustering and final_results:
                clusters = await self._generate_semantic_clusters(final_results)
            
            # 6. Generate suggestions
            suggestions = await self._generate_suggestions(
                query, final_results, context
            )
            
            # 7. Calculate confidence score
            confidence_score = await self._calculate_confidence_score(
                query, final_results, expansion_result
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EnhancedQueryResult(
                query=query,
                results=final_results,
                total_results=len(final_results),
                processing_time=processing_time,
                confidence_score=confidence_score,
                query_expansion=expansion_result,
                semantic_clusters=clusters,
                suggestions=suggestions,
                context=context
            )
            
        except Exception as e:
            logger.error(f"Advanced query processing failed: {e}")
            raise ProcessingError(f"Query processing failed: {e}")
    
    async def _expand_query(
        self,
        query: str,
        context: QueryContext
    ) -> QueryExpansionResult:
        """
        Expand the query with synonyms and related concepts.
        
        Args:
            query: Original query
            context: Query context
            
        Returns:
            Query expansion result
        """
        try:
            # Check cache first
            cache_key = f"expand_{hash(query)}"
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]
            
            # Extract key terms
            key_terms = await self._extract_key_terms(query)
            
            # Generate synonyms
            synonyms = await self._generate_synonyms(key_terms)
            
            # Find related concepts
            related_concepts = await self._find_related_concepts(
                key_terms, context.query_history
            )
            
            # Generate expanded queries
            expanded_queries = await self._generate_expanded_queries(
                query, synonyms, related_concepts
            )
            
            # Calculate confidence based on term coverage
            confidence_score = len(synonyms) / max(len(key_terms), 1)
            
            result = QueryExpansionResult(
                original_query=query,
                expanded_queries=expanded_queries,
                synonyms=synonyms,
                related_concepts=related_concepts,
                confidence_score=confidence_score
            )
            
            # Cache the result
            self.query_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            # Return minimal result on failure
            return QueryExpansionResult(
                original_query=query,
                expanded_queries=[],
                synonyms=[],
                related_concepts=[],
                confidence_score=0.0
            )
    
    async def _execute_semantic_search(
        self,
        query: str,
        semantic_filter: Optional[SemanticFilter],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Execute semantic search with optional filtering."""
        try:
            # Build filter parameters
            filters = {}
            if semantic_filter:
                if semantic_filter.categories:
                    filters['categories'] = semantic_filter.categories
                if semantic_filter.content_types:
                    filters['content_types'] = semantic_filter.content_types
                if semantic_filter.date_range:
                    filters['date_range'] = semantic_filter.date_range
            
            # Execute search through base service
            results = await self.base_query_service.query_documents(
                query=query,
                top_k=top_k,
                filters=filters
            )
            
            return results.get('results', [])
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _deduplicate_and_rank(
        self,
        results: List[Dict[str, Any]],
        original_query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Deduplicate and rank results based on relevance."""
        try:
            # Remove duplicates based on document ID and chunk
            seen = set()
            deduplicated = []
            
            for result in results:
                key = (result.get('document_id'), result.get('chunk_id'))
                if key not in seen:
                    seen.add(key)
                    deduplicated.append(result)
            
            # Rank by relevance score
            deduplicated.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Apply semantic re-ranking
            if self.embedding_model and deduplicated:
                deduplicated = await self._semantic_rerank(
                    deduplicated, original_query
                )
            
            return deduplicated[:top_k]
            
        except Exception as e:
            logger.error(f"Deduplication and ranking failed: {e}")
            return results[:top_k]
    
    async def _semantic_rerank(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Re-rank results using semantic similarity."""
        try:
            # Get query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Get content embeddings
            contents = [r.get('content', '') for r in results]
            content_embeddings = self.embedding_model.encode(contents)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, content_embeddings)[0]
            
            # Update scores with semantic similarity
            for i, result in enumerate(results):
                original_score = result.get('score', 0)
                semantic_score = similarities[i]
                # Combine scores with weights
                combined_score = 0.7 * original_score + 0.3 * semantic_score
                result['score'] = combined_score
                result['semantic_score'] = semantic_score
            
            # Sort by combined score
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic re-ranking failed: {e}")
            return results
    
    async def _apply_semantic_filter(
        self,
        results: List[Dict[str, Any]],
        semantic_filter: SemanticFilter
    ) -> List[Dict[str, Any]]:
        """Apply semantic filtering to results."""
        try:
            filtered_results = []
            
            for result in results:
                # Check similarity threshold
                if result.get('score', 0) < semantic_filter.similarity_threshold:
                    continue
                
                # Check exclude keywords
                if semantic_filter.exclude_keywords:
                    content = result.get('content', '').lower()
                    if any(keyword.lower() in content for keyword in semantic_filter.exclude_keywords):
                        continue
                
                # Check date range
                if semantic_filter.date_range:
                    doc_date = result.get('created_at')
                    if doc_date:
                        if not (semantic_filter.date_range[0] <= doc_date <= semantic_filter.date_range[1]):
                            continue
                
                filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Semantic filtering failed: {e}")
            return results
    
    async def _generate_semantic_clusters(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate semantic clusters from results."""
        try:
            if not results or not self.embedding_model:
                return []
            
            # Get content embeddings
            contents = [r.get('content', '') for r in results]
            embeddings = self.embedding_model.encode(contents)
            
            # Simple clustering based on similarity
            clusters = []
            used_indices = set()
            
            for i, embedding in enumerate(embeddings):
                if i in used_indices:
                    continue
                
                # Find similar results
                similarities = cosine_similarity([embedding], embeddings)[0]
                cluster_indices = [j for j, sim in enumerate(similarities) if sim > 0.7 and j not in used_indices]
                
                if len(cluster_indices) > 1:
                    cluster_results = [results[j] for j in cluster_indices]
                    cluster = {
                        'topic': self._extract_cluster_topic(cluster_results),
                        'results': cluster_results,
                        'size': len(cluster_results),
                        'avg_score': sum(r.get('score', 0) for r in cluster_results) / len(cluster_results)
                    }
                    clusters.append(cluster)
                    used_indices.update(cluster_indices)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Semantic clustering failed: {e}")
            return []
    
    async def _generate_suggestions(
        self,
        query: str,
        results: List[Dict[str, Any]],
        context: QueryContext
    ) -> List[str]:
        """Generate query suggestions based on results and context."""
        try:
            suggestions = []
            
            # Extract common topics from results
            topics = []
            for result in results:
                content = result.get('content', '')
                # Simple topic extraction (could be enhanced with NLP)
                topics.extend(re.findall(r'\b[A-Z][a-z]+\b', content))
            
            # Generate suggestions based on frequent topics
            topic_freq = {}
            for topic in topics:
                topic_freq[topic] = topic_freq.get(topic, 0) + 1
            
            common_topics = sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for topic, freq in common_topics:
                if topic.lower() not in query.lower():
                    suggestions.append(f"What about {topic.lower()}?")
                    suggestions.append(f"Tell me more about {topic.lower()}")
            
            # Add context-based suggestions
            if context.query_history:
                last_query = context.query_history[-1]
                if last_query != query:
                    suggestions.append(f"Compare with: {last_query}")
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
            return []
    
    async def _calculate_confidence_score(
        self,
        query: str,
        results: List[Dict[str, Any]],
        expansion_result: Optional[QueryExpansionResult]
    ) -> float:
        """Calculate confidence score for the query results."""
        try:
            if not results:
                return 0.0
            
            # Base confidence from result scores
            avg_score = sum(r.get('score', 0) for r in results) / len(results)
            
            # Factor in query expansion success
            expansion_factor = 1.0
            if expansion_result:
                expansion_factor = expansion_result.confidence_score
            
            # Factor in result diversity
            diversity_factor = min(len(results) / 10, 1.0)  # Normalize to 0-1
            
            # Combined confidence
            confidence = (avg_score * 0.6 + expansion_factor * 0.2 + diversity_factor * 0.2)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5  # Default confidence
    
    async def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query."""
        # Simple term extraction (could be enhanced with NLP)
        terms = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        # Remove common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        return [term for term in terms if term not in stop_words]
    
    async def _generate_synonyms(self, terms: List[str]) -> List[str]:
        """Generate synonyms for key terms."""
        # Simple synonym mapping (could be enhanced with WordNet or ML)
        synonym_map = {
            'document': ['file', 'paper', 'text', 'record'],
            'search': ['find', 'look', 'query', 'seek'],
            'information': ['data', 'details', 'facts', 'knowledge'],
            'system': ['platform', 'service', 'application', 'tool'],
            'process': ['procedure', 'method', 'approach', 'workflow']
        }
        
        synonyms = []
        for term in terms:
            if term in synonym_map:
                synonyms.extend(synonym_map[term])
        
        return synonyms
    
    async def _find_related_concepts(
        self,
        terms: List[str],
        query_history: List[str]
    ) -> List[str]:
        """Find related concepts based on terms and history."""
        related = []
        
        # Simple concept relation (could be enhanced with knowledge graphs)
        concept_map = {
            'ai': ['machine learning', 'artificial intelligence', 'neural networks'],
            'document': ['processing', 'analysis', 'management'],
            'search': ['retrieval', 'indexing', 'ranking'],
            'data': ['analytics', 'mining', 'science']
        }
        
        for term in terms:
            if term in concept_map:
                related.extend(concept_map[term])
        
        return related
    
    async def _generate_expanded_queries(
        self,
        original_query: str,
        synonyms: List[str],
        related_concepts: List[str]
    ) -> List[str]:
        """Generate expanded queries."""
        expanded = []
        
        # Add synonym-based expansions
        for synonym in synonyms[:3]:  # Limit to avoid too many queries
            expanded.append(f"{original_query} {synonym}")
        
        # Add concept-based expansions
        for concept in related_concepts[:2]:
            expanded.append(f"{original_query} {concept}")
        
        return expanded
    
    def _extract_cluster_topic(self, results: List[Dict[str, Any]]) -> str:
        """Extract topic from cluster results."""
        # Simple topic extraction from most common words
        all_content = ' '.join(r.get('content', '') for r in results)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_content.lower())
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        if word_freq:
            return max(word_freq, key=word_freq.get)
        return "General"
    
    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        try:
            return await self.base_query_service.health_check()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
