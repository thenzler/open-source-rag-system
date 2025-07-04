"""
Query Service - LLM Integration and Response Generation
Handles query processing, retrieval, and response generation using local LLMs.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
import httpx
import openai
from langchain.llms import Ollama
from langchain.schema import LLMResult
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import redis.asyncio as redis

from app.core.config import get_settings
from app.services.vector_service import VectorService
from app.core.exceptions import QueryError, LLMError
from app.models.database import QueryLog, QueryResult, DocumentChunk
from app.schemas import QueryRequest, AdvancedQueryRequest

logger = logging.getLogger(__name__)
settings = get_settings()


class QueryService:
    """Main query processing service integrating retrieval and generation."""
    
    def __init__(self):
        self.vector_service = VectorService()
        self.redis_client: Optional[redis.Redis] = None
        self.llm_client = None
        self.reranker = None
        self.query_cache = {}
        
        # Response templates
        self.templates = {
            "no_sources": "I don't have enough information in the available documents to answer your question accurately. Please try rephrasing your query or check if relevant documents have been uploaded.",
            "low_confidence": "Based on the available documents, here's what I found, though I'm not entirely certain:",
            "standard": "Based on the available documents:",
            "high_confidence": "According to the documents in the knowledge base:"
        }
    
    async def initialize(self):
        """Initialize the query service components."""
        try:
            # Initialize vector service
            await self.vector_service.initialize()
            
            # Initialize Redis
            self.redis_client = redis.from_url(settings.redis_url)
            
            # Initialize LLM client
            await self._initialize_llm()
            
            # Initialize re-ranker if enabled
            if settings.enable_reranking:
                await self._initialize_reranker()
            
            logger.info("Query service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize query service: {e}")
            raise
    
    async def _initialize_llm(self):
        """Initialize the LLM client based on configuration."""
        try:
            if settings.llm_provider == "ollama":
                self.llm_client = OllamaClient(
                    base_url=settings.llm_service_url,
                    model=settings.llm_model_name
                )
            elif settings.llm_provider == "openai":
                self.llm_client = OpenAIClient(
                    api_key=settings.openai_api_key,
                    model=settings.llm_model_name
                )
            elif settings.llm_provider == "vllm":
                self.llm_client = VLLMClient(
                    base_url=settings.vllm_service_url,
                    model=settings.llm_model_name
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
            
            # Test LLM connection
            await self.llm_client.health_check()
            logger.info(f"LLM client initialized: {settings.llm_provider}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise LLMError(f"LLM initialization failed: {e}")
    
    async def _initialize_reranker(self):
        """Initialize the re-ranking model."""
        try:
            if settings.rerank_model == "cross-encoder":
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            else:
                logger.warning(f"Unknown reranker model: {settings.rerank_model}")
                self.reranker = None
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {e}")
            self.reranker = None
    
    async def query_documents(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        user_id: str = None,
        db = None
    ) -> Dict[str, Any]:
        """Process a basic semantic query."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, top_k, min_score, filters)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_result
            
            # Perform vector search
            search_start = time.time()
            search_results = await self.vector_service.search_similar(
                query=query,
                top_k=top_k * 2,  # Get more for potential re-ranking
                score_threshold=min_score,
                filters=filters
            )
            search_time = int((time.time() - search_start) * 1000)
            
            if not search_results:
                return self._create_no_results_response(query, search_time)
            
            # Apply re-ranking if enabled
            if self.reranker and len(search_results) > 1:
                rerank_start = time.time()
                search_results = await self._rerank_results(query, search_results)
                rerank_time = int((time.time() - rerank_start) * 1000)
            else:
                rerank_time = 0
            
            # Limit to requested number of results
            search_results = search_results[:top_k]
            
            # Generate response
            llm_start = time.time()
            response_text, confidence = await self._generate_response(query, search_results)
            llm_time = int((time.time() - llm_start) * 1000)
            
            # Prepare result
            result = {
                "query": query,
                "response": response_text,
                "sources": self._format_sources(search_results),
                "total_sources": len(search_results),
                "confidence_score": confidence,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "retrieval_strategy": "semantic",
                "retrieval_metrics": {
                    "search_time_ms": search_time,
                    "rerank_time_ms": rerank_time,
                    "llm_time_ms": llm_time,
                    "total_candidates": len(search_results)
                }
            }
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise QueryError(f"Query processing failed: {e}")
    
    async def advanced_query(
        self,
        query_request: AdvancedQueryRequest,
        user_id: str = None,
        db = None
    ) -> Dict[str, Any]:
        """Process an advanced query with multiple strategies."""
        start_time = time.time()
        
        try:
            query = query_request.query
            
            # Query expansion if enabled
            if query_request.expand_query:
                expanded_query = await self._expand_query(query)
            else:
                expanded_query = query
            
            # Choose retrieval strategy
            if query_request.retrieval_strategy == "hybrid":
                search_results = await self.vector_service.search_hybrid(
                    query=expanded_query,
                    top_k=query_request.top_k * 3,
                    semantic_weight=1.0 - query_request.keyword_boost,
                    keyword_weight=query_request.keyword_boost,
                    filters=query_request.filters,
                    document_ids=query_request.document_ids
                )
            elif query_request.retrieval_strategy == "keyword":
                search_results = await self.vector_service._keyword_search(
                    query=expanded_query,
                    top_k=query_request.top_k * 2,
                    filters=query_request.filters,
                    document_ids=query_request.document_ids
                )
            else:  # semantic
                search_results = await self.vector_service.search_similar(
                    query=expanded_query,
                    top_k=query_request.top_k * 2,
                    score_threshold=query_request.semantic_threshold,
                    filters=query_request.filters,
                    document_ids=query_request.document_ids
                )
            
            # Apply re-ranking if requested
            rerank_applied = False
            if query_request.rerank and self.reranker and len(search_results) > 1:
                search_results = await self._rerank_results(query, search_results)
                rerank_applied = True
            
            # Limit to final number of results
            final_k = query_request.final_k or query_request.top_k
            search_results = search_results[:final_k]
            
            # Generate response
            response_text, confidence = await self._generate_response(query, search_results)
            
            # Prepare advanced result
            result = {
                "query": query,
                "expanded_query": expanded_query if query_request.expand_query else None,
                "response": response_text,
                "sources": self._format_sources(search_results),
                "total_sources": len(search_results),
                "confidence_score": confidence,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "retrieval_strategy": query_request.retrieval_strategy,
                "reranking_applied": rerank_applied,
                "retrieval_metrics": {
                    "initial_candidates": query_request.top_k * 3 if query_request.retrieval_strategy == "hybrid" else query_request.top_k * 2,
                    "post_reranking": len(search_results),
                    "final_results": final_k
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Advanced query processing failed: {e}")
            raise QueryError(f"Advanced query processing failed: {e}")
    
    async def find_similar_documents(
        self,
        document_id: str,
        top_k: int = 5,
        min_score: float = 0.7,
        user_id: str = None,
        db = None
    ) -> Dict[str, Any]:
        """Find documents similar to a given document."""
        try:
            # Get document vectors
            document_vectors = await self.vector_service.get_document_vectors(document_id)
            
            if not document_vectors:
                raise QueryError(f"Document {document_id} not found or has no vectors")
            
            # Use the first chunk's text as query (could be improved)
            sample_text = document_vectors[0]['payload'].get('text', '')
            if not sample_text:
                raise QueryError("Document has no extractable text for similarity search")
            
            # Perform similarity search excluding the source document
            search_results = await self.vector_service.search_similar(
                query=sample_text,
                top_k=top_k * 2,  # Get more to filter out source document
                score_threshold=min_score,
                filters={"document_id": {"$ne": document_id}}  # Exclude source document
            )
            
            # Group by document and calculate document-level similarity
            doc_similarities = {}
            for result in search_results:
                doc_id = result['payload']['document_id']
                if doc_id == document_id:  # Skip source document
                    continue
                
                if doc_id not in doc_similarities:
                    doc_similarities[doc_id] = {
                        'document_id': doc_id,
                        'filename': result['payload'].get('filename', 'Unknown'),
                        'scores': [],
                        'chunks': []
                    }
                
                doc_similarities[doc_id]['scores'].append(result['score'])
                doc_similarities[doc_id]['chunks'].append(result)
            
            # Calculate average similarity per document
            similar_docs = []
            for doc_id, data in doc_similarities.items():
                avg_score = sum(data['scores']) / len(data['scores'])
                similar_docs.append({
                    'document_id': doc_id,
                    'filename': data['filename'],
                    'similarity_score': avg_score,
                    'matching_chunks': len(data['chunks']),
                    'metadata': data['chunks'][0]['payload'].get('metadata', {})
                })
            
            # Sort by similarity and limit results
            similar_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
            similar_docs = similar_docs[:top_k]
            
            return {
                'source_document': {
                    'document_id': document_id,
                    'vector_count': len(document_vectors)
                },
                'similar_documents': similar_docs
            }
            
        except Exception as e:
            logger.error(f"Similar document search failed: {e}")
            raise QueryError(f"Similar document search failed: {e}")
    
    async def _generate_response(self, query: str, search_results: List[Dict]) -> Tuple[str, float]:
        """Generate LLM response from search results."""
        if not search_results:
            return self.templates["no_sources"], 0.1
        
        # Prepare context from search results
        context_chunks = []
        total_chars = 0
        max_context_chars = 4000  # Leave room for query and response
        
        for result in search_results:
            chunk_text = result['payload'].get('text', '')
            if chunk_text and total_chars + len(chunk_text) < max_context_chars:
                # Add source reference
                doc_info = f"[Source: {result['payload'].get('filename', 'Unknown')}]"
                context_chunks.append(f"{doc_info}\n{chunk_text}")
                total_chars += len(chunk_text) + len(doc_info)
        
        if not context_chunks:
            return self.templates["no_sources"], 0.1
        
        context = "\n\n".join(context_chunks)
        
        # Calculate confidence based on scores
        avg_score = sum(r['score'] for r in search_results) / len(search_results)
        confidence = min(avg_score * 1.2, 1.0)  # Boost and cap at 1.0
        
        # Choose template based on confidence
        if confidence < 0.3:
            template_key = "low_confidence"
        elif confidence > 0.8:
            template_key = "high_confidence"
        else:
            template_key = "standard"
        
        # Prepare prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on provided documents. Always ground your answers in the provided context and be accurate.

Context from documents:
{context}

Question: {query}

Instructions:
1. Answer the question based ONLY on the information provided in the context
2. Be specific and accurate
3. If the context doesn't contain enough information to answer the question, say so
4. Cite relevant information from the sources when possible
5. Keep your answer concise but complete

Answer:"""

        try:
            # Generate response using LLM
            response_text = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature
            )
            
            # Combine template with response
            if template_key != "standard":
                response_text = f"{self.templates[template_key]} {response_text}"
            else:
                response_text = f"{self.templates[template_key]} {response_text}"
            
            return response_text.strip(), confidence
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"I found relevant information but encountered an error generating the response. Please try again.", 0.2
    
    async def _rerank_results(self, query: str, search_results: List[Dict]) -> List[Dict]:
        """Re-rank search results using cross-encoder."""
        if not self.reranker or len(search_results) <= 1:
            return search_results
        
        try:
            # Prepare query-document pairs
            pairs = []
            for result in search_results:
                doc_text = result['payload'].get('text', '')[:512]  # Limit text length
                pairs.append([query, doc_text])
            
            # Get cross-encoder scores
            scores = self.reranker.predict(pairs)
            
            # Update scores and re-sort
            for i, result in enumerate(search_results):
                result['rerank_score'] = float(scores[i])
                result['original_score'] = result['score']
                result['score'] = float(scores[i])  # Use rerank score as new score
            
            # Sort by rerank scores
            search_results.sort(key=lambda x: x['score'], reverse=True)
            
            return search_results
            
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")
            return search_results
    
    async def _expand_query(self, query: str) -> str:
        """Expand query with related terms."""
        try:
            expansion_prompt = f"""Given the following query, suggest 2-3 related terms or synonyms that could help find more relevant information. Return only the additional terms, separated by commas, without explanation.

Query: {query}

Additional terms:"""
            
            additional_terms = await self.llm_client.generate(
                prompt=expansion_prompt,
                max_tokens=50,
                temperature=0.3
            )
            
            # Clean and combine
            terms = [term.strip() for term in additional_terms.split(',') if term.strip()]
            if terms:
                expanded = f"{query} {' '.join(terms[:3])}"  # Limit to 3 additional terms
                return expanded
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
        
        return query
    
    def _format_sources(self, search_results: List[Dict]) -> List[Dict]:
        """Format search results as source citations."""
        sources = []
        
        for result in search_results:
            payload = result['payload']
            
            source = {
                'document_id': payload.get('document_id'),
                'filename': payload.get('filename', 'Unknown'),
                'chunk_id': payload.get('chunk_id'),
                'chunk_index': payload.get('chunk_index', 0),
                'relevance_score': result['score'],
                'page_number': payload.get('page_number'),
                'text_snippet': payload.get('text', '')[:500],  # Limit snippet length
                'start_char': payload.get('start_char'),
                'end_char': payload.get('end_char'),
                'metadata': payload.get('chunk_metadata', {})
            }
            
            # Add rerank score if available
            if 'rerank_score' in result:
                source['rerank_score'] = result['rerank_score']
                source['original_score'] = result.get('original_score', result['score'])
            
            sources.append(source)
        
        return sources
    
    def _create_no_results_response(self, query: str, search_time: int) -> Dict[str, Any]:
        """Create response when no results are found."""
        return {
            "query": query,
            "response": self.templates["no_sources"],
            "sources": [],
            "total_sources": 0,
            "confidence_score": 0.0,
            "processing_time_ms": search_time,
            "retrieval_strategy": "semantic",
            "retrieval_metrics": {
                "search_time_ms": search_time,
                "total_candidates": 0
            }
        }
    
    def _generate_cache_key(self, query: str, top_k: int, min_score: float, filters: Optional[Dict]) -> str:
        """Generate cache key for query."""
        import hashlib
        
        cache_data = {
            'query': query,
            'top_k': top_k,
            'min_score': min_score,
            'filters': filters or {}
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"query:{hashlib.md5(cache_string.encode()).hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached query result."""
        if not self.redis_client or not settings.enable_caching:
            return None
        
        try:
            cached = await self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: Dict):
        """Cache query result."""
        if not self.redis_client or not settings.enable_caching:
            return
        
        try:
            await self.redis_client.setex(
                cache_key,
                settings.cache_ttl_seconds,
                json.dumps(result, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if query service is healthy."""
        try:
            # Check vector service
            vector_healthy = await self.vector_service.health_check()
            
            # Check LLM service
            llm_healthy = await self.llm_client.health_check()
            
            return vector_healthy and llm_healthy
            
        except Exception:
            return False
    
    async def check_llm_health(self) -> bool:
        """Check LLM service health specifically."""
        try:
            return await self.llm_client.health_check()
        except Exception:
            return False


class OllamaClient:
    """Ollama LLM client implementation."""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        """Generate response using Ollama."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        "top_p": 0.9
                    },
                    "stream": False
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise LLMError(f"Ollama generation failed: {e}")
    
    async def health_check(self) -> bool:
        """Check Ollama service health."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False


class OpenAIClient:
    """OpenAI API client implementation."""
    
    def __init__(self, api_key: str, model: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        """Generate response using OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise LLMError(f"OpenAI generation failed: {e}")
    
    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False


class VLLMClient:
    """vLLM client implementation."""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        """Generate response using vLLM."""
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result['choices'][0]['text'].strip()
            
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise LLMError(f"vLLM generation failed: {e}")
    
    async def health_check(self) -> bool:
        """Check vLLM service health."""
        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            return response.status_code == 200
        except Exception:
            return False
