"""
Vector Engine Service
Handles embedding generation, vector storage, and similarity search operations.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException
import redis.asyncio as redis

from app.core.config import get_settings
from app.core.exceptions import VectorSearchError, EmbeddingError

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorService:
    """Main vector operations service for embeddings and search."""
    
    def __init__(self):
        self.client: Optional[AsyncQdrantClient] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.redis_client: Optional[redis.Redis] = None
        self.collection_name = settings.qdrant_collection_name
        self.vector_dimension = settings.vector_dimension
        self.device = settings.embedding_device
        
    async def initialize(self):
        """Initialize vector service components."""
        try:
            # Initialize Qdrant client
            self.client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key if hasattr(settings, 'qdrant_api_key') else None
            )
            
            # Initialize Redis for caching
            self.redis_client = redis.from_url(
                settings.redis_url, 
                encoding="utf-8", 
                decode_responses=True
            )
            
            # Load embedding model
            await self._load_embedding_model()
            
            # Initialize collection
            await self._ensure_collection_exists()
            
            logger.info(f"Vector service initialized with model: {settings.embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector service: {e}")
            raise
    
    async def _load_embedding_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            
            # Load model in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.embedding_model = await loop.run_in_executor(
                None, 
                lambda: SentenceTransformer(
                    settings.embedding_model,
                    device=self.device
                )
            )
            
            # Verify vector dimension
            test_embedding = self.embedding_model.encode(["test"])
            actual_dim = test_embedding.shape[1]
            
            if actual_dim != self.vector_dimension:
                logger.warning(f"Model dimension {actual_dim} doesn't match config {self.vector_dimension}")
                self.vector_dimension = actual_dim
            
            logger.info(f"Embedding model loaded successfully. Dimension: {self.vector_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingError(f"Model loading failed: {e}")
    
    async def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists with proper configuration."""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                
                # Create collection with optimized settings
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dimension,
                        distance=models.Distance.COSINE,
                        hnsw_config=models.HnswConfigDiff(
                            m=16,  # Number of connections per node
                            ef_construct=200,  # Size of dynamic candidate list
                            full_scan_threshold=10000,  # Use exact search for small datasets
                            max_indexing_threads=0  # Auto-detect
                        ),
                        quantization_config=models.ScalarQuantization(
                            scalar=models.ScalarQuantizationConfig(
                                type=models.ScalarType.INT8,
                                quantile=0.99,
                                always_ram=True
                            )
                        ) if settings.enable_quantization else None
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        deleted_threshold=0.2,
                        vacuum_min_vector_number=1000,
                        default_segment_number=0  # Auto-detect
                    ),
                    shard_number=1,  # Start with single shard
                    replication_factor=1
                )
                
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise VectorSearchError(f"Collection setup failed: {e}")
    
    async def add_documents(
        self, 
        texts: List[str], 
        metadatas: List[Dict[str, Any]], 
        document_id: str,
        batch_size: Optional[int] = None
    ) -> List[str]:
        """Add documents to the vector database."""
        try:
            if not texts:
                return []
            
            batch_size = batch_size or settings.embedding_batch_size
            vector_ids = []
            
            # Process in batches to avoid memory issues
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                # Generate embeddings
                start_time = time.time()
                embeddings = await self._generate_embeddings(batch_texts)
                embedding_time = time.time() - start_time
                
                logger.info(f"Generated {len(embeddings)} embeddings in {embedding_time:.2f}s")
                
                # Prepare points for insertion
                points = []
                batch_ids = []
                
                for j, (text, metadata, embedding) in enumerate(zip(batch_texts, batch_metadatas, embeddings)):
                    point_id = str(uuid.uuid4())
                    batch_ids.append(point_id)
                    
                    # Ensure metadata is JSON serializable
                    safe_metadata = self._sanitize_metadata(metadata)
                    safe_metadata.update({
                        'text': text[:1000],  # Store first 1000 chars for quick preview
                        'char_count': len(text),
                        'word_count': len(text.split()),
                        'document_id': document_id,
                        'created_at': time.time()
                    })
                    
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=embedding.tolist(),
                            payload=safe_metadata
                        )
                    )
                
                # Insert into Qdrant
                start_time = time.time()
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                insert_time = time.time() - start_time
                
                logger.info(f"Inserted {len(points)} points in {insert_time:.2f}s")
                vector_ids.extend(batch_ids)
            
            # Cache document embeddings for quick access
            await self._cache_document_info(document_id, len(vector_ids))
            
            return vector_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise VectorSearchError(f"Document addition failed: {e}")
    
    async def search_similar(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        try:
            # Generate query embedding
            query_embedding = await self._generate_embeddings([query])
            
            # Build filter conditions
            filter_conditions = []
            
            if document_ids:
                filter_conditions.append(
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=document_ids)
                    )
                )
            
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value)
                            )
                        )
                    else:
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
            
            # Combine filters
            query_filter = None
            if filter_conditions:
                if len(filter_conditions) == 1:
                    query_filter = models.Filter(must=[filter_conditions[0]])
                else:
                    query_filter = models.Filter(must=filter_conditions)
            
            # Perform search
            start_time = time.time()
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding[0].tolist(),
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            search_time = time.time() - start_time
            
            logger.info(f"Vector search completed in {search_time:.3f}s, found {len(search_results)} results")
            
            # Format results
            results = []
            for result in search_results:
                result_data = {
                    'id': str(result.id),
                    'score': float(result.score),
                    'payload': result.payload
                }
                results.append(result_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise VectorSearchError(f"Search failed: {e}")
    
    async def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        filters: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search."""
        try:
            # Semantic search
            semantic_results = await self.search_similar(
                query=query,
                top_k=top_k * 2,  # Get more results for re-ranking
                filters=filters,
                document_ids=document_ids
            )
            
            # Keyword search (simple text matching)
            keyword_results = await self._keyword_search(
                query=query,
                top_k=top_k * 2,
                filters=filters,
                document_ids=document_ids
            )
            
            # Combine and re-rank results
            combined_results = self._combine_search_results(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight
            )
            
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise VectorSearchError(f"Hybrid search failed: {e}")
    
    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Perform keyword-based search using Qdrant's payload filtering."""
        try:
            # Tokenize query
            query_terms = query.lower().split()
            
            # Build text search conditions
            text_conditions = []
            for term in query_terms:
                text_conditions.append(
                    models.FieldCondition(
                        key="text",
                        match=models.MatchText(text=term)
                    )
                )
            
            # Build filter conditions
            filter_conditions = []
            
            if document_ids:
                filter_conditions.append(
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=document_ids)
                    )
                )
            
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value)
                            )
                        )
                    else:
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
            
            # Combine all conditions
            all_conditions = text_conditions + filter_conditions
            query_filter = models.Filter(should=text_conditions, must=filter_conditions) if all_conditions else None
            
            # Use scroll to get results (since we're not doing vector search)
            scroll_result = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            # Calculate keyword relevance scores
            results = []
            for point in scroll_result[0]:  # scroll_result is (points, next_page_offset)
                text = point.payload.get('text', '').lower()
                
                # Simple TF-IDF-like scoring
                score = 0.0
                for term in query_terms:
                    if term in text:
                        score += text.count(term) / len(text.split())
                
                results.append({
                    'id': str(point.id),
                    'score': score,
                    'payload': point.payload
                })
            
            # Sort by keyword relevance
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []  # Fallback to empty results
    
    def _combine_search_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """Combine semantic and keyword search results with weighted scoring."""
        combined_scores = {}
        
        # Normalize semantic scores
        if semantic_results:
            max_semantic_score = max(r['score'] for r in semantic_results)
            for result in semantic_results:
                doc_id = result['id']
                normalized_score = result['score'] / max_semantic_score if max_semantic_score > 0 else 0
                combined_scores[doc_id] = {
                    'semantic_score': normalized_score * semantic_weight,
                    'keyword_score': 0,
                    'payload': result['payload']
                }
        
        # Normalize keyword scores
        if keyword_results:
            max_keyword_score = max(r['score'] for r in keyword_results) if keyword_results else 1
            for result in keyword_results:
                doc_id = result['id']
                normalized_score = result['score'] / max_keyword_score if max_keyword_score > 0 else 0
                
                if doc_id in combined_scores:
                    combined_scores[doc_id]['keyword_score'] = normalized_score * keyword_weight
                else:
                    combined_scores[doc_id] = {
                        'semantic_score': 0,
                        'keyword_score': normalized_score * keyword_weight,
                        'payload': result['payload']
                    }
        
        # Calculate final scores and sort
        final_results = []
        for doc_id, scores in combined_scores.items():
            final_score = scores['semantic_score'] + scores['keyword_score']
            final_results.append({
                'id': doc_id,
                'score': final_score,
                'semantic_score': scores['semantic_score'],
                'keyword_score': scores['keyword_score'],
                'payload': scores['payload']
            })
        
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results
    
    async def delete_document(self, document_id: str) -> int:
        """Delete all vectors associated with a document."""
        try:
            # Find all points for this document
            scroll_result = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=10000,  # Adjust based on expected document size
                with_payload=False,
                with_vectors=False
            )
            
            points_to_delete = [point.id for point in scroll_result[0]]
            
            if points_to_delete:
                # Delete points
                await self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=points_to_delete
                    )
                )
                
                logger.info(f"Deleted {len(points_to_delete)} vectors for document {document_id}")
            
            # Remove from cache
            await self._remove_document_cache(document_id)
            
            return len(points_to_delete)
            
        except Exception as e:
            logger.error(f"Failed to delete document vectors: {e}")
            raise VectorSearchError(f"Vector deletion failed: {e}")
    
    async def get_document_vectors(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all vectors for a specific document."""
        try:
            vectors = []
            offset = None
            
            while True:
                scroll_result = await self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id)
                            )
                        ]
                    ),
                    offset=offset,
                    limit=100,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, next_offset = scroll_result
                
                for point in points:
                    vectors.append({
                        'id': str(point.id),
                        'payload': point.payload
                    })
                
                if next_offset is None:
                    break
                offset = next_offset
            
            return vectors
            
        except Exception as e:
            logger.error(f"Failed to get document vectors: {e}")
            raise VectorSearchError(f"Vector retrieval failed: {e}")
    
    async def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        try:
            if not texts:
                return np.array([])
            
            # Check cache first
            cached_embeddings = await self._get_cached_embeddings(texts)
            uncached_texts = [text for text, embedding in zip(texts, cached_embeddings) if embedding is None]
            
            if uncached_texts:
                # Generate embeddings for uncached texts
                loop = asyncio.get_event_loop()
                new_embeddings = await loop.run_in_executor(
                    None,
                    lambda: self.embedding_model.encode(
                        uncached_texts,
                        batch_size=settings.embedding_batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                )
                
                # Cache new embeddings
                await self._cache_embeddings(uncached_texts, new_embeddings)
                
                # Merge cached and new embeddings
                result_embeddings = []
                new_idx = 0
                for embedding in cached_embeddings:
                    if embedding is None:
                        result_embeddings.append(new_embeddings[new_idx])
                        new_idx += 1
                    else:
                        result_embeddings.append(embedding)
                
                return np.array(result_embeddings)
            else:
                return np.array([emb for emb in cached_embeddings if emb is not None])
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}")
    
    async def _get_cached_embeddings(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Get cached embeddings for texts."""
        if not self.redis_client:
            return [None] * len(texts)
        
        try:
            cached_embeddings = []
            for text in texts:
                cache_key = f"embedding:{hash(text)}"
                cached = await self.redis_client.get(cache_key)
                if cached:
                    embedding = np.frombuffer(cached, dtype=np.float32)
                    cached_embeddings.append(embedding)
                else:
                    cached_embeddings.append(None)
            
            return cached_embeddings
            
        except Exception as e:
            logger.warning(f"Failed to get cached embeddings: {e}")
            return [None] * len(texts)
    
    async def _cache_embeddings(self, texts: List[str], embeddings: np.ndarray):
        """Cache embeddings for texts."""
        if not self.redis_client:
            return
        
        try:
            for text, embedding in zip(texts, embeddings):
                cache_key = f"embedding:{hash(text)}"
                await self.redis_client.setex(
                    cache_key,
                    settings.cache_ttl_seconds,
                    embedding.astype(np.float32).tobytes()
                )
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")
    
    async def _cache_document_info(self, document_id: str, vector_count: int):
        """Cache document information."""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"doc_info:{document_id}"
            doc_info = {
                'vector_count': vector_count,
                'created_at': time.time()
            }
            await self.redis_client.setex(cache_key, settings.cache_ttl_seconds, str(doc_info))
        except Exception as e:
            logger.warning(f"Failed to cache document info: {e}")
    
    async def _remove_document_cache(self, document_id: str):
        """Remove document from cache."""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"doc_info:{document_id}"
            await self.redis_client.delete(cache_key)
        except Exception as e:
            logger.warning(f"Failed to remove document cache: {e}")
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to ensure JSON serialization."""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool, list, dict)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized
    
    async def health_check(self) -> bool:
        """Check if vector service is healthy."""
        try:
            # Check Qdrant connection
            info = await self.client.get_collection(self.collection_name)
            
            # Check embedding model
            test_embedding = await self._generate_embeddings(["health check"])
            
            return len(test_embedding) > 0 and info is not None
            
        except Exception as e:
            logger.error(f"Vector service health check failed: {e}")
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector collection."""
        try:
            info = await self.client.get_collection(self.collection_name)
            return {
                'name': info.config.params.vectors.size,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'points_count': info.points_count,
                'segments_count': info.segments_count,
                'disk_data_size': info.config.optimizer_config.default_segment_number,
                'ram_data_size': info.config.optimizer_config.memmap_threshold
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    async def optimize_collection(self):
        """Optimize the vector collection for better performance."""
        try:
            await self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000,
                    memmap_threshold=1000000
                )
            )
            logger.info("Collection optimization completed")
        except Exception as e:
            logger.error(f"Collection optimization failed: {e}")
