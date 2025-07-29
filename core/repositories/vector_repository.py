"""
Production Vector Search Repository
Optimized implementation with FAISS support and caching
"""

import gzip
import logging
import pickle
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .interfaces import IVectorSearchRepository
from .models import Embedding

try:
    from config.config import config
except ImportError:
    config = None

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss

    FAISS_AVAILABLE = True
    logger.info("FAISS available for high-performance vector search")
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("FAISS not available, using numpy fallback")


class VectorIndex:
    """Abstract vector index interface"""

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def remove_vectors(self, ids: List[int]):
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        raise NotImplementedError


class FAISSIndex(VectorIndex):
    """FAISS-based vector index for high performance"""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.id_mapping = {}  # FAISS index position -> embedding ID
        self.reverse_mapping = {}  # embedding ID -> FAISS index position
        self._next_position = 0
        self._build_index()

    def _build_index(self):
        """Build appropriate FAISS index based on expected size"""
        if self.dimension <= 0:
            self.dimension = 384  # Default for all-MiniLM-L6-v2 model

        # Choose index type based on expected data size
        # For RAG systems, we typically have 1k-100k vectors
        if hasattr(config, "EXPECTED_VECTOR_COUNT"):
            expected_count = config.EXPECTED_VECTOR_COUNT
        else:
            expected_count = 10000  # Conservative estimate

        if expected_count < 1000:
            # Small dataset: use flat index for exact search
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product
            logger.info(f"Created FAISS Flat index for {expected_count} vectors")
        elif expected_count < 50000:
            # Medium dataset: use IVF for faster search
            nlist = min(100, expected_count // 50)  # ~50 vectors per cluster
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            logger.info(f"Created FAISS IVF index with {nlist} clusters")
        else:
            # Large dataset: use HNSW for very fast approximate search
            self.index = faiss.IndexHNSWFlat(self.dimension)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 128
            logger.info("Created FAISS HNSW index for large dataset")

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors to the index"""
        if vectors.shape[0] == 0:
            return

        # Normalize vectors for cosine similarity (inner product on unit vectors)
        normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        # Train index if needed (for IVF)
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            # For small datasets, create a simple flat index instead
            if normalized_vectors.shape[0] < 100:
                logger.info("Small dataset detected, using flat index")
                self.index = faiss.IndexFlatIP(self.dimension)
                # Re-add any existing vectors
                if self._next_position > 0:
                    logger.warning(
                        "Switching to flat index, existing vectors will be lost"
                    )
                    self._next_position = 0
                    self.id_mapping.clear()
                    self.reverse_mapping.clear()
            elif normalized_vectors.shape[0] >= self.index.nlist * 10:
                self.index.train(normalized_vectors)
                logger.info("FAISS index trained successfully")
            else:
                logger.warning(
                    f"Not enough vectors for training IVF index ({normalized_vectors.shape[0]} < {self.index.nlist * 10}), switching to flat index"
                )
                self.index = faiss.IndexFlatIP(self.dimension)
                # Re-add any existing vectors
                if self._next_position > 0:
                    logger.warning(
                        "Switching to flat index, existing vectors will be lost"
                    )
                    self._next_position = 0
                    self.id_mapping.clear()
                    self.reverse_mapping.clear()

        # Add to index
        start_pos = self._next_position
        self.index.add(normalized_vectors.astype(np.float32))

        # Update mappings
        for i, embedding_id in enumerate(ids):
            position = start_pos + i
            self.id_mapping[position] = embedding_id
            self.reverse_mapping[embedding_id] = position

        self._next_position += len(ids)

        logger.info(
            f"Added {len(ids)} vectors to FAISS index (total: {self._next_position})"
        )

    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors"""
        if self.index.ntotal == 0:
            return np.array([]), np.array([])

        # Normalize query vector
        query_normalized = query_vector / np.linalg.norm(query_vector)
        query_normalized = query_normalized.reshape(1, -1).astype(np.float32)

        # Search
        similarities, indices = self.index.search(
            query_normalized, min(k, self.index.ntotal)
        )

        # Convert FAISS indices to embedding IDs
        embedding_ids = []
        valid_similarities = []

        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx >= 0 and idx in self.id_mapping:  # Valid result
                embedding_ids.append(self.id_mapping[idx])
                valid_similarities.append(sim)

        return np.array(valid_similarities), np.array(embedding_ids)

    def remove_vectors(self, ids: List[int]):
        """Remove vectors from index (requires rebuild for most FAISS indices)"""
        # For simplicity, we mark as removed and rebuild if needed
        for embedding_id in ids:
            if embedding_id in self.reverse_mapping:
                position = self.reverse_mapping[embedding_id]
                del self.id_mapping[position]
                del self.reverse_mapping[embedding_id]

        logger.info(f"Marked {len(ids)} vectors for removal from FAISS index")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "index_type": "FAISS",
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "is_trained": getattr(self.index, "is_trained", True),
            "memory_usage_mb": (
                self.index.ntotal * self.dimension * 4 / (1024 * 1024)
                if self.index
                else 0
            ),
        }


class NumpyIndex(VectorIndex):
    """Numpy-based fallback vector index"""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.ids = []
        self.id_to_position = {}

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors to the index"""
        if vectors.shape[0] == 0:
            return

        # Normalize vectors for cosine similarity
        normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        # Add to storage
        start_pos = len(self.vectors)
        self.vectors = np.vstack([self.vectors, normalized_vectors.astype(np.float32)])

        # Update mappings
        for i, embedding_id in enumerate(ids):
            position = start_pos + i
            self.ids.append(embedding_id)
            self.id_to_position[embedding_id] = position

        logger.info(
            f"Added {len(ids)} vectors to Numpy index (total: {len(self.vectors)})"
        )

    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors using cosine similarity"""
        if len(self.vectors) == 0:
            return np.array([]), np.array([])

        # Normalize query vector
        query_normalized = query_vector / np.linalg.norm(query_vector)

        # Compute cosine similarities
        similarities = np.dot(self.vectors, query_normalized)

        # Get top-k results
        k = min(k, len(similarities))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        top_similarities = similarities[top_indices]
        top_ids = np.array([self.ids[i] for i in top_indices])

        return top_similarities, top_ids

    def remove_vectors(self, ids: List[int]):
        """Remove vectors from index"""
        positions_to_remove = []
        for embedding_id in ids:
            if embedding_id in self.id_to_position:
                positions_to_remove.append(self.id_to_position[embedding_id])

        if not positions_to_remove:
            return

        # Remove from arrays (expensive operation)
        mask = np.ones(len(self.vectors), dtype=bool)
        mask[positions_to_remove] = False

        self.vectors = self.vectors[mask]

        # Rebuild mappings
        new_ids = []
        new_id_to_position = {}

        for i, embedding_id in enumerate(self.ids):
            if embedding_id not in ids:
                new_ids.append(embedding_id)
                new_id_to_position[embedding_id] = len(new_ids) - 1

        self.ids = new_ids
        self.id_to_position = new_id_to_position

        logger.info(f"Removed {len(ids)} vectors from Numpy index")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "index_type": "Numpy",
            "total_vectors": len(self.vectors),
            "dimension": self.dimension,
            "memory_usage_mb": self.vectors.nbytes / (1024 * 1024),
        }


class ProductionVectorRepository(IVectorSearchRepository):
    """Production-ready vector search repository with caching"""

    def __init__(self, cache_size: int = 1000):
        self.index = None
        self.cache_size = cache_size
        self.search_cache = {}  # Simple LRU-like cache
        self.cache_access_times = {}
        self.dimension = 384  # Default for all-MiniLM-L6-v2 model
        self._lock = threading.RLock()
        self._index_loaded = False

        # Initialize index
        self._initialize_index()

        # Initialize loading flag
        self._loading_embeddings = False

    def _initialize_index(self):
        """Initialize the appropriate vector index"""
        if FAISS_AVAILABLE:
            self.index = FAISSIndex(self.dimension)
        else:
            self.index = NumpyIndex(self.dimension)

        logger.info(f"Initialized {self.index.__class__.__name__} for vector search")

    @lru_cache(maxsize=100)
    def _hash_query_vector(self, query_vector_bytes: bytes) -> str:
        """Create a hash for query vector caching"""
        import hashlib

        return hashlib.md5(query_vector_bytes).hexdigest()

    async def build_index(self, embeddings: List[Embedding]) -> bool:
        """Build or rebuild the search index"""
        try:
            with self._lock:
                # Extract vectors and IDs
                vectors = []
                ids = []

                for embedding in embeddings:
                    if embedding.embedding_vector and embedding.id:
                        vectors.append(embedding.embedding_vector)
                        ids.append(embedding.id)

                if not vectors:
                    logger.warning("No valid embeddings to build index")
                    return False

                # Convert to numpy array
                vector_array = np.array(vectors, dtype=np.float32)

                # Update dimension if needed
                if vector_array.shape[1] != self.dimension:
                    self.dimension = vector_array.shape[1]
                    self._initialize_index()

                # Build index
                self.index.add_vectors(vector_array, ids)

                # Clear cache after rebuild
                self.search_cache.clear()
                self.cache_access_times.clear()

                logger.info(f"Built vector index with {len(vectors)} embeddings")
                return True

        except Exception as e:
            logger.error(f"Error building vector index: {e}")
            return False

    async def add_to_index(self, embeddings: List[Embedding]) -> bool:
        """Add new embeddings to the index"""
        try:
            if not embeddings:
                return True

            with self._lock:
                vectors = []
                ids = []

                for embedding in embeddings:
                    if embedding.embedding_vector and embedding.id:
                        vectors.append(embedding.embedding_vector)
                        ids.append(embedding.id)

                if vectors:
                    vector_array = np.array(vectors, dtype=np.float32)
                    self.index.add_vectors(vector_array, ids)

                    # Clear cache to ensure fresh results
                    self.search_cache.clear()
                    self.cache_access_times.clear()

                logger.info(f"Added {len(vectors)} new embeddings to index")
                return True

        except Exception as e:
            logger.error(f"Error adding embeddings to index: {e}")
            return False

    async def remove_from_index(self, embedding_ids: List[int]) -> bool:
        """Remove embeddings from the index"""
        try:
            with self._lock:
                self.index.remove_vectors(embedding_ids)

                # Clear cache
                self.search_cache.clear()
                self.cache_access_times.clear()

                logger.info(f"Removed {len(embedding_ids)} embeddings from index")
                return True

        except Exception as e:
            logger.error(f"Error removing embeddings from index: {e}")
            return False

    async def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[int, float]]:
        """Find similar embeddings with caching"""
        try:
            # Convert to numpy array
            query_array = np.array(query_vector, dtype=np.float32)

            # Check cache
            cache_key = self._hash_query_vector(query_array.tobytes()) + f"_k{top_k}"
            current_time = time.time()

            if cache_key in self.search_cache:
                # Update access time and return cached result
                self.cache_access_times[cache_key] = current_time
                logger.debug("Vector search cache hit")
                return self.search_cache[cache_key]

            # Perform search
            with self._lock:
                similarities, embedding_ids = self.index.search(query_array, top_k)

            # Convert to result format
            results = [
                (int(id_), float(sim)) for sim, id_ in zip(similarities, embedding_ids)
            ]

            # Cache result
            self._cache_result(cache_key, results, current_time)

            logger.debug(f"Vector search found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    def _cache_result(
        self, cache_key: str, result: List[Tuple[int, float]], access_time: float
    ):
        """Cache search result with LRU eviction"""
        # Simple LRU eviction
        if len(self.search_cache) >= self.cache_size:
            # Find least recently used item
            oldest_key = min(
                self.cache_access_times.keys(), key=lambda k: self.cache_access_times[k]
            )
            del self.search_cache[oldest_key]
            del self.cache_access_times[oldest_key]

        self.search_cache[cache_key] = result
        self.cache_access_times[cache_key] = access_time

    async def get_index_statistics(self) -> Dict[str, Any]:
        """Get search index statistics"""
        try:
            with self._lock:
                stats = self.index.get_stats()
                stats.update(
                    {
                        "cache_size": len(self.search_cache),
                        "cache_hit_rate": "N/A",  # Could be calculated with counters
                        "last_updated": time.time(),
                    }
                )
                return stats
        except Exception as e:
            logger.error(f"Error getting index statistics: {e}")
            return {"error": str(e)}

    async def search_similar_text(
        self, query: str, limit: int = 10, threshold: float = 0.7
    ) -> "QueryResult":
        """Find similar documents by text query (convenience method)"""
        try:
            # Ensure embeddings are loaded before searching
            await self._ensure_embeddings_loaded()

            # Import here to avoid circular imports
            from sentence_transformers import SentenceTransformer

            # Initialize embedding model (cached)
            if not hasattr(self, "_embedding_model"):
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            # Convert text query to embedding
            query_embedding = self._embedding_model.encode([query])[0].tolist()

            # Perform vector search
            vector_results = await self.search_similar(
                query_vector=query_embedding, top_k=limit
            )

            logger.debug(f"Vector search returned {len(vector_results)} results")
            for i, (eid, score) in enumerate(vector_results[:3]):
                logger.debug(f"  Result {i+1}: Embedding {eid}, Score: {score:.6f}")

            # Convert results to QueryResult format
            from ..repositories.base import QueryResult
            from ..repositories.models import DocumentChunk

            # Load actual chunk data from database
            chunks = []
            if vector_results:
                embedding_ids = [int(eid) for eid, score in vector_results]
                logger.debug(f"Loading chunk data for embedding IDs: {embedding_ids}")

                chunk_data = await self._load_chunks_by_embedding_ids(embedding_ids)
                logger.debug(f"Loaded chunk data for {len(chunk_data)} embeddings")

                # Match with similarity scores and apply threshold
                for embedding_id, similarity_score in vector_results:
                    logger.debug(
                        f"Processing embedding {embedding_id}, similarity {similarity_score:.6f}, threshold {threshold}"
                    )

                    if embedding_id in chunk_data:
                        if similarity_score >= threshold:
                            chunk_info = chunk_data[embedding_id]
                            chunk = DocumentChunk(
                                id=chunk_info["chunk_id"],
                                document_id=chunk_info["document_id"],
                                text_content=chunk_info["text"],
                                metadata={
                                    "embedding_id": embedding_id,
                                    "similarity_score": similarity_score,
                                    "chunk_index": chunk_info.get("chunk_index", 0),
                                },
                            )
                            chunks.append(chunk)
                            logger.debug(
                                f"Added chunk {chunk_info['chunk_id']} from doc {chunk_info['document_id']}"
                            )
                        else:
                            logger.debug(
                                f"Embedding {embedding_id} filtered out: similarity {similarity_score:.6f} < threshold {threshold}"
                            )
                    else:
                        logger.warning(
                            f"No chunk data found for embedding {embedding_id}"
                        )

            logger.info(
                f"Text search returning {len(chunks)} chunks (threshold: {threshold})"
            )

            # Return QueryResult with chunks
            search_result = QueryResult(
                items=chunks, total_count=len(chunks), has_more=False
            )

            return search_result

        except Exception as e:
            logger.error(f"Error in text-based vector search: {e}")
            import traceback

            logger.error(f"Stack trace: {traceback.format_exc()}")
            # Return empty result on error
            from ..repositories.base import QueryResult

            return QueryResult(items=[], total_count=0, has_more=False)

    async def is_index_ready(self) -> bool:
        """Check if search index is ready"""
        try:
            with self._lock:
                return self.index.get_stats()["total_vectors"] > 0
        except Exception:
            return False

    async def _load_chunks_by_embedding_ids(
        self, embedding_ids: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """Load chunk data from database by embedding IDs"""
        try:
            import sqlite3

            # Get database path
            db_path = getattr(config, "DATABASE_PATH", None) if config else None
            if not db_path:
                db_path = "data/rag_database.db"

            conn = sqlite3.connect(db_path)

            # Create placeholders for IN clause
            placeholders = ",".join("?" * len(embedding_ids))

            cursor = conn.execute(
                f"""
                SELECT e.id as embedding_id, c.id as chunk_id, c.document_id, 
                       c.text, c.chunk_index
                FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.id
                WHERE e.id IN ({placeholders})
            """,
                embedding_ids,
            )

            chunk_data = {}
            for row in cursor.fetchall():
                embedding_id, chunk_id, document_id, text, chunk_index = row
                chunk_data[embedding_id] = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "text": text,
                    "chunk_index": chunk_index,
                }

            conn.close()
            return chunk_data

        except Exception as e:
            logger.error(f"Error loading chunk data: {e}")
            return {}

    async def _load_existing_embeddings(self):
        """Load existing embeddings from database on initialization"""
        try:
            if self._index_loaded:
                return

            logger.info("Loading existing embeddings into vector index...")

            import gzip
            import pickle
            import sqlite3

            # Get database path
            db_path = getattr(config, "DATABASE_PATH", None) if config else None
            if not db_path:
                db_path = "data/rag_database.db"

            conn = sqlite3.connect(db_path)

            cursor = conn.execute(
                """
                SELECT e.id, e.chunk_id, e.embedding_data, e.embedding_model, e.dimensions,
                       c.document_id, c.text, c.chunk_index
                FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.id
                ORDER BY e.id
            """
            )

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                logger.info("No existing embeddings found in database")
                self._index_loaded = True
                return

            logger.info(f"Found {len(rows)} embeddings to load")

            # Convert to Embedding objects
            from .models import Embedding

            embeddings = []

            for row in rows:
                (
                    embedding_id,
                    chunk_id,
                    embedding_data,
                    model,
                    dims,
                    doc_id,
                    text,
                    chunk_idx,
                ) = row

                try:
                    # Decompress embedding data
                    decompressed = gzip.decompress(embedding_data)
                    embedding_vector = pickle.loads(decompressed)

                    # Create Embedding object
                    embedding = Embedding(
                        id=embedding_id,
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        embedding_vector=embedding_vector,
                        embedding_model=model,
                        vector_dimension=dims,
                    )

                    embeddings.append(embedding)

                except Exception as e:
                    logger.error(f"Error processing embedding {embedding_id}: {e}")
                    continue

            if embeddings:
                # Build the index
                success = await self.build_index(embeddings)
                if success:
                    logger.info(
                        f"✅ Loaded {len(embeddings)} embeddings into vector index"
                    )
                else:
                    logger.error(
                        "❌ Failed to build vector index from existing embeddings"
                    )

            self._index_loaded = True

        except Exception as e:
            logger.error(f"Error loading existing embeddings: {e}")
            self._index_loaded = True  # Mark as attempted to avoid retry loops

    async def _ensure_embeddings_loaded(self):
        """Ensure embeddings are loaded before performing searches"""
        if not self._index_loaded and not self._loading_embeddings:
            self._loading_embeddings = True
            try:
                await self._load_existing_embeddings()
            finally:
                self._loading_embeddings = False
