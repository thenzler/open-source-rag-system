# Vector Search Service with FAISS optimization
import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict
import pickle
import os
import logging
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class LegacySearchResult:
    chunk_id: int
    score: float
    content: str
    source_document: str
    metadata: Dict


class FAISSVectorSearch:
    """Optimized vector search using FAISS for fast similarity search"""
    
    def __init__(self, dimension: int = 384, index_type: str = "auto"):
        """
        Initialize FAISS vector search
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
            index_type: Type of index - "auto", "flat", "ivf", "hnsw"
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.chunk_mapping = {}  # Maps FAISS index position to chunk metadata
        self.chunk_texts = {}    # Maps chunk_id to actual text
        self.document_mapping = {}  # Maps chunk_id to document info
        self.index_size = 0
        self._lock = Lock()  # Thread safety for index operations
        
    def _create_index(self, num_vectors: int = 0) -> faiss.Index:
        """Create appropriate FAISS index based on data size and type"""
        if self.index_type == "auto":
            # Auto-select best index type based on dataset size
            if num_vectors < 1000:
                # For small datasets, use exact search
                logger.info("Using Flat index for small dataset")
                return faiss.IndexFlatIP(self.dimension)
            elif num_vectors < 100000:
                # For medium datasets, use IVF
                logger.info("Using IVF index for medium dataset")
                nlist = min(int(np.sqrt(num_vectors)) * 4, 1024)
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                return index
            else:
                # For large datasets, use HNSW
                logger.info("Using HNSW index for large dataset")
                index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = 40
                return index
        elif self.index_type == "flat":
            return faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            nlist = min(int(np.sqrt(max(num_vectors, 100))) * 4, 1024)
            quantizer = faiss.IndexFlatIP(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 40
            return index
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def build_index(self, embeddings: np.ndarray, chunk_ids: List[int], 
                    chunk_texts: List[str], document_info: List[Dict]) -> None:
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Numpy array of embeddings (n_samples, dimension)
            chunk_ids: List of chunk IDs
            chunk_texts: List of chunk text content
            document_info: List of document metadata for each chunk
        """
        with self._lock:
            if len(embeddings) == 0:
                logger.warning("No embeddings provided to build index")
                return
            
            # Ensure embeddings are float32 and normalized
            embeddings = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings)
            
            # Create appropriate index
            self.index = self._create_index(len(embeddings))
            
            # Train index if necessary (for IVF indices)
            if hasattr(self.index, 'train') and not self.index.is_trained:
                logger.info(f"Training index with {len(embeddings)} vectors")
                self.index.train(embeddings)
            
            # Add vectors to index
            self.index.add(embeddings)
            
            # Update mappings
            for i, (chunk_id, chunk_text, doc_info) in enumerate(zip(chunk_ids, chunk_texts, document_info)):
                self.chunk_mapping[i] = chunk_id
                self.chunk_texts[chunk_id] = chunk_text
                self.document_mapping[chunk_id] = doc_info
            
            self.index_size = len(embeddings)
            logger.info(f"Built FAISS index with {self.index_size} vectors")
    
    def add_embeddings(self, new_embeddings: np.ndarray, chunk_ids: List[int],
                      chunk_texts: List[str], document_info: List[Dict]) -> None:
        """Add new embeddings to existing index"""
        with self._lock:
            if self.index is None:
                # First time - build new index
                self.build_index(new_embeddings, chunk_ids, chunk_texts, document_info)
                return
            
            # Normalize new embeddings
            new_embeddings = new_embeddings.astype(np.float32)
            faiss.normalize_L2(new_embeddings)
            
            # Check if we need to rebuild index (for better performance)
            new_total = self.index_size + len(new_embeddings)
            if self._should_rebuild_index(new_total):
                logger.info("Rebuilding index for better performance")
                # Get all existing embeddings
                all_embeddings = self.index.reconstruct_n(0, self.index_size)
                all_embeddings = np.vstack([all_embeddings, new_embeddings])
                
                # Rebuild with all data
                all_chunk_ids = list(self.chunk_mapping.values()) + chunk_ids
                all_chunk_texts = [self.chunk_texts[cid] for cid in self.chunk_mapping.values()] + chunk_texts
                all_doc_info = [self.document_mapping[cid] for cid in self.chunk_mapping.values()] + document_info
                
                self.build_index(all_embeddings, all_chunk_ids, all_chunk_texts, all_doc_info)
            else:
                # Just add to existing index
                start_idx = self.index.ntotal
                self.index.add(new_embeddings)
                
                # Update mappings
                for i, (chunk_id, chunk_text, doc_info) in enumerate(zip(chunk_ids, chunk_texts, document_info)):
                    self.chunk_mapping[start_idx + i] = chunk_id
                    self.chunk_texts[chunk_id] = chunk_text
                    self.document_mapping[chunk_id] = doc_info
                
                self.index_size = self.index.ntotal
                logger.info(f"Added {len(new_embeddings)} vectors, total: {self.index_size}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, 
              min_score: float = 0.0) -> List[LegacySearchResult]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of LegacySearchResult objects sorted by score
        """
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("Index is empty, returning no results")
                return []
            
            # Prepare query
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Adjust k if necessary
            k = min(k, self.index.ntotal)
            
            # Set search parameters for better quality/speed tradeoff
            if hasattr(self.index, 'nprobe'):
                # For IVF indices, search more cells for better recall
                self.index.nprobe = min(self.index.nlist, 10)
            elif hasattr(self.index, 'hnsw'):
                # For HNSW, increase ef for better recall
                self.index.hnsw.efSearch = max(k * 2, 40)
            
            # Perform search
            scores, indices = self.index.search(query_embedding, k)
            
            # Convert results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Invalid result
                    continue
                    
                if score < min_score:  # Below threshold
                    continue
                
                chunk_id = self.chunk_mapping.get(idx)
                if chunk_id is None:
                    logger.warning(f"No mapping found for index {idx}")
                    continue
                
                result = LegacySearchResult(
                    chunk_id=chunk_id,
                    score=float(score),
                    content=self.chunk_texts.get(chunk_id, ""),
                    source_document=self.document_mapping.get(chunk_id, {}).get("filename", "Unknown"),
                    metadata=self.document_mapping.get(chunk_id, {})
                )
                results.append(result)
            
            return results
    
    def batch_search(self, query_embeddings: np.ndarray, k: int = 5,
                    min_score: float = 0.0) -> List[List[LegacySearchResult]]:
        """Batch search for multiple queries simultaneously"""
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return [[] for _ in range(len(query_embeddings))]
            
            # Prepare queries
            query_embeddings = query_embeddings.astype(np.float32)
            faiss.normalize_L2(query_embeddings)
            
            # Adjust k
            k = min(k, self.index.ntotal)
            
            # Configure search parameters
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = min(self.index.nlist, 10)
            elif hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = max(k * 2, 40)
            
            # Batch search
            scores_batch, indices_batch = self.index.search(query_embeddings, k)
            
            # Convert results for each query
            all_results = []
            for scores, indices in zip(scores_batch, indices_batch):
                results = []
                for score, idx in zip(scores, indices):
                    if idx == -1 or score < min_score:
                        continue
                    
                    chunk_id = self.chunk_mapping.get(idx)
                    if chunk_id is None:
                        continue
                    
                    result = LegacySearchResult(
                        chunk_id=chunk_id,
                        score=float(score),
                        content=self.chunk_texts.get(chunk_id, ""),
                        source_document=self.document_mapping.get(chunk_id, {}).get("filename", "Unknown"),
                        metadata=self.document_mapping.get(chunk_id, {})
                    )
                    results.append(result)
                
                all_results.append(results)
            
            return all_results
    
    def remove_document(self, document_id: str) -> int:
        """Remove all chunks belonging to a document"""
        with self._lock:
            # Find chunks to remove
            chunks_to_remove = []
            for chunk_id, doc_info in self.document_mapping.items():
                if doc_info.get("document_id") == document_id:
                    chunks_to_remove.append(chunk_id)
            
            if not chunks_to_remove:
                return 0
            
            # For now, we need to rebuild the entire index
            # (FAISS doesn't support efficient deletion)
            # In production, you might want to mark as deleted and rebuild periodically
            
            # Get all chunks except the ones to remove
            remaining_chunks = []
            remaining_embeddings = []
            
            for idx, chunk_id in self.chunk_mapping.items():
                if chunk_id not in chunks_to_remove:
                    remaining_chunks.append({
                        "chunk_id": chunk_id,
                        "chunk_text": self.chunk_texts[chunk_id],
                        "doc_info": self.document_mapping[chunk_id],
                        "embedding": self.index.reconstruct(int(idx))
                    })
            
            # Clear and rebuild
            self.chunk_mapping.clear()
            self.chunk_texts.clear()
            self.document_mapping.clear()
            
            if remaining_chunks:
                embeddings = np.array([c["embedding"] for c in remaining_chunks])
                chunk_ids = [c["chunk_id"] for c in remaining_chunks]
                chunk_texts = [c["chunk_text"] for c in remaining_chunks]
                doc_infos = [c["doc_info"] for c in remaining_chunks]
                
                self.build_index(embeddings, chunk_ids, chunk_texts, doc_infos)
            else:
                self.index = None
                self.index_size = 0
            
            return len(chunks_to_remove)
    
    def save_index(self, index_path: str, metadata_path: str) -> None:
        """Save index and metadata to disk"""
        with self._lock:
            if self.index is None:
                logger.warning("No index to save")
                return
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata = {
                "dimension": self.dimension,
                "index_type": self.index_type,
                "index_size": self.index_size,
                "chunk_mapping": self.chunk_mapping,
                "chunk_texts": self.chunk_texts,
                "document_mapping": self.document_mapping
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved index with {self.index_size} vectors to {index_path}")
    
    def load_index(self, index_path: str, metadata_path: str) -> None:
        """Load index and metadata from disk"""
        with self._lock:
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                raise FileNotFoundError("Index files not found")
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.dimension = metadata["dimension"]
            self.index_type = metadata["index_type"]
            self.index_size = metadata["index_size"]
            self.chunk_mapping = metadata["chunk_mapping"]
            self.chunk_texts = metadata["chunk_texts"]
            self.document_mapping = metadata["document_mapping"]
            
            logger.info(f"Loaded index with {self.index_size} vectors from {index_path}")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        with self._lock:
            if self.index is None:
                return {"status": "empty", "total_vectors": 0}
            
            stats = {
                "status": "ready",
                "total_vectors": self.index.ntotal,
                "dimension": self.dimension,
                "index_type": self.index.__class__.__name__,
                "documents": len(set(doc["document_id"] for doc in self.document_mapping.values())),
                "chunks": len(self.chunk_mapping)
            }
            
            # Add index-specific stats
            if hasattr(self.index, 'nlist'):
                stats["nlist"] = self.index.nlist
            if hasattr(self.index, 'nprobe'):
                stats["nprobe"] = self.index.nprobe
                
            return stats
    
    def _should_rebuild_index(self, new_total: int) -> bool:
        """Determine if index should be rebuilt for better performance"""
        if self.index_type != "auto":
            return False
        
        # Check if we're crossing a threshold that warrants different index type
        current_type = self.index.__class__.__name__
        
        if current_type == "IndexFlatIP" and new_total > 1000:
            return True  # Switch to IVF
        elif current_type == "IndexIVFFlat" and new_total > 100000:
            return True  # Switch to HNSW
        
        return False
    
    def optimize_index(self) -> None:
        """Optimize index for better search performance"""
        with self._lock:
            if self.index is None:
                return
            
            # For IVF indices, we can retrain with current data
            if hasattr(self.index, 'train') and hasattr(self.index, 'is_trained'):
                logger.info("Retraining IVF index for better clustering")
                # Get all vectors
                vectors = self.index.reconstruct_n(0, self.index.ntotal)
                
                # Create new index and train
                new_index = self._create_index(self.index.ntotal)
                if hasattr(new_index, 'train'):
                    new_index.train(vectors)
                new_index.add(vectors)
                
                self.index = new_index
                logger.info("Index optimization complete")


# Compatibility wrapper for existing code
class OptimizedVectorStore:
    """Wrapper to integrate FAISS search with existing codebase"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.faiss_search = FAISSVectorSearch(dimension=self.dimension)
        self.chunk_counter = 0
        
    def add_documents(self, texts: List[str], metadatas: List[Dict]) -> None:
        """Add documents to the vector store"""
        if not texts:
            return
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings)
        
        # Create chunk IDs
        chunk_ids = list(range(self.chunk_counter, self.chunk_counter + len(texts)))
        self.chunk_counter += len(texts)
        
        # Add to FAISS index
        self.faiss_search.add_embeddings(embeddings, chunk_ids, texts, metadatas)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0]
        
        # Search
        results = self.faiss_search.search(query_embedding, k=k, min_score=0.0)
        
        # Format results for compatibility
        formatted_results = []
        for result in results:
            formatted_results.append((
                result.content,
                result.score,
                result.metadata
            ))
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        return self.faiss_search.get_stats()
    
    def save(self, directory: str) -> None:
        """Save vector store to disk"""
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "faiss.index")
        metadata_path = os.path.join(directory, "faiss_metadata.pkl")
        self.faiss_search.save_index(index_path, metadata_path)
    
    def load(self, directory: str) -> None:
        """Load vector store from disk"""
        index_path = os.path.join(directory, "faiss.index")
        metadata_path = os.path.join(directory, "faiss_metadata.pkl")
        self.faiss_search.load_index(index_path, metadata_path)