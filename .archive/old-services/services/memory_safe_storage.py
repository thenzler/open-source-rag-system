#!/usr/bin/env python3
"""
Memory Safe Storage Service
Provides memory-safe document storage with capacity limits
Simple fallback without database dependencies
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)

class MemorySafeStorage:
    """
    Memory-safe storage with capacity limits and warnings
    Prevents memory crashes by enforcing document/chunk limits
    """
    
    def __init__(self, max_documents: int = 1000, max_chunks: int = 10000):
        self.max_documents = max_documents
        self.max_chunks = max_chunks
        self.warning_threshold = 0.8  # Warn at 80% capacity
        
        # Storage
        self.documents = []
        self.chunks = []
        self.embeddings = []
        self.doc_id_counter = 1
        
        logger.info(f"Memory-safe storage initialized - capacity: {max_documents} documents, {max_chunks} chunks")
    
    def add_document(self, filename: str, chunks: List[str], 
                    embeddings: Optional[List[np.ndarray]] = None,
                    metadata: Dict[str, Any] = None) -> int:
        """
        Add document with capacity checking
        
        Returns:
            Document ID
        Raises:
            MemoryError: If capacity limits exceeded
        """
        
        # Check document capacity
        if len(self.documents) >= self.max_documents:
            raise MemoryError(
                f"Document storage limit reached ({self.max_documents} documents). "
                "Cannot add more documents."
            )
        
        # Check chunk capacity
        if len(self.chunks) + len(chunks) > self.max_chunks:
            raise MemoryError(
                f"Chunk storage limit would be exceeded. "
                f"Current: {len(self.chunks)}, Adding: {len(chunks)}, Limit: {self.max_chunks}"
            )
        
        # Capacity warnings
        doc_usage = len(self.documents) / self.max_documents
        chunk_usage = (len(self.chunks) + len(chunks)) / self.max_chunks
        
        if doc_usage >= self.warning_threshold:
            warnings.warn(
                f"Document storage at {len(self.documents)}/{self.max_documents} "
                f"({int(doc_usage * 100)}% of capacity)",
                ResourceWarning
            )
        
        if chunk_usage >= self.warning_threshold:
            warnings.warn(
                f"Chunk storage at {len(self.chunks) + len(chunks)}/{self.max_chunks} "
                f"({int(chunk_usage * 100)}% of capacity)",
                ResourceWarning
            )
        
        # Add document
        doc_id = self.doc_id_counter
        self.doc_id_counter += 1
        
        document = {
            'id': doc_id,
            'filename': filename,
            'upload_timestamp': datetime.now(),
            'chunk_count': len(chunks),
            'metadata': metadata or {}
        }
        self.documents.append(document)
        
        # Add chunks
        for i, chunk in enumerate(chunks):
            self.chunks.append({
                'document_id': doc_id,
                'chunk_index': i,
                'text': chunk,
                'filename': filename
            })
            
            # Add embedding if provided
            if embeddings and i < len(embeddings):
                embedding = embeddings[i]
                if embedding is not None:
                    logger.info(f"Adding embedding {i}: type={type(embedding)}, shape={getattr(embedding, 'shape', 'no shape')}")
                    self.embeddings.append(embedding)
                else:
                    logger.warning(f"Embedding {i} is None")
                    self.embeddings.append(None)
            else:
                logger.warning(f"No embedding provided for chunk {i} (embeddings len: {len(embeddings) if embeddings else 0})")
                self.embeddings.append(None)
        
        logger.info(f"Added document {filename} (ID: {doc_id}) with {len(chunks)} chunks")
        return doc_id
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
              min_similarity: float = 0.4) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity"""
        
        logger.info(f"Search called: embeddings count={len(self.embeddings)}, chunks count={len(self.chunks)}")
        
        if not self.embeddings:
            logger.warning("No embeddings available for search")
            return []
        
        # Count non-None embeddings
        valid_embeddings = sum(1 for emb in self.embeddings if emb is not None)
        logger.info(f"Valid embeddings: {valid_embeddings}/{len(self.embeddings)}")
        
        # Calculate similarities
        similarities = []
        max_similarity = 0.0
        
        for i, doc_embedding in enumerate(self.embeddings):
            if doc_embedding is not None:
                # Cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                max_similarity = max(max_similarity, similarity)
                
                if similarity >= min_similarity:
                    similarities.append({
                        'index': i,
                        'similarity': float(similarity)
                    })
                elif i < 3:  # Log first few for debugging
                    logger.info(f"Chunk {i}: similarity={similarity:.4f} (below threshold {min_similarity})")
        
        logger.info(f"Search results: found {len(similarities)} chunks above threshold, max similarity: {max_similarity:.4f}")
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get top results
        results = []
        for item in similarities[:top_k]:
            chunk_idx = item['index']
            if chunk_idx < len(self.chunks):
                chunk = self.chunks[chunk_idx]
                results.append({
                    'chunk_id': chunk_idx,
                    'document_id': chunk['document_id'],
                    'text': chunk['text'],
                    'similarity': item['similarity'],
                    'filename': chunk['filename'],
                    'metadata': {}
                })
        
        return results
    
    def get_all_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get paginated list of documents"""
        start = offset
        end = min(offset + limit, len(self.documents))
        return self.documents[start:end]
    
    def delete_document(self, document_id: int) -> bool:
        """Delete document and its chunks"""
        # Remove document
        self.documents = [
            doc for doc in self.documents 
            if doc['id'] != document_id
        ]
        
        # Remove chunks and embeddings
        indices_to_remove = []
        for i, chunk in enumerate(self.chunks):
            if chunk['document_id'] == document_id:
                indices_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for i in reversed(indices_to_remove):
            del self.chunks[i]
            if i < len(self.embeddings):
                del self.embeddings[i]
        
        logger.info(f"Deleted document {document_id}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        # Calculate memory usage estimate
        memory_usage_mb = 0
        if self.embeddings:
            # Rough estimate: 384 floats * 4 bytes per float
            embeddings_size = len(self.embeddings) * 384 * 4 / 1024 / 1024  # MB
            chunks_size = sum(len(chunk['text']) for chunk in self.chunks) / 1024 / 1024
            memory_usage_mb = embeddings_size + chunks_size
        
        doc_usage = len(self.documents) / self.max_documents
        chunk_usage = len(self.chunks) / self.max_chunks
        
        return {
            'storage_mode': 'memory_safe',
            'documents': len(self.documents),
            'chunks': len(self.chunks),
            'embeddings': len(self.embeddings),
            'max_documents': self.max_documents,
            'max_chunks': self.max_chunks,
            'memory_usage_mb': round(memory_usage_mb, 2),
            'capacity_documents': f"{len(self.documents)}/{self.max_documents}",
            'capacity_chunks': f"{len(self.chunks)}/{self.max_chunks}",
            'usage_percentage_docs': int(doc_usage * 100),
            'usage_percentage_chunks': int(chunk_usage * 100),
            'is_near_limit': doc_usage >= self.warning_threshold or chunk_usage >= self.warning_threshold,
            'warning_threshold': int(self.warning_threshold * 100)
        }

# Global instance
_memory_safe_storage: Optional[MemorySafeStorage] = None

def get_memory_safe_storage() -> MemorySafeStorage:
    """Get or create memory-safe storage instance"""
    global _memory_safe_storage
    if _memory_safe_storage is None:
        _memory_safe_storage = MemorySafeStorage()
    return _memory_safe_storage