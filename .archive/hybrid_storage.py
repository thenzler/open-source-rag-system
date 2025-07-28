#!/usr/bin/env python3
"""
Hybrid Storage Service
Seamlessly switches between database and in-memory storage
Ensures system works with or without database
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)

# Try to import database components
try:
    from services.vector_store_db import DatabaseVectorStore, get_db_vector_store
    from database_migration import DatabaseManager
    DB_AVAILABLE = True
except ImportError as e:
    DB_AVAILABLE = False
    DatabaseVectorStore = None
    print(f"Database storage not available - using in-memory fallback: {e}")

# Import configuration
try:
    from config.database_config import get_storage_config
except ImportError:
    # Fallback if config not available
    class MockConfig:
        def __init__(self):
            self.storage_mode = 'memory'
            self.max_memory_documents = 1000
            self.max_memory_chunks = 10000
            self.memory_warning_threshold = 0.8
            self.database_url = ""
        
        def is_database_available(self):
            return False
    
    def get_storage_config():
        return MockConfig()
    
    print("Database config not available - using fallback configuration")

class HybridVectorStore:
    """
    Hybrid storage that seamlessly switches between database and memory
    Provides consistent interface regardless of backend
    """
    
    def __init__(self):
        self.config = get_storage_config()
        self.storage_backend = None
        self.is_database_mode = False
        
        # In-memory storage (fallback)
        self.memory_documents = []
        self.memory_chunks = []
        self.memory_embeddings = []
        self.memory_doc_id_counter = 0
        
        # Initialize storage backend
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize appropriate storage backend"""
        if DB_AVAILABLE and self.config.is_database_available():
            try:
                self.storage_backend = get_db_vector_store(self.config.database_url)
                self.is_database_mode = True
                logger.info("Using database storage backend")
            except Exception as e:
                logger.error(f"Failed to initialize database storage: {e}")
                self._fallback_to_memory()
        else:
            self._fallback_to_memory()
    
    def _fallback_to_memory(self):
        """Fallback to in-memory storage"""
        self.is_database_mode = False
        self.storage_backend = None
        logger.warning("Using in-memory storage (limited capacity)")
        
        # Check if we need to migrate data
        if hasattr(self, 'memory_documents') and self.memory_documents:
            logger.info(f"Keeping {len(self.memory_documents)} documents in memory")
    
    def add_document(self, filename: str, chunks: List[str], 
                    embeddings: Optional[List[np.ndarray]] = None,
                    metadata: Dict[str, Any] = None) -> int:
        """
        Add document with chunks to storage
        
        Returns:
            Document ID
        """
        if self.is_database_mode:
            try:
                return self.storage_backend.add_document(filename, chunks, metadata)
            except Exception as e:
                logger.error(f"Database storage failed: {e}")
                self._fallback_to_memory()
        
        # In-memory storage
        if len(self.memory_documents) >= self.config.max_memory_documents:
            raise MemoryError(
                f"Memory storage limit reached ({self.config.max_memory_documents} documents). "
                "Please configure database storage for larger datasets."
            )
        
        # Check memory usage warning
        if len(self.memory_documents) >= self.config.max_memory_documents * self.config.memory_warning_threshold:
            warnings.warn(
                f"Memory storage at {len(self.memory_documents)} documents "
                f"({int(100 * len(self.memory_documents) / self.config.max_memory_documents)}% of limit). "
                "Consider setting up database storage.",
                ResourceWarning
            )
        
        # Add to memory storage
        doc_id = self.memory_doc_id_counter
        self.memory_doc_id_counter += 1
        
        self.memory_documents.append({
            'id': doc_id,
            'filename': filename,
            'upload_timestamp': datetime.now(),
            'chunk_count': len(chunks),
            'metadata': metadata or {}
        })
        
        # Add chunks and embeddings
        for i, chunk in enumerate(chunks):
            self.memory_chunks.append({
                'document_id': doc_id,
                'chunk_index': i,
                'text': chunk,
                'filename': filename
            })
            
            if embeddings and i < len(embeddings):
                self.memory_embeddings.append(embeddings[i])
            else:
                self.memory_embeddings.append(None)
        
        logger.info(f"Added document {filename} to memory storage (ID: {doc_id})")
        return doc_id
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
              min_similarity: float = 0.4) -> List[Dict[str, Any]]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar chunks with metadata
        """
        if self.is_database_mode:
            try:
                # Database backend expects query string, not embedding
                # This is a compatibility layer
                return self.storage_backend.search(
                    query="",  # Not used in our implementation
                    top_k=top_k,
                    min_similarity=min_similarity
                )
            except Exception as e:
                logger.error(f"Database search failed: {e}")
                self._fallback_to_memory()
        
        # In-memory search
        if not self.memory_embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.memory_embeddings):
            if doc_embedding is not None:
                # Cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                if similarity >= min_similarity:
                    similarities.append({
                        'index': i,
                        'similarity': float(similarity)
                    })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get top results
        results = []
        for item in similarities[:top_k]:
            chunk_idx = item['index']
            if chunk_idx < len(self.memory_chunks):
                chunk = self.memory_chunks[chunk_idx]
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
        if self.is_database_mode:
            try:
                return self.storage_backend.get_all_documents(limit, offset)
            except Exception as e:
                logger.error(f"Database query failed: {e}")
                self._fallback_to_memory()
        
        # In-memory pagination
        start = offset
        end = min(offset + limit, len(self.memory_documents))
        
        return self.memory_documents[start:end]
    
    def delete_document(self, document_id: int) -> bool:
        """Delete document and its chunks"""
        if self.is_database_mode:
            try:
                return self.storage_backend.delete_document(document_id)
            except Exception as e:
                logger.error(f"Database delete failed: {e}")
                self._fallback_to_memory()
        
        # In-memory delete
        # Remove document
        self.memory_documents = [
            doc for doc in self.memory_documents 
            if doc['id'] != document_id
        ]
        
        # Remove chunks and embeddings
        indices_to_remove = []
        for i, chunk in enumerate(self.memory_chunks):
            if chunk['document_id'] == document_id:
                indices_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for i in reversed(indices_to_remove):
            del self.memory_chunks[i]
            if i < len(self.memory_embeddings):
                del self.memory_embeddings[i]
        
        logger.info(f"Deleted document {document_id} from memory storage")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        if self.is_database_mode:
            try:
                stats = self.storage_backend.get_stats()
                stats['storage_mode'] = 'database'
                return stats
            except Exception as e:
                logger.error(f"Failed to get database stats: {e}")
        
        # In-memory stats
        memory_usage_mb = 0
        if self.memory_embeddings:
            # Estimate memory usage (rough calculation)
            embeddings_size = len(self.memory_embeddings) * 384 * 4 / 1024 / 1024  # MB
            chunks_size = sum(len(chunk['text']) for chunk in self.memory_chunks) / 1024 / 1024
            memory_usage_mb = embeddings_size + chunks_size
        
        return {
            'storage_mode': 'memory',
            'documents': len(self.memory_documents),
            'chunks': len(self.memory_chunks),
            'embeddings': len(self.memory_embeddings),
            'memory_usage_mb': round(memory_usage_mb, 2),
            'capacity_documents': f"{len(self.memory_documents)}/{self.config.max_memory_documents}",
            'capacity_chunks': f"{len(self.memory_chunks)}/{self.config.max_memory_chunks}",
            'is_near_limit': len(self.memory_documents) >= self.config.max_memory_documents * 0.8
        }
    
    def migrate_to_database(self, database_url: str) -> bool:
        """Migrate current in-memory data to database"""
        if self.is_database_mode:
            logger.info("Already using database storage")
            return True
        
        if not DB_AVAILABLE:
            logger.error("Database dependencies not installed")
            return False
        
        try:
            # Create database storage
            db_store = DatabaseVectorStore(database_url)
            
            # Migrate documents
            logger.info(f"Migrating {len(self.memory_documents)} documents to database...")
            
            # Group chunks by document
            doc_chunks = {}
            doc_embeddings = {}
            
            for i, chunk in enumerate(self.memory_chunks):
                doc_id = chunk['document_id']
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                    doc_embeddings[doc_id] = []
                
                doc_chunks[doc_id].append(chunk['text'])
                if i < len(self.memory_embeddings):
                    doc_embeddings[doc_id].append(self.memory_embeddings[i])
            
            # Migrate each document
            for doc in self.memory_documents:
                old_id = doc['id']
                chunks = doc_chunks.get(old_id, [])
                
                new_id = db_store.add_document(
                    filename=doc['filename'],
                    chunks=chunks,
                    metadata=doc.get('metadata', {})
                )
                
                logger.info(f"Migrated {doc['filename']} (ID: {old_id} -> {new_id})")
            
            # Switch to database mode
            self.storage_backend = db_store
            self.is_database_mode = True
            
            # Clear memory storage
            self.memory_documents.clear()
            self.memory_chunks.clear()
            self.memory_embeddings.clear()
            
            logger.info("Migration to database completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

# Global hybrid storage instance
_hybrid_storage: Optional[HybridVectorStore] = None

def get_hybrid_storage() -> HybridVectorStore:
    """Get or create hybrid storage instance"""
    global _hybrid_storage
    if _hybrid_storage is None:
        _hybrid_storage = HybridVectorStore()
    return _hybrid_storage

# Compatibility functions for existing code
def find_similar_chunks_hybrid(query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
    """Find similar chunks using hybrid storage"""
    storage = get_hybrid_storage()
    return storage.search(query_embedding, top_k=top_k, min_similarity=0.4)

def add_document_hybrid(filename: str, chunks: List[str], embeddings: List[np.ndarray]) -> int:
    """Add document using hybrid storage"""
    storage = get_hybrid_storage()
    return storage.add_document(filename, chunks, embeddings)

def get_storage_stats() -> Dict[str, Any]:
    """Get current storage statistics"""
    storage = get_hybrid_storage()
    return storage.get_stats()