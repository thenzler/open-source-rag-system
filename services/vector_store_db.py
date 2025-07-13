#!/usr/bin/env python3
"""
Database-backed Vector Store Service
Replaces in-memory storage with PostgreSQL + pgvector
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
import json
from datetime import datetime, timedelta

# Import database manager
try:
    from database_migration import DatabaseManager, Document, DocumentChunk
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    DatabaseManager = None

logger = logging.getLogger(__name__)

class DatabaseVectorStore:
    """
    Vector store backed by PostgreSQL with pgvector
    Handles millions of documents without memory issues
    """
    
    def __init__(self, connection_string=None, embedding_model=None):
        if not DB_AVAILABLE:
            raise ImportError("Please install database dependencies: pip install sqlalchemy pgvector psycopg2-binary")
        
        self.db = DatabaseManager(connection_string)
        self.embedding_model = embedding_model or SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384  # all-MiniLM-L6-v2 dimension
        
        # Cache settings
        self.cache_ttl = timedelta(hours=1)
        self._query_cache = {}
        
        logger.info("Database vector store initialized")
    
    def add_document(self, filename: str, chunks: List[str], metadata: Dict[str, Any] = None) -> int:
        """
        Add document and chunks to database
        
        Returns:
            Document ID
        """
        # Add document record
        doc_id = self.db.add_document(
            filename=filename,
            file_size=sum(len(chunk) for chunk in chunks),
            file_type=filename.split('.')[-1] if '.' in filename else 'unknown',
            metadata=metadata or {}
        )
        
        # Generate embeddings for chunks
        if chunks:
            embeddings = self.embedding_model.encode(chunks)
            chunks_with_embeddings = list(zip(chunks, embeddings.tolist()))
            self.db.add_chunks(doc_id, chunks_with_embeddings)
        
        logger.info(f"Added document {filename} with {len(chunks)} chunks to database")
        return doc_id
    
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.4) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in database
        
        Args:
            query: Search query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar chunks with metadata
        """
        # Check cache first
        cache_key = self._get_cache_key(query, top_k, min_similarity)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return cached_result
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search in database
        results = self.db.search_similar_chunks(
            query_embedding=query_embedding.tolist(),
            top_k=top_k,
            min_similarity=min_similarity
        )
        
        # Cache results
        self._cache_result(cache_key, results)
        
        return results
    
    def update_document_chunks(self, document_id: int, new_chunks: List[str]) -> bool:
        """Update chunks for an existing document"""
        session = self.db.get_session()
        try:
            # Delete old chunks
            session.query(DocumentChunk).filter_by(document_id=document_id).delete()
            
            # Add new chunks
            if new_chunks:
                embeddings = self.embedding_model.encode(new_chunks)
                chunks_with_embeddings = list(zip(new_chunks, embeddings.tolist()))
                self.db.add_chunks(document_id, chunks_with_embeddings)
            
            session.commit()
            logger.info(f"Updated document {document_id} with {len(new_chunks)} chunks")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update document {document_id}: {e}")
            return False
        finally:
            session.close()
    
    def delete_document(self, document_id: int) -> bool:
        """Delete document and all its chunks"""
        try:
            self.db.delete_document(document_id)
            logger.info(f"Deleted document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def get_all_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get paginated list of documents"""
        session = self.db.get_session()
        try:
            docs = session.query(Document).limit(limit).offset(offset).all()
            return [
                {
                    'id': doc.id,
                    'filename': doc.filename,
                    'file_size': doc.file_size,
                    'file_type': doc.file_type,
                    'upload_timestamp': doc.upload_timestamp.isoformat(),
                    'processing_status': doc.processing_status,
                    'metadata': doc.metadata
                }
                for doc in docs
            ]
        finally:
            session.close()
    
    def get_document_chunks(self, document_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chunks for a specific document"""
        session = self.db.get_session()
        try:
            chunks = session.query(DocumentChunk).filter_by(
                document_id=document_id
            ).order_by(DocumentChunk.chunk_index).limit(limit).all()
            
            return [
                {
                    'chunk_id': chunk.id,
                    'chunk_index': chunk.chunk_index,
                    'text': chunk.text,
                    'character_count': chunk.character_count,
                    'metadata': chunk.metadata
                }
                for chunk in chunks
            ]
        finally:
            session.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = self.db.get_stats()
        stats['cache_size'] = len(self._query_cache)
        stats['embedding_model'] = self.embedding_model.get_config()['model_name_or_path']
        stats['embedding_dimension'] = self.embedding_dimension
        return stats
    
    def optimize_indices(self):
        """Optimize database indices for better performance"""
        with self.db.engine.connect() as conn:
            # Vacuum and analyze tables
            conn.execute("VACUUM ANALYZE documents")
            conn.execute("VACUUM ANALYZE document_chunks")
            
            # Reindex vector indices
            conn.execute("REINDEX INDEX idx_embedding")
            conn.commit()
        
        logger.info("Database indices optimized")
    
    def _get_cache_key(self, query: str, top_k: int, min_similarity: float) -> str:
        """Generate cache key for query"""
        key_data = f"{query}:{top_k}:{min_similarity}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached result if still valid"""
        if cache_key in self._query_cache:
            cached_time, cached_result = self._query_cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_result
            else:
                del self._query_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: List[Dict[str, Any]]):
        """Cache query result"""
        self._query_cache[cache_key] = (datetime.now(), result)
        
        # Limit cache size
        if len(self._query_cache) > 1000:
            # Remove oldest entries
            sorted_cache = sorted(self._query_cache.items(), key=lambda x: x[1][0])
            for key, _ in sorted_cache[:100]:
                del self._query_cache[key]
    
    def clear_cache(self):
        """Clear query cache"""
        self._query_cache.clear()
        logger.info("Query cache cleared")

# Singleton instance
_db_vector_store: Optional[DatabaseVectorStore] = None

def get_db_vector_store(connection_string=None) -> DatabaseVectorStore:
    """Get or create database vector store instance"""
    global _db_vector_store
    if _db_vector_store is None:
        _db_vector_store = DatabaseVectorStore(connection_string)
    return _db_vector_store

# Example migration function
def migrate_to_database(old_vector_store, documents, chunks, embeddings):
    """Migrate from in-memory to database storage"""
    db_store = get_db_vector_store()
    
    print("Starting migration to database...")
    
    # Group chunks by document
    doc_chunks = {}
    for i, chunk in enumerate(chunks):
        doc_id = chunk.get('document_id', 0)
        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []
        doc_chunks[doc_id].append((i, chunk['text']))
    
    # Migrate each document
    for old_doc_id, doc in enumerate(documents):
        # Get chunks for this document
        chunk_texts = []
        if old_doc_id in doc_chunks:
            # Sort by chunk index to maintain order
            sorted_chunks = sorted(doc_chunks[old_doc_id], key=lambda x: x[0])
            chunk_texts = [text for _, text in sorted_chunks]
        
        # Add to database
        new_doc_id = db_store.add_document(
            filename=doc['filename'],
            chunks=chunk_texts,
            metadata=doc
        )
        
        print(f"Migrated: {doc['filename']} ({len(chunk_texts)} chunks)")
    
    print(f"Migration complete! Stats: {db_store.get_stats()}")
    return db_store