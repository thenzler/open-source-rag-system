#!/usr/bin/env python3
"""
Persistent Storage Service with SQLite
Provides persistent document storage with future PostgreSQL migration path
"""

import sqlite3
import logging
import json
import pickle
import gzip
from typing import List, Dict, Any, Optional, Union
import numpy as np
from datetime import datetime
from pathlib import Path
import threading
import hashlib

logger = logging.getLogger(__name__)

class PersistentStorage:
    """
    SQLite-based persistent storage for documents, chunks, and embeddings
    Designed for easy migration to PostgreSQL in the future
    """
    
    def __init__(self, db_path: str = "data/rag_database.db"):
        self.db_path = db_path
        self.connection_lock = threading.Lock()
        self._init_database()
        
        logger.info(f"Persistent storage initialized: {db_path}")
    
    def _init_database(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    original_filename TEXT,
                    file_type TEXT,
                    file_size INTEGER,
                    content_type TEXT,
                    status TEXT DEFAULT 'processed',
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_count INTEGER DEFAULT 0,
                    metadata_json TEXT,
                    content_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    character_count INTEGER,
                    word_count INTEGER,
                    quality_score REAL,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            """)
            
            # Embeddings table (storing as compressed binary data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id INTEGER NOT NULL,
                    embedding_data BLOB NOT NULL,
                    embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2',
                    dimensions INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chunk_id) REFERENCES chunks (id) ON DELETE CASCADE
                )
            """)
            
            # Create indices for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            
            conn.commit()
            logger.info("Database tables initialized successfully")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper settings"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
        return conn
    
    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize numpy array to compressed binary data"""
        if embedding is None:
            return b''
        
        # Convert to numpy array if it's a list
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        elif not isinstance(embedding, np.ndarray):
            logger.warning(f"Unexpected embedding type: {type(embedding)}")
            return b''
        
        # Ensure float32 for consistency
        embedding = embedding.astype(np.float32)
        
        # Compress with gzip for space efficiency
        return gzip.compress(embedding.tobytes())
    
    def _deserialize_embedding(self, data: bytes) -> Optional[np.ndarray]:
        """Deserialize binary data back to numpy array"""
        if not data:
            return None
        
        try:
            # Decompress and reconstruct numpy array
            decompressed = gzip.decompress(data)
            # Assuming 384 dimensions for all-MiniLM-L6-v2
            embedding = np.frombuffer(decompressed, dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error(f"Error deserializing embedding: {e}")
            return None
    
    def add_document(self, filename: str, chunks: List[str], 
                    embeddings: Optional[List[np.ndarray]] = None,
                    metadata: Dict[str, Any] = None) -> int:
        """
        Add document with chunks and embeddings
        
        Args:
            filename: Document filename
            chunks: List of text chunks
            embeddings: Optional list of embeddings for chunks
            metadata: Optional metadata dictionary
            
        Returns:
            int: Document ID
        """
        
        metadata = metadata or {}
        content_hash = hashlib.md5(f"{filename}{''.join(chunks)}".encode()).hexdigest()
        
        with self.connection_lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                try:
                    # Insert document
                    cursor.execute("""
                        INSERT INTO documents (
                            filename, original_filename, file_type, file_size, 
                            content_type, chunk_count, metadata_json, content_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        filename,
                        metadata.get('original_filename', filename),
                        metadata.get('file_type', 'unknown'),
                        metadata.get('file_size', 0),
                        metadata.get('content_type', 'unknown'),
                        len(chunks),
                        json.dumps(metadata),
                        content_hash
                    ))
                    
                    document_id = cursor.lastrowid
                    
                    # Insert chunks
                    for i, chunk in enumerate(chunks):
                        cursor.execute("""
                            INSERT INTO chunks (
                                document_id, chunk_index, text, character_count, word_count
                            ) VALUES (?, ?, ?, ?, ?)
                        """, (
                            document_id,
                            i,
                            chunk,
                            len(chunk),
                            len(chunk.split())
                        ))
                        
                        chunk_id = cursor.lastrowid
                        
                        # Insert embedding if provided
                        if embeddings and i < len(embeddings) and embeddings[i] is not None:
                            embedding_data = self._serialize_embedding(embeddings[i])
                            if embedding_data:
                                cursor.execute("""
                                    INSERT INTO embeddings (
                                        chunk_id, embedding_data, dimensions
                                    ) VALUES (?, ?, ?)
                                """, (
                                    chunk_id,
                                    embedding_data,
                                    len(embeddings[i]) if embeddings[i] is not None else 0
                                ))
                    
                    conn.commit()
                    logger.info(f"Added document '{filename}' (ID: {document_id}) with {len(chunks)} chunks")
                    return document_id
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error adding document: {e}")
                    raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
              min_similarity: float = 0.4) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity"""
        
        if query_embedding is None:
            return []
        
        with self.connection_lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get all embeddings with chunk and document info
                cursor.execute("""
                    SELECT 
                        e.chunk_id,
                        e.embedding_data,
                        c.text,
                        c.chunk_index,
                        d.id as document_id,
                        d.filename
                    FROM embeddings e
                    JOIN chunks c ON e.chunk_id = c.id
                    JOIN documents d ON c.document_id = d.id
                    WHERE e.embedding_data IS NOT NULL AND e.embedding_data != ''
                """)
                
                results = []
                max_similarity = 0.0
                
                for row in cursor.fetchall():
                    # Deserialize embedding
                    doc_embedding = self._deserialize_embedding(row['embedding_data'])
                    if doc_embedding is None:
                        continue
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    
                    max_similarity = max(max_similarity, similarity)
                    
                    if similarity >= min_similarity:
                        results.append({
                            'chunk_id': row['chunk_id'],
                            'document_id': row['document_id'],
                            'text': row['text'],
                            'similarity': float(similarity),
                            'filename': row['filename'],
                            'chunk_index': row['chunk_index'],
                            'metadata': {}
                        })
                
                # Sort by similarity and return top_k
                results.sort(key=lambda x: x['similarity'], reverse=True)
                
                logger.info(f"Search results: {len(results)} chunks above threshold, max similarity: {max_similarity:.4f}")
                return results[:top_k]
    
    def get_all_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get paginated list of documents"""
        
        with self.connection_lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        id, filename, original_filename, file_type, file_size,
                        content_type, status, upload_date, chunk_count, metadata_json
                    FROM documents
                    ORDER BY upload_date DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                
                documents = []
                for row in cursor.fetchall():
                    metadata = json.loads(row['metadata_json'] or '{}')
                    documents.append({
                        'id': row['id'],
                        'filename': row['filename'],
                        'original_filename': row['original_filename'],
                        'file_type': row['file_type'],
                        'file_size': row['file_size'],
                        'content_type': row['content_type'],
                        'status': row['status'],
                        'upload_timestamp': datetime.fromisoformat(row['upload_date']),
                        'chunk_count': row['chunk_count'],
                        'metadata': metadata
                    })
                
                return documents
    
    def delete_document(self, document_id: int) -> bool:
        """Delete document and all related chunks and embeddings"""
        
        with self.connection_lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                try:
                    # Delete document (CASCADE will handle chunks and embeddings)
                    cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
                    deleted_count = cursor.rowcount
                    
                    conn.commit()
                    
                    if deleted_count > 0:
                        logger.info(f"Deleted document {document_id}")
                        return True
                    else:
                        logger.warning(f"Document {document_id} not found")
                        return False
                        
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error deleting document {document_id}: {e}")
                    return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        with self.connection_lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Count documents
                cursor.execute("SELECT COUNT(*) FROM documents")
                doc_count = cursor.fetchone()[0]
                
                # Count chunks
                cursor.execute("SELECT COUNT(*) FROM chunks")
                chunk_count = cursor.fetchone()[0]
                
                # Count embeddings
                cursor.execute("SELECT COUNT(*) FROM embeddings WHERE embedding_data IS NOT NULL")
                embedding_count = cursor.fetchone()[0]
                
                # Get database file size
                db_path = Path(self.db_path)
                db_size_mb = db_path.stat().st_size / 1024 / 1024 if db_path.exists() else 0
                
                return {
                    'storage_mode': 'persistent_sqlite',
                    'database_path': str(db_path.absolute()),
                    'documents': doc_count,
                    'chunks': chunk_count,
                    'embeddings': embedding_count,
                    'database_size_mb': round(db_size_mb, 2),
                    'is_persistent': True
                }
    
    def get_document_by_id(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get single document by ID"""
        
        with self.connection_lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        id, filename, original_filename, file_type, file_size,
                        content_type, status, upload_date, chunk_count, metadata_json
                    FROM documents WHERE id = ?
                """, (document_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                metadata = json.loads(row['metadata_json'] or '{}')
                return {
                    'id': row['id'],
                    'filename': row['filename'],
                    'original_filename': row['original_filename'],
                    'file_type': row['file_type'],
                    'file_size': row['file_size'],
                    'content_type': row['content_type'],
                    'status': row['status'],
                    'upload_timestamp': datetime.fromisoformat(row['upload_date']),
                    'chunk_count': row['chunk_count'],
                    'metadata': metadata
                }

# Global instance
_persistent_storage: Optional[PersistentStorage] = None

def get_persistent_storage(db_path: str = "data/rag_database.db") -> PersistentStorage:
    """Get or create persistent storage instance"""
    global _persistent_storage
    if _persistent_storage is None:
        _persistent_storage = PersistentStorage(db_path)
    return _persistent_storage

def init_persistent_storage(db_path: str = "data/rag_database.db") -> PersistentStorage:
    """Initialize persistent storage (for startup)"""
    global _persistent_storage
    _persistent_storage = PersistentStorage(db_path)
    return _persistent_storage