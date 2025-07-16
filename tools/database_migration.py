#!/usr/bin/env python3
"""
Database Migration for RAG System
Moves from in-memory storage to PostgreSQL with pgvector
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from datetime import datetime
import numpy as np

Base = declarative_base()

# Database Models
class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500))
    file_size = Column(Integer)
    file_type = Column(String(50))
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    processing_status = Column(String(50), default='pending')
    metadata = Column(JSON)
    
    # Indexes for fast lookup
    __table_args__ = (
        Index('idx_filename', 'filename'),
        Index('idx_upload_timestamp', 'upload_timestamp'),
        Index('idx_processing_status', 'processing_status'),
    )

class DocumentChunk(Base):
    __tablename__ = 'document_chunks'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(384))  # 384 dimensions for all-MiniLM-L6-v2
    character_count = Column(Integer)
    metadata = Column(JSON)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_document_id', 'document_id'),
        Index('idx_embedding', 'embedding', postgresql_using='ivfflat'),  # Vector similarity index
    )

class QueryCache(Base):
    __tablename__ = 'query_cache'
    
    id = Column(Integer, primary_key=True)
    query_hash = Column(String(64), unique=True)
    query_text = Column(Text)
    results = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_query_hash', 'query_hash'),
        Index('idx_created_at', 'created_at'),
    )

# Database connection manager
class DatabaseManager:
    def __init__(self, connection_string=None):
        if not connection_string:
            connection_string = os.getenv(
                'DATABASE_URL',
                'postgresql://user:password@localhost/ragdb'
            )
        
        self.engine = create_engine(connection_string)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        
        # Create pgvector extension
        with self.engine.connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
    
    def get_session(self):
        return self.SessionLocal()
    
    def add_document(self, filename, file_path=None, file_size=0, file_type='unknown', metadata=None):
        """Add a new document to the database"""
        session = self.get_session()
        try:
            doc = Document(
                filename=filename,
                file_path=file_path,
                file_size=file_size,
                file_type=file_type,
                metadata=metadata or {}
            )
            session.add(doc)
            session.commit()
            return doc.id
        finally:
            session.close()
    
    def add_chunks(self, document_id, chunks_with_embeddings):
        """Add document chunks with embeddings"""
        session = self.get_session()
        try:
            for i, (text, embedding) in enumerate(chunks_with_embeddings):
                chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_index=i,
                    text=text,
                    embedding=embedding,
                    character_count=len(text),
                    metadata={}
                )
                session.add(chunk)
            session.commit()
        finally:
            session.close()
    
    def search_similar_chunks(self, query_embedding, top_k=5, min_similarity=0.4):
        """Vector similarity search using pgvector"""
        session = self.get_session()
        try:
            # Convert to numpy array if needed
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
            
            # Use pgvector's <-> operator for cosine distance
            # Note: pgvector uses distance, so we convert to similarity
            results = session.query(
                DocumentChunk,
                (1 - DocumentChunk.embedding.cosine_distance(query_embedding)).label('similarity')
            ).filter(
                (1 - DocumentChunk.embedding.cosine_distance(query_embedding)) >= min_similarity
            ).order_by(
                DocumentChunk.embedding.cosine_distance(query_embedding)
            ).limit(top_k).all()
            
            # Format results
            similar_chunks = []
            for chunk, similarity in results:
                doc = session.query(Document).filter_by(id=chunk.document_id).first()
                similar_chunks.append({
                    'chunk_id': chunk.id,
                    'document_id': chunk.document_id,
                    'text': chunk.text,
                    'similarity': float(similarity),
                    'filename': doc.filename if doc else 'Unknown',
                    'metadata': chunk.metadata
                })
            
            return similar_chunks
        finally:
            session.close()
    
    def get_all_documents(self):
        """Get all documents (paginated for large datasets)"""
        session = self.get_session()
        try:
            return session.query(Document).all()
        finally:
            session.close()
    
    def delete_document(self, document_id):
        """Delete document and its chunks"""
        session = self.get_session()
        try:
            # Delete chunks first
            session.query(DocumentChunk).filter_by(document_id=document_id).delete()
            # Delete document
            session.query(Document).filter_by(id=document_id).delete()
            session.commit()
        finally:
            session.close()
    
    def get_stats(self):
        """Get database statistics"""
        session = self.get_session()
        try:
            doc_count = session.query(Document).count()
            chunk_count = session.query(DocumentChunk).count()
            
            # Get index statistics
            with self.engine.connect() as conn:
                result = conn.execute("""
                    SELECT 
                        pg_size_pretty(pg_total_relation_size('document_chunks')) as chunks_size,
                        pg_size_pretty(pg_total_relation_size('documents')) as docs_size
                """).fetchone()
            
            return {
                'documents': doc_count,
                'chunks': chunk_count,
                'chunks_table_size': result.chunks_size if result else 'Unknown',
                'documents_table_size': result.docs_size if result else 'Unknown'
            }
        finally:
            session.close()

# Migration script
def migrate_from_memory(documents, document_chunks, document_embeddings):
    """Migrate existing in-memory data to database"""
    db = DatabaseManager()
    
    print("Starting database migration...")
    
    # Migrate documents
    doc_id_mapping = {}
    for old_id, doc in enumerate(documents):
        new_id = db.add_document(
            filename=doc['filename'],
            file_path=doc.get('file_path'),
            file_size=doc.get('file_size', 0),
            file_type=doc['filename'].split('.')[-1] if '.' in doc['filename'] else 'unknown',
            metadata=doc
        )
        doc_id_mapping[old_id] = new_id
        print(f"Migrated document: {doc['filename']} (ID: {old_id} -> {new_id})")
    
    # Migrate chunks with embeddings
    for i, chunk in enumerate(document_chunks):
        if i < len(document_embeddings) and document_embeddings[i] is not None:
            old_doc_id = chunk.get('document_id', 0)
            new_doc_id = doc_id_mapping.get(old_doc_id, old_doc_id)
            
            db.add_chunks(new_doc_id, [(chunk['text'], document_embeddings[i])])
            
            if i % 100 == 0:
                print(f"Migrated {i} chunks...")
    
    print(f"Migration complete! Stats: {db.get_stats()}")

if __name__ == "__main__":
    # Example usage
    db = DatabaseManager()
    print("Database initialized!")
    print(f"Stats: {db.get_stats()}")