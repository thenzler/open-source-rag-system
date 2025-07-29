#!/usr/bin/env python3
"""
Script to rebuild the vector index from existing documents
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def rebuild_index():
    """Rebuild vector index from existing documents"""
    try:
        # Import after path is set
        from core.di.services import ServiceConfiguration, get_container
        from core.repositories.interfaces import IDocumentRepository, IVectorSearchRepository
        from core.repositories.models import DocumentStatus
        
        print("Initializing services...")
        
        # Configure DI container
        ServiceConfiguration.configure_all()
        container = get_container()
        
        # Get repositories
        doc_repo = container.get(IDocumentRepository)
        vector_repo = container.get(IVectorSearchRepository)
        
        # Get all documents
        from core.repositories.base import SearchOptions
        options = SearchOptions(page_size=1000)
        docs = await doc_repo.list_all(options)
        
        print(f"Found {docs.total_count} documents in database")
        
        # Get all successfully processed documents
        processed_count = 0
        for doc in docs.items:
            if doc.status in [DocumentStatus.COMPLETED, DocumentStatus.PROCESSED]:
                processed_count += 1
        
        print(f"Found {processed_count} processed documents")
        
        # Get embeddings from database
        import sqlite3
        import pickle
        import gzip
        
        conn = sqlite3.connect('data/rag_database.db')
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        embedding_count = cursor.fetchone()[0]
        print(f"Found {embedding_count} embeddings in database")
        
        if embedding_count > 0:
            # Load all embeddings
            from core.repositories.models import Embedding
            embeddings = []
            
            cursor = conn.execute("""
                SELECT id, chunk_id, embedding_data, embedding_model, dimensions 
                FROM embeddings
            """)
            
            for row in cursor.fetchall():
                # Decompress embedding vector
                embedding_vector = pickle.loads(gzip.decompress(row[2]))
                
                embedding = Embedding(
                    id=row[0],
                    chunk_id=row[1],
                    document_id=0,  # We don't have this in the current schema
                    embedding_vector=embedding_vector,
                    embedding_model=row[3],
                    vector_dimension=row[4]
                )
                embeddings.append(embedding)
            
            print(f"Loaded {len(embeddings)} embeddings")
            
            # Build vector index
            print("Building vector index...")
            success = await vector_repo.build_index(embeddings)
            
            if success:
                print("OK Vector index rebuilt successfully!")
            else:
                print("ERROR Failed to rebuild vector index")
        else:
            print("No embeddings found. Upload and process documents first.")
        
        conn.close()
        
    except Exception as e:
        print(f"Error rebuilding index: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(rebuild_index())