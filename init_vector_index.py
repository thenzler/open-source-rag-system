#!/usr/bin/env python3
"""
Initialize vector index from database embeddings
"""
import asyncio
import sys
from pathlib import Path
import sqlite3
import pickle
import gzip

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def init_index():
    try:
        from core.di.services import ServiceConfiguration, get_container
        from core.repositories.interfaces import IVectorSearchRepository, IDocumentRepository
        from core.repositories.models import Embedding
        from core.services.document_service import DocumentProcessingService
        
        # Configure DI
        ServiceConfiguration.configure_all()
        container = get_container()
        
        # Get services
        vector_repo = container.get(IVectorSearchRepository)
        doc_repo = container.get(IDocumentRepository)
        
        print("Loading embeddings from database...")
        
        # Load embeddings directly from SQLite
        conn = sqlite3.connect('data/rag_database.db')
        
        # Get all embeddings with chunk info
        cursor = conn.execute("""
            SELECT e.id, e.chunk_id, e.embedding_data, e.embedding_model, e.dimensions,
                   c.document_id, c.text
            FROM embeddings e
            JOIN chunks c ON e.chunk_id = c.id
        """)
        
        embeddings = []
        chunk_texts = {}
        
        for row in cursor.fetchall():
            # Decompress embedding vector
            embedding_vector = pickle.loads(gzip.decompress(row[2]))
            
            # Create Embedding object
            embedding = Embedding(
                id=row[0],
                chunk_id=row[1],
                document_id=row[5],
                embedding_vector=embedding_vector,
                embedding_model=row[3],
                vector_dimension=row[4]
            )
            embeddings.append(embedding)
            
            # Store chunk text for later
            chunk_texts[row[1]] = row[6]
        
        print(f"Loaded {len(embeddings)} embeddings")
        
        if embeddings:
            # Build the index
            print("Building vector index...")
            success = await vector_repo.build_index(embeddings)
            
            if success:
                print("OK Vector index initialized successfully!")
                
                # Test with a query
                print("\nTesting with sample query...")
                test_results = await vector_repo.search_similar_text(
                    query="Recycling",
                    limit=3
                )
                print(f"Test query returned {test_results.total_count} results")
            else:
                print("ERROR Failed to build vector index")
        else:
            print("No embeddings found in database")
        
        conn.close()
        
        # Now process any unprocessed documents
        print("\nChecking for unprocessed documents...")
        from core.repositories.base import SearchOptions
        from core.repositories.models import DocumentStatus
        
        all_docs = await doc_repo.list_all(SearchOptions(page_size=100))
        unprocessed = []
        
        for doc in all_docs.items:
            if doc.status in ['uploaded', 'failed']:
                unprocessed.append(doc)
        
        if unprocessed:
            print(f"Found {len(unprocessed)} unprocessed documents")
            # Could process them here if needed
        else:
            print("All documents are processed")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(init_index())