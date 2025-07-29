#!/usr/bin/env python3
"""
Rebuild Vector Index
Load existing embeddings from database into FAISS index
"""
import os
import sys
import asyncio
import logging
import sqlite3
import pickle
import gzip
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def rebuild_vector_index():
    """Rebuild the vector index from database embeddings"""
    try:
        logger.info("Starting vector index rebuild...")
        
        # Import after path setup
        from core.repositories.factory import RepositoryFactory
        from core.repositories.models import Embedding
        
        # Initialize repositories
        rag_repo = RepositoryFactory.create_production_repository()
        await rag_repo.initialize()
        
        vector_repo = rag_repo.vector_search
        
        # Get all embeddings from database
        logger.info("Loading embeddings from database...")
        
        db_path = "data/rag_database.db"
        conn = sqlite3.connect(db_path)
        
        cursor = conn.execute("""
            SELECT e.id, e.chunk_id, e.embedding_data, e.embedding_model, e.dimensions,
                   c.document_id, c.text, c.chunk_index
            FROM embeddings e
            JOIN chunks c ON e.chunk_id = c.id
            ORDER BY e.id
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        logger.info(f"Found {len(rows)} embeddings in database")
        
        if not rows:
            logger.error("No embeddings found in database!")
            return False
        
        # Convert to Embedding objects
        embeddings = []
        
        for row in rows:
            embedding_id, chunk_id, embedding_data, model, dims, doc_id, text, chunk_idx = row
            
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
                    vector_dimension=dims
                )
                
                # Add metadata for search results
                embedding.metadata = {
                    'chunk_index': chunk_idx,
                    'text_content': text,
                    'document_id': doc_id
                }
                
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Error processing embedding {embedding_id}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(embeddings)} embeddings")
        
        # Rebuild the vector index
        logger.info("Rebuilding vector index...")
        success = await vector_repo.build_index(embeddings)
        
        if success:
            logger.info("✅ Vector index rebuilt successfully!")
            
            # Test the index
            logger.info("Testing rebuilt index...")
            stats = await vector_repo.get_index_statistics()
            logger.info(f"Index statistics: {stats}")
            
            # Test search
            test_query = "Welche Regeln gelten für Gewerbeabfall?"
            result = await vector_repo.search_similar_text(test_query, limit=5, threshold=0.1)
            
            logger.info(f"Test search returned {len(result.items)} results")
            
            for i, item in enumerate(result.items[:3]):
                similarity = item.metadata.get('similarity_score', 0.0)
                content_preview = (item.text_content or '')[:100] + "..." if len(item.text_content or '') > 100 else item.text_content
                logger.info(f"Result {i+1}: Doc {item.document_id}, Similarity: {similarity:.6f}, Content: {content_preview}")
            
            if len(result.items) > 0:
                logger.info("🎉 Vector search is now working!")
                return True
            else:
                logger.warning("❌ Vector search still returns no results")
                return False
        else:
            logger.error("❌ Failed to rebuild vector index")
            return False
        
    except Exception as e:
        logger.error(f"Vector index rebuild failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(rebuild_vector_index())
    sys.exit(0 if success else 1)