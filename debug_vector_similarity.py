#!/usr/bin/env python3
"""
Debug Vector Search Similarity Issues
Investigate why vector search returns 0.000 similarity for all queries
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

async def debug_vector_search():
    """Debug the vector search similarity calculation"""
    try:
        logger.info("Starting vector search debug...")
        
        # Check database contents
        db_path = "data/rag_database.db"
        if not Path(db_path).exists():
            logger.error(f"Database {db_path} does not exist")
            return
        
        conn = sqlite3.connect(db_path)
        
        # Check chunks table
        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        logger.info(f"Chunks in database: {chunk_count}")
        
        if chunk_count == 0:
            logger.error("No chunks found in database!")
            return
        
        # Check embeddings table
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        embedding_count = cursor.fetchone()[0]
        logger.info(f"Embeddings in database: {embedding_count}")
        
        if embedding_count == 0:
            logger.error("No embeddings found in database!")
            return
        
        # Check some sample chunks
        cursor = conn.execute("""
            SELECT c.document_id, c.chunk_index, c.text, e.embedding_data
            FROM chunks c
            JOIN embeddings e ON c.id = e.chunk_id
            LIMIT 5
        """)
        
        samples = cursor.fetchall()
        logger.info(f"Sample chunks with embeddings: {len(samples)}")
        
        for i, (doc_id, chunk_idx, text, embedding_data) in enumerate(samples):
            text_preview = text[:100] + "..." if len(text) > 100 else text
            logger.info(f"Sample {i+1}: Doc {doc_id}, Chunk {chunk_idx}, Text: {text_preview}")
            
            # Check embedding data
            try:
                decompressed = gzip.decompress(embedding_data)
                embedding = pickle.loads(decompressed)
                logger.info(f"  Embedding shape: {np.array(embedding).shape}")
                logger.info(f"  Embedding norm: {np.linalg.norm(embedding):.6f}")
                logger.info(f"  Embedding range: [{np.min(embedding):.6f}, {np.max(embedding):.6f}]")
            except Exception as e:
                logger.error(f"  Error loading embedding: {e}")
        
        conn.close()
        
        # Test with sentence transformers directly
        logger.info("\nTesting sentence transformers directly...")
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test texts
        test_texts = [
            "Welche Regeln gelten für Gewerbeabfall?",
            "Gewerbeabfall Regeln Entsorgung",
            "Commercial waste disposal rules",
            "Abfall Gewerbe Müll"
        ]
        
        embeddings = model.encode(test_texts)
        logger.info(f"Generated {len(embeddings)} test embeddings")
        
        for i, emb in enumerate(embeddings):
            logger.info(f"Test {i+1}: '{test_texts[i]}'")
            logger.info(f"  Shape: {emb.shape}")
            logger.info(f"  Norm: {np.linalg.norm(emb):.6f}")
            logger.info(f"  Range: [{np.min(emb):.6f}, {np.max(emb):.6f}]")
        
        # Calculate similarities between test embeddings
        logger.info("\nSimilarity matrix between test embeddings:")
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)
        
        for i in range(len(test_texts)):
            for j in range(len(test_texts)):
                if i != j:
                    logger.info(f"Similarity '{test_texts[i][:30]}...' <-> '{test_texts[j][:30]}...': {sim_matrix[i][j]:.6f}")
        
        # Now test the actual vector repository
        logger.info("\nTesting vector repository...")
        
        from core.repositories.factory import RepositoryFactory
        
        rag_repo = RepositoryFactory.create_production_repository()
        await rag_repo.initialize()
        vector_repo = rag_repo.vector_search
        
        # Test search
        test_query = "Welche Regeln gelten für Gewerbeabfall?"
        result = await vector_repo.search_similar_text(test_query, limit=5, threshold=0.1)
        
        logger.info(f"Vector repository search returned {len(result.items)} results")
        
        for i, item in enumerate(result.items):
            similarity = item.metadata.get('similarity_score', 0.0)
            content_preview = (item.text_content or '')[:100] + "..." if len(item.text_content or '') > 100 else item.text_content
            logger.info(f"Result {i+1}: Doc {item.document_id}, Similarity: {similarity:.6f}, Content: {content_preview}")
        
        # Test direct FAISS
        logger.info("\nTesting FAISS index directly...")
        
        # Get FAISS index statistics
        stats = await vector_repo.get_index_statistics()
        logger.info(f"FAISS index statistics: {stats}")
        
        # Test with lower threshold
        result_low = await vector_repo.search_similar_text(test_query, limit=10, threshold=0.0)
        logger.info(f"Low threshold search returned {len(result_low.items)} results")
        
        for i, item in enumerate(result_low.items[:5]):
            similarity = item.metadata.get('similarity_score', 0.0)
            content_preview = (item.text_content or '')[:100] + "..." if len(item.text_content or '') > 100 else item.text_content
            logger.info(f"Low threshold result {i+1}: Doc {item.document_id}, Similarity: {similarity:.6f}, Content: {content_preview}")
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_vector_search())