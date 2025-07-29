#!/usr/bin/env python3
"""
Debug Search Flow
Investigate why search returns no results despite loaded embeddings
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_search_flow():
    """Debug the complete search flow step by step"""
    try:
        logger.info("=== DEBUGGING SEARCH FLOW ===")
        
        # Import after path setup
        from core.repositories.factory import RepositoryFactory
        
        # Initialize repositories
        rag_repo = RepositoryFactory.create_production_repository()
        await rag_repo.initialize()
        vector_repo = rag_repo.vector_search
        
        # Test query
        test_query = "Welche Regeln gelten für Gewerbeabfall?"
        logger.info(f"Testing query: {test_query}")
        
        # Step 1: Check index statistics
        logger.info("\n--- STEP 1: Index Statistics ---")
        stats = await vector_repo.get_index_statistics()
        logger.info(f"Index stats: {stats}")
        
        # Step 2: Test search_similar directly  
        logger.info("\n--- STEP 2: Direct Vector Search ---")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([test_query])[0].tolist()
        
        logger.info(f"Query embedding shape: {len(query_embedding)}")
        logger.info(f"Query embedding sample: {query_embedding[:5]}...")
        
        # Direct search
        vector_results = await vector_repo.search_similar(
            query_vector=query_embedding,
            top_k=5
        )
        
        logger.info(f"Direct vector search results: {len(vector_results)} items")
        for i, (embedding_id, similarity) in enumerate(vector_results[:3]):
            logger.info(f"  Result {i+1}: Embedding {embedding_id}, Similarity: {similarity:.6f}")
        
        # Step 3: Test search_similar_text
        logger.info("\n--- STEP 3: Text-Based Search ---")
        text_results = await vector_repo.search_similar_text(
            query=test_query,
            limit=5,
            threshold=0.1  # Very low threshold
        )
        
        logger.info(f"Text search results: {len(text_results.items)} items")
        for i, item in enumerate(text_results.items[:3]):
            similarity = item.metadata.get('similarity_score', 0.0)
            content_preview = (item.text_content or '')[:100] + "..." if len(item.text_content or '') > 100 else item.text_content
            logger.info(f"  Result {i+1}: Doc {item.document_id}, Similarity: {similarity:.6f}, Content: {content_preview}")
        
        # Step 4: Check database directly
        logger.info("\n--- STEP 4: Database Check ---")
        import sqlite3
        conn = sqlite3.connect('data/rag_database.db')
        
        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        logger.info(f"Chunks in database: {chunk_count}")
        
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings") 
        embedding_count = cursor.fetchone()[0]
        logger.info(f"Embeddings in database: {embedding_count}")
        
        # Sample some chunks
        cursor = conn.execute("""
            SELECT c.text, e.id as embedding_id
            FROM chunks c
            JOIN embeddings e ON c.id = e.chunk_id
            WHERE c.text LIKE '%Gewerbe%' OR c.text LIKE '%Abfall%'
            LIMIT 3
        """)
        
        relevant_chunks = cursor.fetchall()
        logger.info(f"Relevant chunks found: {len(relevant_chunks)}")
        for text, emb_id in relevant_chunks:
            text_preview = text[:100] + "..." if len(text) > 100 else text
            logger.info(f"  Embedding {emb_id}: {text_preview}")
        
        conn.close()
        
        # Step 5: Test chunk loading
        logger.info("\n--- STEP 5: Chunk Loading Test ---")
        if vector_results:
            embedding_ids = [int(eid) for eid, score in vector_results[:3]]
            chunk_data = await vector_repo._load_chunks_by_embedding_ids(embedding_ids)
            logger.info(f"Loaded chunk data for {len(chunk_data)} embeddings")
            
            for emb_id, data in chunk_data.items():
                text_preview = data['text'][:100] + "..." if len(data['text']) > 100 else data['text']
                logger.info(f"  Embedding {emb_id}: Doc {data['document_id']}, Text: {text_preview}")
        
        return True
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_search_flow())
    sys.exit(0 if success else 1)