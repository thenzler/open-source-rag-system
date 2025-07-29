#!/usr/bin/env python3
"""
Debug search_similar_text method specifically
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_search_similar_text():
    """Debug the search_similar_text method directly"""
    try:
        logger.info("=== DEBUGGING search_similar_text METHOD ===")
        
        # Import after path setup
        from core.repositories.factory import RepositoryFactory
        
        # Initialize repositories
        rag_repo = RepositoryFactory.create_production_repository()
        await rag_repo.initialize()
        vector_repo = rag_repo.vector_search
        
        # Test query
        test_query = "Welche Regeln gelten für Gewerbeabfall?"
        logger.info(f"Testing query: {test_query}")
        
        # Test with very low threshold
        logger.info(f"\n--- TESTING WITH THRESHOLD 0.1 ---")
        result = await vector_repo.search_similar_text(
            query=test_query,
            limit=5,
            threshold=0.1
        )
        
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result items: {len(result.items)}")
        logger.info(f"Result total_count: {result.total_count}")
        
        for i, item in enumerate(result.items):
            logger.info(f"Item {i+1}:")
            logger.info(f"  Document ID: {item.document_id}")
            logger.info(f"  Chunk ID: {item.id}")
            logger.info(f"  Similarity: {item.metadata.get('similarity_score', 'N/A')}")
            logger.info(f"  Content: {item.text_content[:100]}...")
        
        # Test with default threshold (0.7)
        logger.info(f"\n--- TESTING WITH DEFAULT THRESHOLD 0.7 ---")
        result2 = await vector_repo.search_similar_text(
            query=test_query,
            limit=5,
            threshold=0.7
        )
        
        logger.info(f"Result with 0.7 threshold: {len(result2.items)} items")
        
        return True
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_search_similar_text())
    sys.exit(0 if success else 1)