#!/usr/bin/env python3
"""
Debug vector search functionality
"""
import asyncio
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def debug_search():
    try:
        from core.di.services import ServiceConfiguration, get_container
        from core.repositories.interfaces import IVectorSearchRepository
        from sentence_transformers import SentenceTransformer
        
        # Configure DI
        ServiceConfiguration.configure_all()
        container = get_container()
        
        # Get vector repository
        vector_repo = container.get(IVectorSearchRepository)
        
        # Check if index is ready
        is_ready = await vector_repo.is_index_ready()
        print(f"Vector index ready: {is_ready}")
        
        # Get index stats
        stats = await vector_repo.get_index_statistics()
        print(f"Index statistics: {stats}")
        
        # Generate a test query embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query = "Recycling Altglas"
        query_embedding = model.encode([query])[0]
        print(f"Query embedding shape: {query_embedding.shape}")
        
        # Try direct vector search
        print("\nTrying direct vector search...")
        results = await vector_repo.search_similar(
            query_vector=query_embedding.tolist(),
            top_k=5
        )
        print(f"Direct search results: {len(results)} found")
        for embedding_id, score in results[:3]:
            print(f"  Embedding {embedding_id}: score {score:.3f}")
        
        # Try text search
        print("\nTrying text-based search...")
        from core.repositories.base import QueryResult
        text_results = await vector_repo.search_similar_text(
            query=query,
            limit=5,
            threshold=0.0  # Lower threshold to see any results
        )
        print(f"Text search results: {text_results.total_count} found")
        for chunk in text_results.items[:3]:
            print(f"  Chunk {chunk.id}: {chunk.text_content[:50]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_search())