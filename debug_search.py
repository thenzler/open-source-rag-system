#!/usr/bin/env python3
"""
Debug script to isolate the SearchResult issue
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    print("Testing imports...")
    
    # Test importing the repository directly
    from core.repositories.vector_repository import ProductionVectorRepository
    print("OK ProductionVectorRepository imported")
    
    # Test importing the base classes
    from core.repositories.base import QueryResult
    print("OK QueryResult imported")
    
    # Test importing models
    from core.repositories.models import DocumentChunk
    print("OK DocumentChunk imported")
    
    # Try to create the repository
    repo = ProductionVectorRepository()
    print("OK Repository created")
    
    # Test the specific method that's failing
    import asyncio
    async def test_search():
        try:
            result = await repo.search_similar_text("test query", limit=5)
            print(f"OK search_similar_text returned: {type(result)}")
            print(f"OK Result has {len(result.items)} items")
            return True
        except Exception as e:
            print(f"ERROR search_similar_text failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Run the test
    success = asyncio.run(test_search())
    if success:
        print("SUCCESS: Vector search is working!")
    else:
        print("FAILED: Vector search has issues")
        
except Exception as e:
    print(f"Error during import/test: {e}")
    import traceback
    traceback.print_exc()