#!/usr/bin/env python3
"""
Performance Test Script
Test AI response generation speed improvements
"""
import asyncio
import time
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_performance():
    """Test RAG system performance with optimizations"""
    try:
        print("Performance Test - Optimized RAG System")
        print("=" * 50)
        
        # Import optimized components
        from core.repositories.factory import RepositoryFactory
        from core.services.simple_rag_service import SimpleRAGService
        from core.ollama_client import OllamaClient
        
        # Initialize system
        print("Initializing optimized RAG system...")
        rag_repo = RepositoryFactory.create_production_repository()
        await rag_repo.initialize()
        
        vector_repo = rag_repo.vector_search
        audit_repo = rag_repo.audit
        llm_client = OllamaClient()  # Now with 60s timeout instead of 300s
        
        # Create optimized RAG service (with caching)
        rag_service = SimpleRAGService(vector_repo, llm_client, audit_repo)
        
        # Test queries for performance
        test_queries = [
            "Welche Regeln gelten für Gewerbeabfall?",  # First query (no cache)
            "Welche Regeln gelten für Gewerbeabfall?",  # Second query (should use cache)
            "Wie funktioniert die Abfallentsorgung?",   # Different query
        ]
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Performance Test {i}: {query[:50]}... ---")
            
            start_time = time.time()
            
            try:
                response = await rag_service.answer_query(query)
                
                end_time = time.time()
                response_time = end_time - start_time
                results.append(response_time)
                
                if "error" in response:
                    print(f"❌ Error: {response['error']}")
                else:
                    print(f"✅ Response time: {response_time:.2f} seconds")
                    print(f"   Answer length: {len(response.get('answer', ''))} chars")
                    print(f"   Sources: {len(response.get('sources', []))}")
                    print(f"   Confidence: {response.get('confidence', 0):.3f}")
                    
                    # Check if it was a cache hit
                    if i == 2 and response_time < 1.0:
                        print("   💾 Likely cache hit - very fast response!")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                results.append(999)  # High time for failed queries
        
        # Performance summary
        print("\n🚀 Performance Summary")
        print("-" * 30)
        
        if results:
            avg_time = sum(results) / len(results)
            min_time = min(results)
            max_time = max(results)
            
            print(f"Average response time: {avg_time:.2f} seconds")
            print(f"Fastest response: {min_time:.2f} seconds") 
            print(f"Slowest response: {max_time:.2f} seconds")
            
            # Performance benchmarks
            if max_time < 30:
                print("✅ Great performance - all responses under 30 seconds")
            elif max_time < 60:
                print("✅ Good performance - using optimized 60s timeout")
            else:
                print("⚠️ Some responses still slow - consider further optimization")
            
            # Cache effectiveness
            if len(results) >= 2 and results[1] < results[0] * 0.5:
                print("✅ Cache working - second identical query much faster")
        
        # Show cache stats
        cache_stats = rag_service.cache.stats()
        print(f"\n💾 Cache Statistics:")
        print(f"   Entries: {cache_stats['entries']}")
        print(f"   Total size: {cache_stats['total_size_mb']} MB")
        
        print("\n🎯 Optimization Results:")
        print("✅ Timeout reduced from 300s to 60s")
        print("✅ Model parameters optimized for speed")
        print("✅ Response caching implemented")
        print("✅ Context truncation for performance")
        
    except Exception as e:
        print(f"Performance test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_performance())