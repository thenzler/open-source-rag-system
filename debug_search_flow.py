#!/usr/bin/env python3
"""Debug the exact search flow step by step"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=== Testing Search Flow Step by Step ===")

try:
    # 1. Test memory-safe storage directly
    from services.memory_safe_storage import get_memory_safe_storage
    storage = get_memory_safe_storage()
    
    print(f"1. Storage stats: {storage.get_stats()}")
    
    # 2. Test embedding model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Test search directly
    query = "Was kommt in die Biotonne"
    query_embedding = model.encode([query])[0]
    
    print(f"2. Query: '{query}'")
    print(f"3. Query embedding shape: {query_embedding.shape}")
    
    # 4. Direct search
    results = storage.search(query_embedding, top_k=5, min_similarity=0.0)
    
    print(f"4. Direct search results: {len(results)} found")
    for i, result in enumerate(results[:3]):
        print(f"   Result {i}: similarity={result['similarity']:.4f}")
        print(f"               text={result['text'][:100]}...")
        print(f"               filename={result['filename']}")
        print()
    
    # 5. Test find_similar_chunks function
    print("5. Testing find_similar_chunks function...")
    
    # Import the API module
    import importlib.util
    spec = importlib.util.spec_from_file_location("simple_api", "C:/Users/THE/open-source-rag-system/simple_api.py")
    
    # This will be tricky since the API has many dependencies
    # Let's just test the core logic manually
    
    print("=== DIAGNOSIS ===")
    if len(results) > 0:
        max_sim = max(r['similarity'] for r in results)
        print(f"Max similarity: {max_sim:.4f}")
        
        if max_sim > 0.2:
            print("✅ Search is working and should return results!")
            print("❌ Problem is likely in the endpoint formatting/filtering")
        else:
            print("❌ Similarity too low - need better embeddings or documents")
    else:
        print("❌ No results found - storage or embedding problem")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()