#!/usr/bin/env python3
"""Debug script to check storage status"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Test memory-safe storage directly
    from services.memory_safe_storage import get_memory_safe_storage
    
    storage = get_memory_safe_storage()
    stats = storage.get_stats()
    
    print("=== Memory-Safe Storage Debug ===")
    print(f"Storage mode: {stats['storage_mode']}")
    print(f"Documents: {stats['documents']}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Embeddings: {stats['embeddings']}")
    print(f"Capacity: {stats['capacity_documents']}")
    
    if stats['documents'] > 0:
        print("\n=== Documents Found ===")
        docs = storage.get_all_documents()
        for doc in docs:
            print(f"- ID: {doc['id']}, Name: {doc['filename']}, Chunks: {doc['chunk_count']}")
    else:
        print("\n❌ No documents found in memory-safe storage")
    
    # Test the has_documents function
    print("\n=== Testing has_documents() function ===")
    
    # Import the function from the API
    import importlib.util
    spec = importlib.util.spec_from_file_location("simple_api", "simple_api.py")
    api_module = importlib.util.module_from_spec(spec)
    
    # We need to set up the module's globals first
    api_module.memory_safe_storage = storage
    
    # Define the function locally since we can't import it easily
    def has_documents():
        if storage:
            try:
                stats = storage.get_stats()
                return stats.get('documents', 0) > 0
            except:
                pass
        return False
    
    result = has_documents()
    print(f"has_documents() returns: {result}")
    
    if not result and stats['documents'] > 0:
        print("❌ BUG FOUND: has_documents() returns False but storage has documents!")
    elif result and stats['documents'] == 0:
        print("❌ BUG FOUND: has_documents() returns True but storage is empty!")
    else:
        print("✅ has_documents() function working correctly")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()