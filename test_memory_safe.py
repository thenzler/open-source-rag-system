#!/usr/bin/env python3
"""Test script for memory-safe storage"""

try:
    from services.memory_safe_storage import get_memory_safe_storage
    
    storage = get_memory_safe_storage()
    print("[OK] Memory-safe storage imported successfully")
    
    stats = storage.get_stats()
    print(f"Storage stats: {stats}")
    
    # Test adding a document
    test_chunks = ["This is a test chunk.", "This is another test chunk."]
    doc_id = storage.add_document("test.txt", test_chunks)
    print(f"Added test document with ID: {doc_id}")
    
    # Test stats after adding
    stats = storage.get_stats()
    print(f"Updated stats: {stats}")
    
    print("\nMemory-safe storage is working correctly!")
    
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()