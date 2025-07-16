#!/usr/bin/env python3
"""Debug upload process step by step"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=== Upload Process Debug ===")

# Step 1: Check if memory-safe storage can be imported and initialized
try:
    from services.memory_safe_storage import get_memory_safe_storage
    storage = get_memory_safe_storage()
    print("[OK] Memory-safe storage imported and initialized")
    
    stats = storage.get_stats()
    print(f"   Initial state: {stats['documents']} docs, {stats['chunks']} chunks, {stats['embeddings']} embeddings")
except Exception as e:
    print(f"[ERROR] Memory-safe storage failed: {e}")
    sys.exit(1)

# Step 2: Test creating a simple document
try:
    test_chunks = ["This is test chunk 1", "This is test chunk 2"]
    
    # Create fake embeddings (384 dimensions like sentence-transformers)
    import numpy as np
    test_embeddings = [
        np.random.rand(384).astype(np.float32),
        np.random.rand(384).astype(np.float32)
    ]
    
    print(f"[OK] Created test data: {len(test_chunks)} chunks, {len(test_embeddings)} embeddings")
    print(f"   Embedding types: {[type(emb) for emb in test_embeddings]}")
    print(f"   Embedding shapes: {[emb.shape for emb in test_embeddings]}")
    
    # Try adding to storage
    doc_id = storage.add_document(
        filename="test_debug.txt",
        chunks=test_chunks,
        embeddings=test_embeddings,
        metadata={"test": "debug"}
    )
    
    print(f"[OK] Document added successfully with ID: {doc_id}")
    
    # Check stats after adding
    stats_after = storage.get_stats()
    print(f"   After adding: {stats_after['documents']} docs, {stats_after['chunks']} chunks, {stats_after['embeddings']} embeddings")
    
    if stats_after['documents'] == 0:
        print("[ERROR] BUG: Document was added but stats show 0 documents!")
    else:
        print("[OK] Document appears in stats correctly")
        
    # Test search function
    test_query_embedding = np.random.rand(384).astype(np.float32)
    search_results = storage.search(test_query_embedding, top_k=5, min_similarity=0.0)
    
    print(f"[OK] Search test: found {len(search_results)} results")
    
    if len(search_results) == 0:
        print("[ERROR] BUG: Search returns no results despite documents in storage!")
    else:
        print("[OK] Search working correctly")
        for i, result in enumerate(search_results):
            print(f"   Result {i}: similarity={result['similarity']:.3f}, filename={result['filename']}")

except Exception as e:
    print(f"[ERROR] Test document creation failed: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Test the has_documents function logic manually
print("\n=== Testing has_documents logic ===")

def test_has_documents():
    """Replicate the has_documents function logic"""
    if storage:
        try:
            stats = storage.get_stats()
            result = stats.get('documents', 0) > 0
            print(f"   Memory-safe storage stats: {stats['documents']} documents")
            print(f"   has_documents would return: {result}")
            return result
        except Exception as e:
            print(f"   Exception in memory-safe storage check: {e}")
            pass
    
    # Fallback would check legacy storage (which we don't have here)
    print("   Would fall back to legacy storage (empty in this test)")
    return False

has_docs_result = test_has_documents()
actual_docs = storage.get_stats()['documents']

if has_docs_result != (actual_docs > 0):
    print(f"[ERROR] BUG: has_documents() returns {has_docs_result} but actual docs: {actual_docs}")
else:
    print(f"[OK] has_documents() logic working correctly: {has_docs_result}")

print("\n=== Debug Complete ===")