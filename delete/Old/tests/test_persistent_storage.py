#!/usr/bin/env python3
"""
Test script for persistent storage
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from services.persistent_storage import get_persistent_storage
    
    print("Testing persistent storage...")
    
    # Initialize storage
    storage = get_persistent_storage('test_rag.db')
    
    # Get initial stats
    print("Initial stats:", storage.get_stats())
    
    # Test adding a document
    test_chunks = ["This is test chunk 1", "This is test chunk 2"]
    test_embeddings = [np.random.random(384).astype(np.float32) for _ in test_chunks]
    
    doc_id = storage.add_document(
        filename="test_doc.txt",
        chunks=test_chunks,
        embeddings=test_embeddings,
        metadata={"test": True}
    )
    
    print(f"Added document with ID: {doc_id}")
    
    # Get stats after adding
    print("Stats after adding:", storage.get_stats())
    
    # Test search
    query_embedding = np.random.random(384).astype(np.float32)
    results = storage.search(query_embedding, top_k=5, min_similarity=0.0)
    print(f"Search results: {len(results)} found")
    
    # Test document listing
    docs = storage.get_all_documents()
    print(f"Document list: {len(docs)} documents")
    
    print("✓ Persistent storage test completed successfully!")
    
except Exception as e:
    print(f"❌ Error testing persistent storage: {e}")
    import traceback
    traceback.print_exc()