#!/usr/bin/env python3
"""
Debug context preparation
"""

import sys
sys.path.append("C:/Users/THE/open-source-rag-system")

def debug_context():
    print("=== CONTEXT DEBUG ===")
    
    # Import the search function
    try:
        # We need to import these from the server
        import importlib.util
        spec = importlib.util.spec_from_file_location("simple_api", "C:/Users/THE/open-source-rag-system/simple_api.py")
        simple_api = importlib.util.module_from_spec(spec)
        
        # Mock minimal initialization
        simple_api.embedding_model = True  # Mock for testing
        simple_api.documents = []
        simple_api.document_chunks = []
        simple_api.embeddings = []
        
        spec.loader.exec_module(simple_api)
        
        print("OK: Imported server functions")
        
        # Test finding similar chunks
        print("Testing find_similar_chunks...")
        similar_chunks = simple_api.find_similar_chunks("test", 5)
        print(f"Found {len(similar_chunks)} chunks")
        
        if similar_chunks:
            print(f"First chunk keys: {list(similar_chunks[0].keys())}")
            print(f"First chunk has 'text': {'text' in similar_chunks[0]}")
            
            # Test context preparation
            print("Testing prepare_context_for_llm...")
            context = simple_api.prepare_context_for_llm(similar_chunks)
            print(f"Context length: {len(context)}")
            print(f"Context preview: {context[:200]}...")
        else:
            print("No chunks found - this explains the fallback!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_context()